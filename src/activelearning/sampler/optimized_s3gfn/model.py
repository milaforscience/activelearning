"""Inference optimizations for the experimental S3-GFN model.

This module keeps the reference S3-GFN loss and trajectory semantics while
avoiding a second full policy forward for terminal fidelity actions and
reusing deterministic frozen-prior sequence probabilities.
"""

from __future__ import annotations

import inspect
import math
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from activelearning.sampler.s3gfn.model import (
    GeneratedSequences,
    S3GFNModel,
)
from activelearning.sampler.s3gfn.losses import (
    negative_replay_contrastive_loss,
    relative_trajectory_balance_loss,
    sequence_log_probabilities_from_logits,
)


@dataclass(frozen=True)
class _CompactPastKeyValues:
    """Compact snapshot for linear-attention generation caches."""

    layers: tuple[tuple[Tensor, Tensor], ...]
    sequence_length: int


class OptimizedS3GFNModel(S3GFNModel):
    """S3-GFN model with cached generation states and prior likelihoods.

    The model delegates RTB, fidelity, tokenization, and validation semantics
    to :class:`~activelearning.sampler.s3gfn.model.S3GFNModel`. During
    evaluation-mode generation it preserves Hugging Face's ``generate``
    implementation and reuses its incremental cache for the terminal hidden
    state. Training-mode generation uses the reference terminal forward because
    GP-MoLFormer redraws random features while training. The frozen prior cache
    is enabled only for deterministic prior evaluation.
    """

    _MAX_GENERATION_CACHE_BYTES = 64 * 1024 * 1024
    _COMPILE_MODES = {
        "eager",
        "default",
        "reduce-overhead",
        "max-autotune-no-cudagraphs",
        "max-autotune",
    }

    def __init__(
        self,
        policy: torch.nn.Module,
        prior: torch.nn.Module,
        tokenizer: Any,
        initial_log_z: float = 0.0,
        fidelity_head: torch.nn.Module | None = None,
        *,
        deterministic_eval: bool | None = True,
        prior_cache_enabled: bool = True,
        prior_cache_capacity: int = 8192,
        precision: Literal["fp32", "cuda_auto"] = "fp32",
        fixed_feature_maps: bool = False,
        parallel_cuda_rollout: bool = False,
        compile_mode: str = "eager",
        deferred_sync: bool = False,
        carried_prior_scores: bool = False,
        overlap_online_prior: bool = False,
        combined_aux_policy_batch: bool = False,
        stop_check_interval: int = 1,
    ) -> None:
        """Create an optimized S3-GFN model.

        Parameters
        ----------
        policy : torch.nn.Module
            Trainable causal language model.
        prior : torch.nn.Module
            Frozen causal language model used as the RTB reference.
        tokenizer : Any
            Right-padded tokenizer with distinct pad and EOS ids.
        initial_log_z : float, optional
            Initial RTB normalizer.
        fidelity_head : torch.nn.Module or None, optional
            Terminal fidelity-action head.
        prior_cache_enabled : bool, optional
            Whether deterministic prior sequence probabilities may be reused.
        prior_cache_capacity : int, optional
            Maximum number of sequence probabilities retained in memory.

        Raises
        ------
        ValueError
            If ``prior_cache_capacity`` is not positive.
        """
        if prior_cache_capacity <= 0:
            raise ValueError("prior_cache_capacity must be positive.")
        if precision not in {"fp32", "cuda_auto"}:
            raise ValueError("precision must be 'fp32' or 'cuda_auto'.")
        if compile_mode not in self._COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {sorted(self._COMPILE_MODES)}."
            )
        if stop_check_interval < 0:
            raise ValueError("stop_check_interval must be nonnegative.")
        super().__init__(
            policy=policy,
            prior=prior,
            tokenizer=tokenizer,
            initial_log_z=initial_log_z,
            fidelity_head=fidelity_head,
        )
        self.prior_cache_enabled = bool(
            prior_cache_enabled and deterministic_eval is True
        )
        self.prior_cache_capacity = prior_cache_capacity
        self.deterministic_eval = deterministic_eval
        self.precision = precision
        self.fixed_feature_maps = bool(fixed_feature_maps)
        self.parallel_cuda_rollout = bool(parallel_cuda_rollout)
        self.compile_mode = compile_mode
        self.deferred_sync = bool(deferred_sync)
        self.carried_prior_scores = bool(
            carried_prior_scores and deterministic_eval is True
        )
        self.overlap_online_prior = bool(overlap_online_prior)
        self.combined_aux_policy_batch = bool(combined_aux_policy_batch)
        self.stop_check_interval = int(stop_check_interval)
        self._prior_sequence_cache: OrderedDict[tuple[int, ...], Tensor] = OrderedDict()
        self.prior_cache_hits = 0
        self.prior_cache_misses = 0
        self.feature_map_redraw_count = 0
        self.compile_cold_start_s: float | None = None
        self.compile_graph_count = 0
        self.compile_graph_breaks = 0
        self.compile_recompilations = 0
        self.compile_fallback_reason: str | None = None
        self._compiled_rollout_transition: Any | None = None
        self._compile_transition_failed = False
        self._compiled_policy_regions: dict[tuple[int, bool], Any] = {}
        self._failed_policy_regions: set[tuple[int, bool]] = set()
        self._compiled_decode_policy: Any | None = None
        self._decode_compile_failed = False
        self._feature_map_modules = tuple(self._find_feature_map_modules())
        self._feature_map_weight_ids = {
            id(module): id(getattr(module, "weight", None))
            for module in self._feature_map_modules
        }
        if self.fixed_feature_maps:
            self._install_fixed_feature_map_hooks()

    def _apply(self, fn: Any) -> "OptimizedS3GFNModel":
        """Move or cast the module and invalidate device/dtype-bound scores."""
        result = super()._apply(fn)
        if hasattr(self, "_prior_sequence_cache"):
            self.clear_prior_cache()
        return result

    def train(self, mode: bool = True) -> "OptimizedS3GFNModel":
        """Set training mode while keeping selected feature maps deterministic."""
        super().train(mode)
        if self.fixed_feature_maps:
            self._set_feature_maps_eval()
        return self

    @property
    def precision_dtype(self) -> torch.dtype | None:
        """Return the selected CUDA autocast dtype, or ``None`` when inactive."""
        if self.precision != "cuda_auto" or self.device.type != "cuda":
            return None
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @property
    def uses_grad_scaler(self) -> bool:
        """Return whether CUDA autocast needs loss scaling."""
        return self.precision_dtype is torch.float16

    def autocast_context(self) -> Any:
        """Return the configured autocast context for model forwards."""
        dtype = self.precision_dtype
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _find_feature_map_modules(self) -> list[torch.nn.Module]:
        """Find GP-MoLFormer random-feature modules without importing remote code."""
        return [
            module
            for module in self.policy.modules()
            if type(module).__name__ == "MolformerFeatureMap"
        ]

    def _enable_fixed_feature_maps(self) -> None:
        """Install feature-map guards after loading a configured model."""
        if not self.fixed_feature_maps:
            return
        if not self._feature_map_modules:
            self._feature_map_modules = tuple(self._find_feature_map_modules())
            self._feature_map_weight_ids = {
                id(module): id(getattr(module, "weight", None))
                for module in self._feature_map_modules
            }
        self._install_fixed_feature_map_hooks()
        self._set_feature_maps_eval()

    def _install_fixed_feature_map_hooks(self) -> None:
        """Install idempotent hooks that preserve trainable policy behavior."""
        if getattr(self, "_fixed_feature_map_hooks_installed", False):
            return
        self.policy.register_forward_pre_hook(
            self._feature_map_guard,
            with_kwargs=True,
        )
        for module in self._feature_map_modules:
            module.register_forward_hook(
                self._track_feature_map_redraw,
                with_kwargs=True,
            )
        self._fixed_feature_map_hooks_installed = True

    def _set_feature_maps_eval(self) -> None:
        """Keep only random-feature modules in evaluation mode."""
        for module in self._feature_map_modules:
            module.train(False)

    def _feature_map_guard(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Restore feature-map evaluation mode before each policy forward."""
        del module, args, kwargs
        self._set_feature_maps_eval()

    def _track_feature_map_redraw(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        """Count projection-buffer replacement performed by remote model code."""
        del args, kwargs, output
        module_id = id(module)
        current_weight_id = id(getattr(module, "weight", None))
        previous_weight_id = self._feature_map_weight_ids.get(module_id)
        if previous_weight_id is not None and current_weight_id != previous_weight_id:
            self.feature_map_redraw_count += 1
        self._feature_map_weight_ids[module_id] = current_weight_id

    @classmethod
    def from_pretrained(
        cls,
        policy_model_name_or_path: str = "ibm-research/GP-MoLFormer-Uniq",
        prior_model_name_or_path: str | None = None,
        tokenizer_name_or_path: str = "ibm-research/MoLFormer-XL-both-10pct",
        *,
        trust_remote_code: bool = False,
        cache_dir: str | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        initial_log_z: float = 0.0,
        n_fidelities: int | None = None,
        deterministic_eval: bool | None = True,
        prior_cache_enabled: bool = True,
        prior_cache_capacity: int = 8192,
        precision: Literal["fp32", "cuda_auto"] = "fp32",
        fixed_feature_maps: bool = False,
        parallel_cuda_rollout: bool = False,
        compile_mode: str = "eager",
        deferred_sync: bool = False,
        carried_prior_scores: bool = False,
        overlap_online_prior: bool = False,
        combined_aux_policy_batch: bool = False,
        stop_check_interval: int = 1,
    ) -> "OptimizedS3GFNModel":
        """Load an optimized model while preserving reference load behavior.

        Prior caching is automatically disabled unless
        ``deterministic_eval is True``. This prevents cached likelihoods from
        changing behavior for checkpoints that intentionally redraw random
        features between evaluations.
        """
        if prior_cache_capacity <= 0:
            raise ValueError("prior_cache_capacity must be positive.")
        if precision not in {"fp32", "cuda_auto"}:
            raise ValueError("precision must be 'fp32' or 'cuda_auto'.")
        if compile_mode not in cls._COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {sorted(cls._COMPILE_MODES)}."
            )
        if stop_check_interval < 0:
            raise ValueError("stop_check_interval must be nonnegative.")
        model = super().from_pretrained(
            policy_model_name_or_path=policy_model_name_or_path,
            prior_model_name_or_path=prior_model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
            initial_log_z=initial_log_z,
            n_fidelities=n_fidelities,
            deterministic_eval=deterministic_eval,
        )
        model.prior_cache_enabled = bool(
            prior_cache_enabled and deterministic_eval is True
        )
        model.prior_cache_capacity = prior_cache_capacity
        model.deterministic_eval = deterministic_eval
        model.precision = precision
        model.fixed_feature_maps = bool(fixed_feature_maps)
        model.parallel_cuda_rollout = bool(parallel_cuda_rollout)
        model.compile_mode = compile_mode
        model.deferred_sync = bool(deferred_sync)
        model.carried_prior_scores = bool(carried_prior_scores)
        model.overlap_online_prior = bool(overlap_online_prior)
        model.combined_aux_policy_batch = bool(combined_aux_policy_batch)
        model.stop_check_interval = int(stop_check_interval)
        if model.fixed_feature_maps:
            model._enable_fixed_feature_maps()
        return model

    def clear_prior_cache(self) -> None:
        """Remove all cached frozen-prior sequence probabilities."""
        self._prior_sequence_cache.clear()

    def prior_sequence_log_probabilities(self, input_ids: Tensor) -> Tensor:
        """Compute prior probabilities, reusing deterministic sequence results.

        Cache keys remove only trailing padding, so replay batches padded to
        different widths still hit the same entry without collapsing an
        internally sampled padding token. Cache misses are de-duplicated and
        evaluated in one prior forward.
        """
        if not self.prior_cache_enabled:
            with self.autocast_context():
                return super().prior_sequence_log_probabilities(input_ids)

        input_ids = input_ids.to(self.device)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence_length).")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token ids.")
        if input_ids.shape[0] == 0 or input_ids.shape[1] < 2:
            with self.autocast_context():
                return super().prior_sequence_log_probabilities(input_ids)

        keys = self._prior_cache_keys(input_ids)
        missing_keys: list[tuple[int, ...]] = []
        missing_indices: list[int] = []
        seen_missing: set[tuple[int, ...]] = set()
        resolved_values: dict[tuple[int, ...], Tensor] = {}
        for index, key in enumerate(keys):
            if key in self._prior_sequence_cache:
                self.prior_cache_hits += 1
                self._prior_sequence_cache.move_to_end(key)
                resolved_values[key] = self._prior_sequence_cache[key]
            elif key not in seen_missing:
                seen_missing.add(key)
                missing_keys.append(key)
                missing_indices.append(index)
                self.prior_cache_misses += 1
            else:
                self.prior_cache_hits += 1

        if missing_indices:
            missing_input_ids = input_ids[missing_indices]
            with self.autocast_context():
                missing_values = super().prior_sequence_log_probabilities(
                    missing_input_ids
                )
            for key, value in zip(missing_keys, missing_values, strict=True):
                self._store_prior_value(key, value)
                resolved_values[key] = self._prior_sequence_cache[key]

        return torch.stack(
            [resolved_values[key] for key in keys],
        ).to(device=self.device, dtype=self.log_z.dtype)

    @torch.no_grad()
    def prior_sequence_log_probabilities_uncached(self, input_ids: Tensor) -> Tensor:
        """Evaluate the frozen prior without constructing Python cache keys."""
        with self.autocast_context():
            values = super().prior_sequence_log_probabilities(input_ids)
        return values.to(device=self.device, dtype=self.log_z.dtype).detach()

    def policy_sequence_log_probabilities(self, input_ids: Tensor) -> Tensor:
        """Score policy sequences under the configured CUDA autocast context."""
        input_ids = input_ids.to(self.device)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence_length).")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token ids.")
        if input_ids.shape[0] == 0 or input_ids.shape[1] < 2:
            return torch.zeros(
                input_ids.shape[0],
                device=self.device,
                dtype=self.log_z.dtype,
            )
        with self.autocast_context():
            logits = self._policy_forward_region(
                input_ids=input_ids[:, :-1],
                attention_mask=input_ids[:, :-1].ne(self.pad_token_id).long(),
                need_hidden_states=False,
            )
            values = sequence_log_probabilities_from_logits(
                logits,
                labels=input_ids[:, 1:],
                pad_token_id=self.pad_token_id,
            )
        return values.to(dtype=self.log_z.dtype)

    def _policy_sequence_log_probabilities_and_terminal_hidden_states(
        self,
        input_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Score full sequences with a compiled or eager policy region."""
        input_ids = input_ids.to(self.device)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence_length).")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token ids.")
        if input_ids.shape[1] < 2:
            raise ValueError("input_ids must contain at least two tokens.")
        attention_mask = input_ids.ne(self.pad_token_id).long()
        with self.autocast_context():
            logits, hidden_states = self._policy_forward_region(
                input_ids=input_ids,
                attention_mask=attention_mask,
                need_hidden_states=True,
            )
            if logits.shape[:2] != input_ids.shape:
                raise ValueError(
                    "The causal LM logits must align with the shifted sequence labels."
                )
            sequence_scores = sequence_log_probabilities_from_logits(
                logits[:, :-1],
                labels=input_ids[:, 1:],
                pad_token_id=self.pad_token_id,
            )
            terminal_positions = attention_mask.sum(dim=1) - 1
            if bool((terminal_positions < 0).any()):
                raise ValueError(
                    "Terminal fidelity actions require at least one non-padding token."
                )
            batch_positions = torch.arange(
                input_ids.shape[0],
                device=input_ids.device,
            )
            terminal_states = hidden_states[
                batch_positions,
                terminal_positions,
            ]
        return sequence_scores.to(dtype=self.log_z.dtype), terminal_states

    def _policy_forward_region(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        need_hidden_states: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Run one bucketed trainable-policy forward region."""
        region_key = (int(input_ids.shape[1]), need_hidden_states)
        if (
            self.compile_mode == "eager"
            or self.device.type != "cuda"
            or region_key in self._failed_policy_regions
        ):
            return self._eager_policy_forward(
                input_ids,
                attention_mask,
                need_hidden_states,
            )
        compiled = self._compiled_policy_regions.get(region_key)
        if compiled is None:
            try:
                if need_hidden_states:

                    def forward_region(
                        ids: Tensor, mask: Tensor
                    ) -> tuple[Tensor, Tensor]:
                        outputs = self.policy(
                            input_ids=ids,
                            attention_mask=mask,
                            output_hidden_states=True,
                        )
                        hidden_states = getattr(outputs, "hidden_states", None)
                        if not hidden_states:
                            raise ValueError(
                                "The policy must return hidden states for terminal "
                                "fidelity actions."
                            )
                        return outputs.logits, hidden_states[-1]
                else:

                    def forward_region(ids: Tensor, mask: Tensor) -> Tensor:
                        return self.policy(
                            input_ids=ids,
                            attention_mask=mask,
                        ).logits

                compiled = torch.compile(
                    forward_region,
                    fullgraph=False,
                    mode=self.compile_mode,
                )
                self._compiled_policy_regions[region_key] = compiled
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                self.compile_fallback_reason = str(error)
                self.compile_graph_breaks += 1
                self._failed_policy_regions.add(region_key)
                return self._eager_policy_forward(
                    input_ids,
                    attention_mask,
                    need_hidden_states,
                )
        started = time.perf_counter()
        try:
            result = compiled(input_ids, attention_mask)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            self.compile_fallback_reason = str(error)
            self.compile_graph_breaks += 1
            self._failed_policy_regions.add(region_key)
            return self._eager_policy_forward(
                input_ids,
                attention_mask,
                need_hidden_states,
            )
        if self.compile_cold_start_s is None:
            self.compile_cold_start_s = time.perf_counter() - started
            self.compile_graph_count = 1
        elif self.compile_graph_count < len(self._compiled_policy_regions):
            self.compile_graph_count = len(self._compiled_policy_regions)
            self.compile_recompilations += 1
        return result

    def _eager_policy_forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        need_hidden_states: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Run the same region eagerly when compilation is unavailable."""
        if not need_hidden_states:
            return self.policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError(
                "The policy must return hidden states for terminal fidelity actions."
            )
        return outputs.logits, hidden_states[-1]

    def prior_trajectory_log_probabilities(
        self,
        input_ids: Tensor,
        fidelity_indices: Tensor | None = None,
        *,
        precomputed_sequence_log_probabilities: Tensor | None = None,
    ) -> Tensor:
        """Score trajectories, optionally reusing carried sequence priors."""
        if precomputed_sequence_log_probabilities is None:
            with self.autocast_context():
                return (
                    super()
                    .prior_trajectory_log_probabilities(
                        input_ids,
                        fidelity_indices=fidelity_indices,
                    )
                    .to(dtype=self.log_z.dtype)
                )

        input_ids = input_ids.to(self.device)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence_length).")
        sequence_scores = precomputed_sequence_log_probabilities.to(
            device=self.device,
            dtype=self.log_z.dtype,
        ).reshape(-1)
        if sequence_scores.shape[0] != input_ids.shape[0]:
            raise ValueError("Precomputed prior scores must align with input_ids.")
        finite_mask = torch.isfinite(sequence_scores)
        if not bool(finite_mask.all()):
            missing_scores = self.prior_sequence_log_probabilities_uncached(
                input_ids[~finite_mask]
            )
            sequence_scores = sequence_scores.clone()
            sequence_scores[~finite_mask] = missing_scores
        if self.fidelity_head is None:
            if fidelity_indices is not None:
                raise ValueError(
                    "Fidelity indices require a model with multiple fidelities."
                )
            return sequence_scores.detach()
        if fidelity_indices is None:
            raise ValueError(
                "Fidelity indices are required for a multi-fidelity trajectory."
            )
        return (
            sequence_scores
            + self.fidelity_head.uniform_prior_log_prob(
                fidelity_indices,
                batch_size=input_ids.shape[0],
                device=self.device,
                dtype=self.log_z.dtype,
            )
        ).detach()

    def policy_trajectory_log_probabilities(
        self,
        input_ids: Tensor,
        fidelity_indices: Tensor | None = None,
    ) -> Tensor:
        """Compute policy trajectory probabilities with optional autocast."""
        with self.autocast_context():
            values = super().policy_trajectory_log_probabilities(
                input_ids,
                fidelity_indices=fidelity_indices,
            )
        return values.to(dtype=self.log_z.dtype)

    def on_policy_loss(
        self,
        positive_input_ids: Tensor,
        reward_scores: Tensor,
        beta: float,
        fidelity_indices: Tensor | None = None,
        *,
        precomputed_sequence_log_probabilities: Tensor | None = None,
    ) -> Tensor | None:
        """Evaluate on-policy RTB, optionally using a carried prior score."""
        if precomputed_sequence_log_probabilities is None:
            with self.autocast_context():
                return super().on_policy_loss(
                    positive_input_ids,
                    reward_scores,
                    beta,
                    fidelity_indices=fidelity_indices,
                )
        if positive_input_ids.ndim != 2:
            raise ValueError(
                "positive_input_ids must have shape (batch, sequence_length)."
            )
        if positive_input_ids.shape[0] == 0:
            return None
        positive_input_ids = positive_input_ids.to(self.device)
        reward_scores = reward_scores.to(
            device=self.device,
            dtype=self.log_z.dtype,
        ).reshape(-1)
        if reward_scores.shape[0] != positive_input_ids.shape[0]:
            raise ValueError("Positive trajectories and reward scores must align.")
        policy_scores = self.policy_trajectory_log_probabilities(
            positive_input_ids,
            fidelity_indices=fidelity_indices,
        )
        prior_scores = self.prior_trajectory_log_probabilities(
            positive_input_ids,
            fidelity_indices=fidelity_indices,
            precomputed_sequence_log_probabilities=(
                precomputed_sequence_log_probabilities
            ),
        )
        return relative_trajectory_balance_loss(
            policy_log_probabilities=policy_scores,
            prior_log_probabilities=prior_scores,
            reward_scores=reward_scores,
            log_z=self.log_z.float(),
            beta=beta,
        )

    def replay_loss(
        self,
        positive_input_ids: Tensor,
        reward_scores: Tensor,
        beta: float,
        negative_input_ids: Tensor | None = None,
        aux_coefficient: float = 0.0,
        positive_fidelity_indices: Tensor | None = None,
        negative_fidelity_indices: Tensor | None = None,
        *,
        precomputed_positive_prior_log_probabilities: Tensor | None = None,
    ) -> Tensor | None:
        """Evaluate replay RTB and optionally batch positive/negative policy work."""
        if not math.isfinite(aux_coefficient) or aux_coefficient < 0.0:
            raise ValueError("aux_coefficient must be finite and nonnegative.")
        self._last_auxiliary_loss = None
        if positive_input_ids.ndim != 2:
            raise ValueError(
                "positive_input_ids must have shape (batch, sequence_length)."
            )
        if positive_input_ids.shape[0] == 0:
            return None

        positive_input_ids = positive_input_ids.to(self.device)
        reward_scores = reward_scores.to(
            device=self.device,
            dtype=self.log_z.dtype,
        ).reshape(-1)
        if positive_input_ids.shape[0] != reward_scores.numel():
            raise ValueError("Positive trajectories and reward scores must align.")

        negative_batch = (
            negative_input_ids is not None
            and negative_input_ids.ndim == 2
            and negative_input_ids.shape[0] > 0
            and aux_coefficient > 0.0
        )
        if negative_input_ids is not None and negative_input_ids.ndim != 2:
            raise ValueError(
                "negative_input_ids must have shape (batch, sequence_length)."
            )

        if self.combined_aux_policy_batch and negative_batch:
            assert negative_input_ids is not None
            combined, positive_width = self._pad_and_combine(
                positive_input_ids,
                negative_input_ids.to(self.device),
                pad_token_id=self.pad_token_id,
            )
            combined_policy_scores = self.policy_trajectory_log_probabilities(
                combined,
                fidelity_indices=self._combine_fidelity_indices(
                    positive_fidelity_indices,
                    negative_fidelity_indices,
                    positive_count=positive_input_ids.shape[0],
                    negative_count=negative_input_ids.shape[0],
                ),
            )
            positive_policy_scores = combined_policy_scores[:positive_width]
            negative_policy_scores = combined_policy_scores[positive_width:]
        else:
            positive_policy_scores = self.policy_trajectory_log_probabilities(
                positive_input_ids,
                fidelity_indices=positive_fidelity_indices,
            )
            negative_policy_scores = None
            if negative_batch:
                assert negative_input_ids is not None
                negative_policy_scores = self.policy_trajectory_log_probabilities(
                    negative_input_ids.to(self.device),
                    fidelity_indices=negative_fidelity_indices,
                )

        positive_prior_scores = self.prior_trajectory_log_probabilities(
            positive_input_ids,
            fidelity_indices=positive_fidelity_indices,
            precomputed_sequence_log_probabilities=(
                precomputed_positive_prior_log_probabilities
            ),
        )
        rtb_loss = relative_trajectory_balance_loss(
            policy_log_probabilities=positive_policy_scores,
            prior_log_probabilities=positive_prior_scores,
            reward_scores=reward_scores,
            log_z=self.log_z.float(),
            beta=beta,
        )
        if negative_policy_scores is not None:
            auxiliary_loss = negative_replay_contrastive_loss(
                positive_policy_scores,
                negative_policy_scores,
            )
            self._last_auxiliary_loss = auxiliary_loss.detach()
            return rtb_loss + aux_coefficient * auxiliary_loss
        return rtb_loss

    @staticmethod
    def _pad_and_combine(
        positive_input_ids: Tensor,
        negative_input_ids: Tensor,
        *,
        pad_token_id: int,
    ) -> tuple[Tensor, int]:
        """Right-pad replay trajectories and concatenate one policy batch."""
        width = max(positive_input_ids.shape[1], negative_input_ids.shape[1])
        positive = F.pad(
            positive_input_ids,
            (0, width - positive_input_ids.shape[1]),
            value=pad_token_id,
        )
        negative = F.pad(
            negative_input_ids,
            (0, width - negative_input_ids.shape[1]),
            value=pad_token_id,
        )
        return torch.cat((positive, negative), dim=0), positive_input_ids.shape[0]

    @staticmethod
    def _combine_fidelity_indices(
        positive_fidelity_indices: Tensor | None,
        negative_fidelity_indices: Tensor | None,
        *,
        positive_count: int,
        negative_count: int,
    ) -> Tensor | None:
        """Concatenate aligned fidelity actions for a combined replay batch."""
        if positive_fidelity_indices is None and negative_fidelity_indices is None:
            return None
        if positive_fidelity_indices is None or negative_fidelity_indices is None:
            raise ValueError("Combined replay batches must use one fidelity mode.")
        if positive_fidelity_indices.shape[0] != positive_count:
            raise ValueError("Positive fidelity indices must align with replay data.")
        if negative_fidelity_indices.shape[0] != negative_count:
            raise ValueError("Negative fidelity indices must align with replay data.")
        return torch.cat(
            (
                positive_fidelity_indices.to(dtype=torch.long),
                negative_fidelity_indices.to(dtype=torch.long),
            ),
            dim=0,
        )

    @torch.no_grad()
    def generate(
        self,
        count: int,
        max_length: int,
        temperature: float = 1.0,
        **generation_kwargs: Any,
    ) -> GeneratedSequences:
        """Generate sequences and reuse cached decoding states for fidelity.

        The policy's own ``generate`` method remains responsible for sampling,
        logits processors, stopping criteria, and RNG behavior. Hooks only
        retain the generation cache; a final one-token cached forward supplies
        terminal states without rewriting any sampled token. Unsupported model
        outputs fall back to the reference full terminal-state forward.
        """
        if count < 0:
            raise ValueError("count must be nonnegative.")
        if max_length < 2:
            raise ValueError("max_length must be at least two.")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and positive.")
        if count == 0:
            return GeneratedSequences(
                input_ids=torch.empty((0, 0), dtype=torch.long, device=self.device),
                smiles=(),
            )
        if self.fidelity_head is None:
            if (
                self.parallel_cuda_rollout
                and self.device.type == "cuda"
                and not self._requires_fallback_generation(generation_kwargs)
            ):
                generated = self._parallel_cuda_generate(
                    count=count,
                    max_length=max_length,
                    temperature=temperature,
                    generation_kwargs=generation_kwargs,
                )
                if generated is not None:
                    return generated
            with self.autocast_context():
                return super().generate(
                    count=count,
                    max_length=max_length,
                    temperature=temperature,
                    **generation_kwargs,
                )
        if (
            self.parallel_cuda_rollout
            and self.device.type == "cuda"
            and not self._requires_fallback_generation(generation_kwargs)
        ):
            generated = self._parallel_cuda_generate(
                count=count,
                max_length=max_length,
                temperature=temperature,
                generation_kwargs=generation_kwargs,
            )
            if generated is not None:
                return generated
        if self.policy.training:
            # GP-MoLFormer redraws its random feature map while training. Its
            # full reference terminal forward is therefore not equivalent to
            # a cached decoding state, even when deterministic_eval is true.
            with self.autocast_context():
                return super().generate(
                    count=count,
                    max_length=max_length,
                    temperature=temperature,
                    **generation_kwargs,
                )
        with self.autocast_context():
            generated_ids, terminal_hidden_states = (
                self._generate_and_capture_terminal_states(
                    count=count,
                    max_length=max_length,
                    temperature=temperature,
                    generation_kwargs=generation_kwargs,
                )
            )
        if terminal_hidden_states is None:
            terminal_hidden_states = self._terminal_hidden_states(generated_ids)
        fidelity_indices = self.fidelity_head.sample(
            terminal_hidden_states,
            temperature=temperature,
        )
        return GeneratedSequences(
            input_ids=generated_ids,
            smiles=tuple(
                self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            ),
            fidelity_indices=fidelity_indices,
        )

    @torch.no_grad()
    def _parallel_cuda_generate(
        self,
        *,
        count: int,
        max_length: int,
        temperature: float,
        generation_kwargs: dict[str, Any],
    ) -> GeneratedSequences | None:
        """Generate GP-MoLFormer trajectories with device-resident state."""
        if generation_kwargs or not self._supports_parallel_rollout():
            return None
        bos_token_id = getattr(
            getattr(self.policy, "config", None), "bos_token_id", None
        )
        if bos_token_id is None:
            return None
        sampling_parameters = self._parallel_sampling_parameters()
        if sampling_parameters is None:
            return None
        top_k, top_p = sampling_parameters

        with self.autocast_context():
            token_state = torch.full(
                (count, max_length),
                self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            token_state[:, 0] = int(bos_token_id)
            attention_state = torch.ones(
                (count, max_length),
                dtype=torch.long,
                device=self.device,
            )
            finished = torch.zeros(
                count,
                dtype=torch.bool,
                device=self.device,
            )
            past_key_values: Any | None = None
            cache_history: list[_CompactPastKeyValues] = []
            needs_terminal_states = self.fidelity_head is not None
            sequence_length = 1
            for position in range(max_length - 1):
                current_length = position + 1
                current_input_ids = (
                    token_state[:, :current_length]
                    if past_key_values is None
                    else token_state[:, position : position + 1]
                )
                outputs = self._decode_policy_forward(
                    input_ids=current_input_ids,
                    attention_mask=attention_state[:, :current_length],
                    past_key_values=past_key_values,
                )
                past_key_values = getattr(outputs, "past_key_values", None)
                if past_key_values is None:
                    return None
                if needs_terminal_states:
                    compact_cache = self._compact_past_key_values(
                        past_key_values,
                        verify_repeated=True,
                    )
                    if compact_cache is None:
                        return None
                    cache_history.append(compact_cache)
                logits = outputs.logits[:, -1].float() / temperature
                logits = self._apply_sampling_warpers(
                    logits,
                    top_k=top_k,
                    top_p=top_p,
                )
                next_tokens = torch.multinomial(
                    torch.softmax(logits, dim=-1),
                    num_samples=1,
                ).squeeze(-1)
                token_state, finished = self._rollout_transition(
                    token_state,
                    next_tokens,
                    finished,
                    position + 1,
                )
                sequence_length = position + 2
                if (
                    self.stop_check_interval > 0
                    and (position + 1) % self.stop_check_interval == 0
                    and bool(finished.all())
                ):
                    break

            terminal_hidden_states = None
            if needs_terminal_states:
                terminal_hidden_states = self._parallel_terminal_hidden_states(
                    token_state=token_state[:, :sequence_length],
                    cache_history=cache_history,
                )
                if terminal_hidden_states is None:
                    return None

            fidelity_indices = (
                None
                if terminal_hidden_states is None
                else self.fidelity_head.sample(
                    terminal_hidden_states,
                    temperature=temperature,
                )
            )
            return GeneratedSequences(
                input_ids=token_state[:, :sequence_length],
                smiles=tuple(
                    self.tokenizer.batch_decode(
                        token_state[:, :sequence_length],
                        skip_special_tokens=True,
                    )
                ),
                fidelity_indices=fidelity_indices,
            )

    def _decode_policy_forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        past_key_values: Any | None,
    ) -> Any:
        """Run one-token decoding through an optional compiled policy module."""
        arguments = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_hidden_states": False,
            "return_dict": True,
        }
        if (
            self.compile_mode == "eager"
            or self.device.type != "cuda"
            or self._decode_compile_failed
        ):
            return self.policy(**arguments)
        if self._compiled_decode_policy is None:
            try:
                self._compiled_decode_policy = torch.compile(
                    self.policy,
                    fullgraph=False,
                    mode=self.compile_mode,
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                self.compile_fallback_reason = str(error)
                self.compile_graph_breaks += 1
                self._decode_compile_failed = True
                return self.policy(**arguments)
        started = time.perf_counter()
        try:
            outputs = self._compiled_decode_policy(**arguments)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            self.compile_fallback_reason = str(error)
            self.compile_graph_breaks += 1
            self._decode_compile_failed = True
            return self.policy(**arguments)
        if self.compile_cold_start_s is None:
            self.compile_cold_start_s = time.perf_counter() - started
        self.compile_graph_count = max(
            self.compile_graph_count,
            len(self._compiled_policy_regions) + 1,
        )
        return outputs

    def _parallel_terminal_hidden_states(
        self,
        *,
        token_state: Tensor,
        cache_history: list[_CompactPastKeyValues],
    ) -> Tensor | None:
        """Finalize terminal hidden states from one cached snapshot per row."""
        if not cache_history:
            return None
        terminal_positions = token_state.ne(self.pad_token_id).sum(dim=1) - 1
        if bool((terminal_positions < 1).any()):
            return None
        positions = torch.arange(
            token_state.shape[1],
            device=token_state.device,
        )
        terminal_prefix = positions[None, :] <= terminal_positions[:, None]
        cache_safe = (~token_state.eq(self.pad_token_id) | ~terminal_prefix).all(dim=1)
        cached_rows = cache_safe.nonzero(as_tuple=False).flatten()
        fallback_rows = (~cache_safe).nonzero(as_tuple=False).flatten()
        hidden_batches: list[tuple[Tensor, Tensor]] = []

        if cached_rows.numel() > 0:
            for terminal_position in torch.unique(
                terminal_positions[cached_rows]
            ).tolist():
                terminal_position = int(terminal_position)
                rows = cached_rows[
                    terminal_positions[cached_rows].eq(terminal_position)
                ]
                history_index = terminal_position - 1
                if history_index < 0 or history_index >= len(cache_history):
                    fallback_rows = torch.cat((fallback_rows, rows))
                    continue
                hidden = self._cached_terminal_hidden_states(
                    input_ids=token_state[rows, terminal_position].unsqueeze(1),
                    attention_mask=torch.ones(
                        (rows.shape[0], terminal_position + 1),
                        dtype=torch.long,
                        device=token_state.device,
                    ),
                    unresolved_indices=rows,
                    past_key_values=cache_history[history_index],
                )
                if hidden is None:
                    fallback_rows = torch.cat((fallback_rows, rows))
                else:
                    hidden_batches.append((rows, hidden))

        terminal_hidden_states: Tensor | None = None
        if hidden_batches:
            terminal_hidden_states = torch.empty(
                (
                    token_state.shape[0],
                    hidden_batches[0][1].shape[-1],
                ),
                dtype=hidden_batches[0][1].dtype,
                device=hidden_batches[0][1].device,
            )
            for rows, hidden in hidden_batches:
                terminal_hidden_states[rows] = hidden

        if fallback_rows.numel() > 0:
            fallback_hidden = self._terminal_hidden_states(token_state[fallback_rows])
            if terminal_hidden_states is None:
                terminal_hidden_states = torch.empty(
                    (
                        token_state.shape[0],
                        fallback_hidden.shape[-1],
                    ),
                    dtype=fallback_hidden.dtype,
                    device=fallback_hidden.device,
                )
            terminal_hidden_states[fallback_rows] = fallback_hidden
        return terminal_hidden_states

    def _supports_parallel_rollout(self) -> bool:
        """Return whether the loaded policy is the supported linear-attention model."""
        return bool(self._feature_map_modules) and self._supports_hidden_state_capture()

    def _parallel_sampling_parameters(self) -> tuple[int, float] | None:
        """Return sampling settings supported by the device-resident loop."""
        generation_config = getattr(self.policy, "generation_config", None)
        if generation_config is None:
            return 0, 1.0
        unsupported_defaults = {
            "typical_p": 1.0,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
            "min_p": None,
            "repetition_penalty": 1.0,
            "encoder_repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "bad_words_ids": None,
            "renormalize_logits": False,
            "remove_invalid_values": False,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "constraints": None,
            "force_words_ids": None,
            "sequence_bias": None,
            "watermarking_config": None,
        }
        for name, default in unsupported_defaults.items():
            if getattr(generation_config, name, default) != default:
                return None
        for name in ("min_length", "min_new_tokens"):
            if getattr(generation_config, name, 0) not in (None, 0):
                return None
        configured_eos = getattr(generation_config, "eos_token_id", None)
        if configured_eos not in (None, self.eos_token_id):
            return None
        configured_pad = getattr(generation_config, "pad_token_id", None)
        if configured_pad not in (None, self.pad_token_id):
            return None
        top_k = getattr(generation_config, "top_k", 0)
        top_p = getattr(generation_config, "top_p", 1.0)
        if top_k is None:
            top_k = 0
        if top_p is None:
            top_p = 1.0
        if not isinstance(top_k, int) or top_k < 0:
            return None
        if not isinstance(top_p, (float, int)) or not 0.0 < top_p <= 1.0:
            return None
        return top_k, float(top_p)

    @staticmethod
    def _apply_sampling_warpers(
        logits: Tensor,
        *,
        top_k: int,
        top_p: float,
    ) -> Tensor:
        """Apply the common top-k/top-p warpers used by HF sampling."""
        if top_k > 0 and top_k < logits.shape[-1]:
            top_k = min(top_k, logits.shape[-1])
            threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
            remove = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities > top_p
            remove[..., 0] = False
            logits = logits.masked_fill(
                torch.zeros_like(logits, dtype=torch.bool).scatter(
                    dim=-1,
                    index=sorted_indices,
                    src=remove,
                ),
                -torch.inf,
            )
        return logits

    @staticmethod
    def _eager_rollout_transition(
        token_state: Tensor,
        next_tokens: Tensor,
        finished: Tensor,
        position: int,
        pad_token_id: int,
        eos_token_id: int,
    ) -> tuple[Tensor, Tensor]:
        """Apply one batched transition without creating a new token buffer."""
        sampled = torch.where(
            finished,
            torch.full_like(next_tokens, pad_token_id),
            next_tokens,
        )
        token_state[:, position] = sampled
        return token_state, finished | sampled.eq(eos_token_id)

    def _rollout_transition(
        self,
        token_state: Tensor,
        next_tokens: Tensor,
        finished: Tensor,
        position: int,
    ) -> tuple[Tensor, Tensor]:
        """Run an eager or compiled device-side transition."""
        if self.compile_mode == "eager" or self._compile_transition_failed:
            return self._eager_rollout_transition(
                token_state,
                next_tokens,
                finished,
                position,
                self.pad_token_id,
                self.eos_token_id,
            )
        if self.device.type != "cuda":
            return self._eager_rollout_transition(
                token_state,
                next_tokens,
                finished,
                position,
                self.pad_token_id,
                self.eos_token_id,
            )
        if self._compiled_rollout_transition is None:
            try:

                def transition(
                    state: torch.Tensor,
                    tokens: torch.Tensor,
                    done: torch.Tensor,
                    step: int,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    return self._eager_rollout_transition(
                        state,
                        tokens,
                        done,
                        step,
                        self.pad_token_id,
                        self.eos_token_id,
                    )

                self._compiled_rollout_transition = torch.compile(
                    transition,
                    fullgraph=True,
                    mode=self.compile_mode,
                )
            except (AttributeError, RuntimeError, TypeError) as error:
                self.compile_fallback_reason = str(error)
                self.compile_graph_breaks += 1
                self._compile_transition_failed = True
                return self._eager_rollout_transition(
                    token_state,
                    next_tokens,
                    finished,
                    position,
                    self.pad_token_id,
                    self.eos_token_id,
                )
        started = time.perf_counter()
        try:
            result = self._compiled_rollout_transition(
                token_state,
                next_tokens,
                finished,
                position,
            )
        except (AttributeError, RuntimeError, TypeError) as error:
            self.compile_fallback_reason = str(error)
            self.compile_graph_breaks += 1
            self._compile_transition_failed = True
            return self._eager_rollout_transition(
                token_state,
                next_tokens,
                finished,
                position,
                self.pad_token_id,
                self.eos_token_id,
            )
        if self.compile_cold_start_s is None:
            self.compile_cold_start_s = time.perf_counter() - started
            self.compile_graph_count = 1
        return result

    def _generate_and_capture_terminal_states(
        self,
        *,
        count: int,
        max_length: int,
        temperature: float,
        generation_kwargs: dict[str, Any],
    ) -> tuple[Tensor, Tensor | None]:
        """Run generation and capture terminal hidden states when possible."""
        if (
            not self._supports_hidden_state_capture()
            or self._requires_fallback_generation(
                generation_kwargs,
            )
        ):
            generated_ids = self.policy.generate(
                do_sample=True,
                max_length=max_length,
                num_return_sequences=count,
                temperature=temperature,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                **generation_kwargs,
            )
            return self._generation_sequences(generated_ids), None

        past_key_values_history: list[Any] = []
        compact_cache_supported: bool | None = None
        generation_history_complete = True

        def prepare_inputs(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            del module
            working_args = args
            working_kwargs = dict(kwargs)
            # Do not rewrite padding ids here: a sampling model may emit the
            # padding id for an unfinished row, and changing it changes RNG
            # behavior and subsequent logits.
            working_kwargs["return_dict"] = True
            return working_args, working_kwargs

        def capture_outputs(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            outputs: Any,
        ) -> None:
            del module, args, kwargs
            nonlocal compact_cache_supported, generation_history_complete
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                generation_history_complete = False
                past_key_values_history.clear()
                return
            if generation_history_complete:
                if compact_cache_supported is True:
                    compact_snapshot = self._compact_past_key_values(
                        past_key_values,
                        verify_repeated=False,
                    )
                    if compact_snapshot is not None:
                        past_key_values_history.append(compact_snapshot)
                    else:
                        compact_cache_supported = False
                        past_key_values_history.clear()
                elif compact_cache_supported is None:
                    compact_snapshot = self._compact_past_key_values(
                        past_key_values,
                    )
                    if (
                        compact_snapshot is not None
                        and compact_snapshot.sequence_length > 1
                    ):
                        compact_cache_supported = True
                        past_key_values_history[:] = [
                            snapshot
                            if isinstance(snapshot, _CompactPastKeyValues)
                            else self._compact_past_key_values(
                                snapshot,
                                verify_repeated=False,
                            )
                            for snapshot in past_key_values_history
                        ]
                        if any(
                            not isinstance(snapshot, _CompactPastKeyValues)
                            for snapshot in past_key_values_history
                        ):
                            compact_cache_supported = False
                            past_key_values_history.clear()
                            return
                        past_key_values_history.append(compact_snapshot)
                    else:
                        # The first cache has length one for both ordinary
                        # transformer and linear-attention implementations.
                        # Keep it until a later call identifies the format.
                        generation_history_complete = (
                            self._append_bounded_generation_cache(
                                past_key_values_history,
                                past_key_values,
                            )
                        )

        prepare_handle = self.policy.register_forward_pre_hook(
            prepare_inputs,
            with_kwargs=True,
        )
        capture_handle = self.policy.register_forward_hook(
            capture_outputs,
            with_kwargs=True,
        )
        try:
            generated = self.policy.generate(
                do_sample=True,
                max_length=max_length,
                num_return_sequences=count,
                temperature=temperature,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                **generation_kwargs,
            )
        finally:
            capture_handle.remove()
            prepare_handle.remove()

        generated_ids = self._generation_sequences(generated)
        if not generation_history_complete or not past_key_values_history:
            return generated_ids, None

        terminal_positions = generated_ids.ne(self.pad_token_id).sum(dim=1) - 1
        if bool((terminal_positions < 1).any()):
            return generated_ids, None

        # A padding id emitted before the reference terminal position is an
        # ordinary sampled token to ``generate`` but is masked by the
        # reference full forward. Use that exact reference path for those
        # uncommon rows instead of mixing incompatible cached semantics.
        positions = torch.arange(
            generated_ids.shape[1],
            device=generated_ids.device,
        )
        terminal_prefix_mask = positions[None, :] <= terminal_positions[:, None]
        cache_safe = (~generated_ids.eq(self.pad_token_id) | ~terminal_prefix_mask).all(
            dim=1
        )
        cached_rows = cache_safe.nonzero(as_tuple=False).flatten()
        fallback_rows = (~cache_safe).nonzero(as_tuple=False).flatten()

        terminal_hidden_states: Tensor | None = None
        if cached_rows.numel() > 0:
            hidden_batches: list[tuple[Tensor, Tensor]] = []
            initial_sequence_length = self._initial_generation_length(
                past_key_values_history,
            )
            batched_hidden_states = self._batched_compact_terminal_hidden_states(
                input_ids=generated_ids,
                terminal_positions=terminal_positions,
                rows=cached_rows,
                initial_sequence_length=initial_sequence_length,
                past_key_values_history=past_key_values_history,
            )
            if batched_hidden_states is not None:
                hidden_batches.append((cached_rows, batched_hidden_states))
                cached_rows = cached_rows[:0]
            for position in torch.unique(terminal_positions[cached_rows]).tolist():
                position = int(position)
                group_rows = cached_rows[terminal_positions[cached_rows].eq(position)]
                history_index = position - initial_sequence_length
                if history_index < 0 or history_index >= len(past_key_values_history):
                    fallback_rows = torch.cat((fallback_rows, group_rows))
                    continue
                history_past = past_key_values_history[history_index]
                terminal_tokens = generated_ids[group_rows, position].unsqueeze(1)
                terminal_attention_mask = (
                    generated_ids[group_rows, : position + 1]
                    .ne(self.pad_token_id)
                    .long()
                )
                hidden_batch = self._cached_terminal_hidden_states(
                    input_ids=terminal_tokens,
                    attention_mask=terminal_attention_mask,
                    unresolved_indices=group_rows,
                    past_key_values=history_past,
                )
                if hidden_batch is None:
                    fallback_rows = torch.cat((fallback_rows, group_rows))
                else:
                    hidden_batches.append((group_rows, hidden_batch))

            if hidden_batches:
                hidden_size = hidden_batches[0][1].shape[-1]
                terminal_hidden_states = torch.empty(
                    (generated_ids.shape[0], hidden_size),
                    dtype=hidden_batches[0][1].dtype,
                    device=hidden_batches[0][1].device,
                )
                for group_rows, hidden_batch in hidden_batches:
                    terminal_hidden_states[group_rows] = hidden_batch

        if fallback_rows.numel() > 0:
            fallback_hidden_states = self._terminal_hidden_states(
                generated_ids[fallback_rows]
            )
            if terminal_hidden_states is None:
                terminal_hidden_states = torch.empty(
                    (generated_ids.shape[0], fallback_hidden_states.shape[-1]),
                    dtype=fallback_hidden_states.dtype,
                    device=fallback_hidden_states.device,
                )
            terminal_hidden_states[fallback_rows] = fallback_hidden_states

        if terminal_hidden_states is None:
            return generated_ids, None
        return generated_ids, terminal_hidden_states

    def _batched_compact_terminal_hidden_states(
        self,
        *,
        input_ids: Tensor,
        terminal_positions: Tensor,
        rows: Tensor,
        initial_sequence_length: int,
        past_key_values_history: list[Any],
    ) -> Tensor | None:
        """Finalize all linear-attention terminal rows in one forward."""
        if not self._accepts_forward_argument("position_ids"):
            return None
        if not all(
            isinstance(snapshot, _CompactPastKeyValues)
            for snapshot in past_key_values_history
        ):
            return None

        history_indices = (terminal_positions[rows] - initial_sequence_length).tolist()
        if any(
            index < 0 or index >= len(past_key_values_history)
            for index in history_indices
        ):
            return None
        snapshots = [past_key_values_history[index] for index in history_indices]
        if not all(
            isinstance(snapshot, _CompactPastKeyValues) for snapshot in snapshots
        ):
            return None
        sequence_lengths = [snapshot.sequence_length for snapshot in snapshots]
        if not sequence_lengths:
            return None
        max_sequence_length = max(sequence_lengths)
        compact_layers: list[tuple[Tensor, Tensor]] = []
        for layer_index in range(len(snapshots[0].layers)):
            key_rows = torch.cat(
                [
                    snapshot.layers[layer_index][0].index_select(
                        0,
                        row.view(1).to(snapshot.layers[layer_index][0].device),
                    )
                    for row, snapshot in zip(rows, snapshots, strict=True)
                ],
                dim=0,
            )
            value_rows = torch.cat(
                [
                    snapshot.layers[layer_index][1].index_select(
                        0,
                        row.view(1).to(snapshot.layers[layer_index][1].device),
                    )
                    for row, snapshot in zip(rows, snapshots, strict=True)
                ],
                dim=0,
            )
            compact_layers.append(
                (
                    key_rows.expand(-1, -1, max_sequence_length, -1),
                    value_rows,
                )
            )

        terminal_tokens = input_ids[
            rows,
            terminal_positions[rows],
        ].unsqueeze(1)
        terminal_attention_mask = torch.ones(
            (rows.shape[0], max_sequence_length + 1),
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = terminal_positions[rows].unsqueeze(1)
        outputs = self.policy(
            input_ids=terminal_tokens,
            attention_mask=terminal_attention_mask,
            position_ids=position_ids,
            past_key_values=tuple(compact_layers),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        return self._last_hidden_state_from_outputs(outputs)

    def _accepts_forward_argument(self, name: str) -> bool:
        """Return whether the policy accepts a named forward argument."""
        try:
            signature = inspect.signature(self.policy.forward)
        except (TypeError, ValueError):
            return False
        return name in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    @staticmethod
    def _compact_past_key_values(
        past_key_values: Any,
        *,
        verify_repeated: bool = True,
    ) -> _CompactPastKeyValues | None:
        """Compact a GP-MoLFormer running-sum cache when its shape permits it."""
        if not isinstance(past_key_values, (tuple, list)) or not past_key_values:
            return None
        compact_layers: list[tuple[Tensor, Tensor]] = []
        sequence_length: int | None = None
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)) or len(layer) != 2:
                return None
            key_state, value_state = layer
            if (
                not isinstance(key_state, Tensor)
                or not isinstance(value_state, Tensor)
                or key_state.ndim != 4
                or value_state.ndim < 3
                or value_state.shape[2] != 1
            ):
                return None
            if sequence_length is None:
                sequence_length = int(key_state.shape[2])
            elif key_state.shape[2] != sequence_length:
                return None
            if (
                verify_repeated
                and key_state.shape[2] > 1
                and not torch.equal(
                    key_state,
                    key_state[..., -1:, :].expand_as(key_state),
                )
            ):
                return None
            compact_layers.append(
                (
                    key_state[..., -1:, :].detach().clone(),
                    value_state.detach().clone(),
                )
            )
        if sequence_length is None:
            return None
        return _CompactPastKeyValues(
            layers=tuple(compact_layers),
            sequence_length=sequence_length,
        )

    def _append_bounded_generation_cache(
        self,
        history: list[Any],
        past_key_values: Any,
    ) -> bool:
        """Retain ordinary KV caches only while their memory remains bounded."""
        retained_bytes = sum(
            self._past_key_values_nbytes(snapshot) for snapshot in history
        )
        next_bytes = self._past_key_values_nbytes(past_key_values)
        if retained_bytes + next_bytes <= self._MAX_GENERATION_CACHE_BYTES:
            history.append(past_key_values)
            return True
        else:
            history.clear()
            return False

    @staticmethod
    def _past_key_values_nbytes(past_key_values: Any) -> int:
        """Return the storage size of tensor values in a cache snapshot."""
        if isinstance(past_key_values, _CompactPastKeyValues):
            return sum(
                key_state.numel() * key_state.element_size()
                + value_state.numel() * value_state.element_size()
                for key_state, value_state in past_key_values.layers
            )
        if isinstance(past_key_values, (tuple, list)):
            return sum(
                OptimizedS3GFNModel._past_key_values_nbytes(value)
                for value in past_key_values
            )
        if isinstance(past_key_values, Tensor):
            return past_key_values.numel() * past_key_values.element_size()
        return 0

    @staticmethod
    def _initial_generation_length(past_key_values_history: list[Any]) -> int:
        """Return the token count processed by generation's first forward."""
        if not past_key_values_history:
            return 1
        if isinstance(past_key_values_history[0], _CompactPastKeyValues):
            return past_key_values_history[0].sequence_length
        first_past = past_key_values_history[0]
        if not isinstance(first_past, (tuple, list)) or not first_past:
            return 1
        first_layer = first_past[0]
        if not isinstance(first_layer, (tuple, list)) or not first_layer:
            return 1
        first_state = first_layer[0]
        if not isinstance(first_state, Tensor) or first_state.ndim < 3:
            return 1
        return int(first_state.shape[2])

    def _cached_terminal_hidden_states(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        unresolved_indices: Tensor,
        past_key_values: Any | None,
    ) -> Tensor | None:
        """Run one-token terminal forwards using generation's cached states."""
        if past_key_values is None:
            return None
        selected_past = self._select_past_batch(
            past_key_values,
            unresolved_indices,
        )
        if selected_past is None:
            return None
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=selected_past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        return self._last_hidden_state_from_outputs(outputs)

    def _store_prior_value(self, key: tuple[int, ...], value: Tensor) -> None:
        """Store one detached prior value and enforce the cache capacity."""
        self._prior_sequence_cache[key] = (
            value.detach().to(device="cpu", dtype=torch.float32).clone()
        )
        self._prior_sequence_cache.move_to_end(key)
        while len(self._prior_sequence_cache) > self.prior_cache_capacity:
            self._prior_sequence_cache.popitem(last=False)

    def _prior_cache_keys(self, input_ids: Tensor) -> list[tuple[int, ...]]:
        """Build padding-invariant cache keys with one host transfer."""
        cpu_input_ids = input_ids.detach().to(device="cpu")
        keys: list[tuple[int, ...]] = []
        for row in cpu_input_ids:
            row_values = row.tolist()
            end = len(row_values)
            while end > 0 and row_values[end - 1] == self.pad_token_id:
                end -= 1
            keys.append(tuple(int(token) for token in row_values[:end]))
        return keys

    def _supports_hidden_state_capture(self) -> bool:
        """Return whether the policy forward accepts capture-related kwargs."""
        try:
            signature = inspect.signature(self.policy.forward)
        except (TypeError, ValueError):
            return False
        parameters = signature.parameters.values()
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        required_names = {
            "output_hidden_states",
            "past_key_values",
            "return_dict",
            "use_cache",
        }
        return accepts_kwargs or required_names.issubset(signature.parameters)

    def _requires_fallback_generation(
        self,
        generation_kwargs: dict[str, Any],
    ) -> bool:
        """Return whether generation may reorder or expand cached batch rows."""
        generation_config = getattr(self.policy, "generation_config", None)
        if any(
            generation_kwargs.get(
                name,
                getattr(generation_config, name, 1),
            )
            != 1
            for name in ("num_beams", "num_beam_groups")
        ):
            return True
        return any(name in generation_kwargs for name in ("input_ids", "inputs_embeds"))

    @staticmethod
    def _last_hidden_state_from_outputs(outputs: Any) -> Tensor | None:
        """Extract the final-token hidden state from a causal-LM output."""
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            return None
        if isinstance(hidden_states, (tuple, list)):
            if not hidden_states:
                return None
            hidden_states = hidden_states[-1]
        if not isinstance(hidden_states, Tensor) or hidden_states.ndim != 3:
            return None
        return hidden_states[:, -1]

    @staticmethod
    def _generation_sequences(generated: Any) -> Tensor:
        """Normalize tensor and ``Generate*Output`` generation results."""
        sequences = getattr(generated, "sequences", generated)
        if not isinstance(sequences, Tensor) or sequences.ndim != 2:
            raise ValueError(
                "The policy generation output must contain two-dimensional sequences."
            )
        return sequences

    @staticmethod
    def _select_past_batch(
        past_key_values: Any,
        indices: Tensor,
    ) -> Any | None:
        """Select batch rows from tuple-based transformer KV caches."""
        if isinstance(past_key_values, _CompactPastKeyValues):
            selected_layers = []
            for key_state, value_state in past_key_values.layers:
                selected_key = key_state.index_select(
                    0,
                    indices.to(key_state.device),
                ).expand(
                    -1,
                    -1,
                    past_key_values.sequence_length,
                    -1,
                )
                selected_value = value_state.index_select(
                    0,
                    indices.to(value_state.device),
                )
                selected_layers.append((selected_key, selected_value))
            return tuple(selected_layers)
        if not isinstance(past_key_values, (tuple, list)):
            return None
        selected_layers: list[Any] = []
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)):
                return None
            selected_states: list[Tensor] = []
            for state in layer:
                if not isinstance(state, Tensor) or state.ndim == 0:
                    return None
                selected_states.append(state.index_select(0, indices.to(state.device)))
            selected_layers.append(tuple(selected_states))
        return tuple(selected_layers)


__all__ = ["GeneratedSequences", "OptimizedS3GFNModel"]

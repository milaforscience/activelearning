"""Experimental sampler using the optimized S3-GFN model."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import Tensor

from activelearning.sampler.optimized_s3gfn.model import OptimizedS3GFNModel
from activelearning.sampler.optimized_s3gfn.replay_buffer import (
    OptimizedReplayBatch,
    OptimizedReplayBuffer,
)
from activelearning.sampler.s3gfn.losses import relative_trajectory_balance_loss
from activelearning.sampler.s3gfn.replay_buffer import ReplayBuffer
from activelearning.sampler.s3gfn.sampler import (
    S3GFNSampler,
    _PreparedMoleculeBatch,
    _canonicalize_to_smiles,
)


@dataclass(frozen=True)
class _DeferredScalar:
    """One detached scalar materialized at an explicit synchronization boundary."""

    tensor: Tensor


@dataclass(frozen=True)
class _DeferredTrainingStep:
    """Training-step metadata whose scalar metrics remain device-resident."""

    generated_count: int
    valid_count: int
    synthesizable_count: int
    online_loss: float | _DeferredScalar | None
    replay_loss: float | _DeferredScalar | None
    auxiliary_loss: float | _DeferredScalar | None
    log_z: Tensor
    raw_reward_scores: tuple[float, ...]


class OptimizedS3GFNSampler(S3GFNSampler):
    """S3-GFN sampler with independently selectable CUDA optimizations.

    The reference sampler remains the behavioral baseline. Every optimization
    in this class is disabled by default except the deterministic prior cache
    already present in :class:`OptimizedS3GFNModel`.
    """

    def __init__(
        self,
        n_samples: int,
        fidelities: Any,
        *,
        precision: str = "fp32",
        fixed_feature_maps: bool = False,
        parallel_cuda_rollout: bool = False,
        compile_mode: str = "eager",
        deferred_sync: bool = False,
        carried_prior_scores: bool = False,
        overlap_online_prior: bool = False,
        combined_aux_policy_batch: bool = False,
        stop_check_interval: int = 1,
        prior_cache_enabled: bool = True,
        prior_cache_capacity: int = 8192,
        **kwargs: Any,
    ) -> None:
        """Initialize an optimized sampler and its ablation switches."""
        if precision not in {"fp32", "cuda_auto"}:
            raise ValueError("precision must be 'fp32' or 'cuda_auto'.")
        if compile_mode not in OptimizedS3GFNModel._COMPILE_MODES:
            raise ValueError(
                "compile_mode must be one of "
                f"{sorted(OptimizedS3GFNModel._COMPILE_MODES)}."
            )
        if stop_check_interval < 0:
            raise ValueError("stop_check_interval must be nonnegative.")
        if prior_cache_capacity <= 0:
            raise ValueError("prior_cache_capacity must be positive.")
        super().__init__(
            n_samples=n_samples,
            fidelities=fidelities,
            **kwargs,
        )
        self.precision = precision
        self.fixed_feature_maps = bool(fixed_feature_maps)
        self.parallel_cuda_rollout = bool(parallel_cuda_rollout)
        self.compile_mode = compile_mode
        self.deferred_sync = bool(deferred_sync)
        self.carried_prior_scores = bool(
            carried_prior_scores and self.deterministic_eval is True
        )
        self.overlap_online_prior = bool(overlap_online_prior)
        self.combined_aux_policy_batch = bool(combined_aux_policy_batch)
        self.stop_check_interval = int(stop_check_interval)
        self.prior_cache_enabled = bool(prior_cache_enabled)
        self.prior_cache_capacity = int(prior_cache_capacity)
        self._active_grad_scaler: Any | None = None
        self._pending_fidelity_resolution: tuple[int, tuple[int, ...]] | None = None
        self._deferred_training_steps: list[_DeferredTrainingStep] = []
        self._collect_deferred_metrics = False

    def _new_round_model(self) -> OptimizedS3GFNModel:
        """Create a fresh optimized policy sharing the frozen prior."""
        if self._pretrained_model is None:
            self._pretrained_model = OptimizedS3GFNModel.from_pretrained(
                policy_model_name_or_path=self.model_name_or_path,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                trust_remote_code=self.trust_remote_code,
                deterministic_eval=self.deterministic_eval,
                cache_dir=self.cache_dir,
                device=self.device,
                n_fidelities=len(self.fidelities),
                prior_cache_enabled=self.prior_cache_enabled,
                prior_cache_capacity=self.prior_cache_capacity,
                precision=self.precision,
                fixed_feature_maps=self.fixed_feature_maps,
                parallel_cuda_rollout=self.parallel_cuda_rollout,
                compile_mode=self.compile_mode,
                deferred_sync=self.deferred_sync,
                carried_prior_scores=self.carried_prior_scores,
                overlap_online_prior=self.overlap_online_prior,
                combined_aux_policy_batch=self.combined_aux_policy_batch,
                stop_check_interval=self.stop_check_interval,
            )
            self._keep_pretrained_template_on_cpu()
        else:
            self._keep_pretrained_template_on_cpu()

        model = OptimizedS3GFNModel(
            policy=copy.deepcopy(self._pretrained_model.policy),
            prior=self._pretrained_model.prior,
            tokenizer=self._pretrained_model.tokenizer,
            fidelity_head=copy.deepcopy(self._pretrained_model.fidelity_head),
            deterministic_eval=self.deterministic_eval,
            prior_cache_enabled=(
                self.prior_cache_enabled and self.deterministic_eval is True
            ),
            prior_cache_capacity=self.prior_cache_capacity,
            precision=self.precision,
            fixed_feature_maps=self.fixed_feature_maps,
            parallel_cuda_rollout=self.parallel_cuda_rollout,
            compile_mode=self.compile_mode,
            deferred_sync=self.deferred_sync,
            carried_prior_scores=self.carried_prior_scores,
            overlap_online_prior=self.overlap_online_prior,
            combined_aux_policy_batch=self.combined_aux_policy_batch,
            stop_check_interval=self.stop_check_interval,
        ).to(self.device)
        model.train()
        model.prior.eval()
        return model

    def _create_replay_buffers(
        self,
        *,
        pad_token_id: int,
    ) -> tuple[ReplayBuffer, ReplayBuffer | None]:
        """Create optimized positive replay storage when scores are carried."""
        if not self.carried_prior_scores:
            return super()._create_replay_buffers(pad_token_id=pad_token_id)
        seed = self.seed + self._round_index
        positive_buffer = OptimizedReplayBuffer(
            pad_token_id=pad_token_id,
            capacity=self.buffer_size,
            policy="reward",
            seed=seed,
            prior_scores_enabled=True,
        )
        negative_buffer = (
            ReplayBuffer(
                pad_token_id=pad_token_id,
                capacity=self.buffer_size,
                policy="fifo",
                seed=seed,
            )
            if self.aux_coefficient > 0.0
            else None
        )
        return positive_buffer, negative_buffer

    @contextmanager
    def _tf32_scope(self) -> Iterator[None]:
        """Enable optimized TF32 settings for one CUDA training round only."""
        if self.precision != "cuda_auto" or self.device.type != "cuda":
            yield
            return
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        old_precision = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn
            torch.set_float32_matmul_precision(old_precision)

    def _make_grad_scaler(self) -> Any | None:
        """Create an FP16 scaler while leaving BF16 and CPU in FP32 mode."""
        if self.precision != "cuda_auto" or self.device.type != "cuda":
            return None
        if torch.cuda.is_bf16_supported():
            return None
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=True)

    def _assert_finite(self, tensor: Tensor, message: str) -> None:
        """Check a tensor without synchronizing CUDA when metrics are deferred."""
        finite = torch.isfinite(tensor).all()
        if self.deferred_sync and tensor.device.type == "cuda":
            torch._assert_async(finite)
        elif not bool(finite):
            raise FloatingPointError(message)

    def _train_round(self, **kwargs: Any) -> None:
        """Preserve the reference round while scoping precision state."""
        self._active_grad_scaler = self._make_grad_scaler()
        self._collect_deferred_metrics = self.deferred_sync
        if self._collect_deferred_metrics:
            self._deferred_training_steps.clear()
        try:
            with self._tf32_scope():
                super()._train_round(**kwargs)
            if self._collect_deferred_metrics:
                self._flush_deferred_metrics()
        finally:
            self._collect_deferred_metrics = False
            self._active_grad_scaler = None

    def _optimize(
        self,
        optimizer: torch.optim.Optimizer,
        loss: Tensor | None,
        model: Any,
    ) -> float | _DeferredScalar | None:
        """Apply one ordered update with optional autocast gradient scaling."""
        if loss is None or not loss.requires_grad:
            return None
        self._assert_finite(loss.detach(), "S3-GFN loss is non-finite.")
        detached_loss = loss.detach().float()
        optimizer.zero_grad(set_to_none=True)
        scaler = self._active_grad_scaler
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

        optimized_parameters = list(model.policy.parameters())
        if model.fidelity_head is not None:
            optimized_parameters.extend(model.fidelity_head.parameters())
        optimized_parameters.append(model.log_z)
        torch.nn.utils.clip_grad_norm_(
            optimized_parameters,
            self.gradient_clip_norm,
        )
        for parameter in optimized_parameters:
            if parameter.grad is not None:
                self._assert_finite(
                    parameter.grad,
                    "S3-GFN gradients are non-finite.",
                )
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if self.deferred_sync:
            return _DeferredScalar(detached_loss)
        return float(detached_loss.cpu().item())

    def _train_step(
        self, **kwargs: Any
    ) -> tuple[int, int, int, float | None, float | None, float | None]:
        """Run one step and batch deferred metric transfers at the boundary."""
        model = kwargs["model"]
        generated = model.generate(
            count=self.batch_size,
            max_length=self.max_length,
            temperature=self.sampling_temperature,
        )
        prepared = self._prepare_batch(
            model=model,
            smiles=generated.smiles,
            fidelity_indices=generated.fidelity_indices,
            synthesizability=kwargs["synthesizability"],
            molecule_chem=kwargs["molecule_chem"],
            acquisition=kwargs["acquisition"],
            cost_fn=kwargs["cost_fn"],
        )
        online_loss = self._update_generated_batch(
            model=model,
            prepared=prepared,
            positive_buffer=kwargs["positive_buffer"],
            negative_buffer=kwargs["negative_buffer"],
            optimizer=kwargs["optimizer"],
        )
        replay_loss = None
        auxiliary_loss = None
        if prepared.smiles:
            replay_loss, auxiliary_loss = self._update_replay_batch(
                model=model,
                positive_buffer=kwargs["positive_buffer"],
                negative_buffer=kwargs["negative_buffer"],
                optimizer=kwargs["optimizer"],
            )
        if self._collect_deferred_metrics:
            self._deferred_training_steps.append(
                _DeferredTrainingStep(
                    generated_count=len(generated.smiles),
                    valid_count=len(prepared.smiles),
                    synthesizable_count=sum(prepared.synthesizable),
                    online_loss=online_loss,
                    replay_loss=replay_loss,
                    auxiliary_loss=auxiliary_loss,
                    log_z=model.log_z.detach().clone(),
                    raw_reward_scores=prepared.raw_reward_scores,
                )
            )
        else:
            online_loss, replay_loss, auxiliary_loss, log_z = (
                self._materialize_deferred_metrics(
                    online_loss,
                    replay_loss,
                    auxiliary_loss,
                    model.log_z.detach(),
                )
            )
            self.round_metrics.record_training_step(
                generated_count=len(generated.smiles),
                valid_count=len(prepared.smiles),
                synthesizable_count=sum(prepared.synthesizable),
                online_loss=online_loss,
                replay_loss=replay_loss,
                auxiliary_loss=auxiliary_loss,
                log_z=log_z,
                raw_reward_scores=prepared.raw_reward_scores,
            )
        return (
            len(generated.smiles),
            len(prepared.smiles),
            sum(prepared.synthesizable),
            online_loss,
            replay_loss,
            auxiliary_loss,
        )

    def _flush_deferred_metrics(self) -> None:
        """Materialize all round metrics in one device-to-host transfer."""
        if not self._deferred_training_steps:
            return
        tensors: list[Tensor] = []
        indices: list[dict[str, int]] = []
        for step in self._deferred_training_steps:
            step_indices: dict[str, int] = {}
            for name, value in (
                ("online", step.online_loss),
                ("replay", step.replay_loss),
                ("auxiliary", step.auxiliary_loss),
            ):
                if isinstance(value, _DeferredScalar):
                    step_indices[name] = len(tensors)
                    tensors.append(value.tensor.reshape(()))
            step_indices["log_z"] = len(tensors)
            tensors.append(step.log_z.reshape(()).float())
            indices.append(step_indices)

        host_values = torch.stack(tensors).detach().to(device="cpu").tolist()
        for step, step_indices in zip(
            self._deferred_training_steps,
            indices,
            strict=True,
        ):
            self.round_metrics.record_training_step(
                generated_count=step.generated_count,
                valid_count=step.valid_count,
                synthesizable_count=step.synthesizable_count,
                online_loss=(
                    None
                    if "online" not in step_indices
                    else float(host_values[step_indices["online"]])
                ),
                replay_loss=(
                    None
                    if "replay" not in step_indices
                    else float(host_values[step_indices["replay"]])
                ),
                auxiliary_loss=(
                    None
                    if "auxiliary" not in step_indices
                    else float(host_values[step_indices["auxiliary"]])
                ),
                log_z=float(host_values[step_indices["log_z"]]),
                raw_reward_scores=step.raw_reward_scores,
            )
        self._deferred_training_steps.clear()

    def _materialize_deferred_metrics(
        self,
        online_loss: float | _DeferredScalar | None,
        replay_loss: float | _DeferredScalar | None,
        auxiliary_loss: float | _DeferredScalar | None,
        log_z: Tensor,
    ) -> tuple[float | None, float | None, float | None, float]:
        """Transfer all deferred round scalars with one host copy."""
        values: list[Tensor] = []
        locations: list[tuple[str, int]] = []
        for name, value in (
            ("online", online_loss),
            ("replay", replay_loss),
            ("auxiliary", auxiliary_loss),
        ):
            if isinstance(value, _DeferredScalar):
                locations.append((name, len(values)))
                values.append(value.tensor.reshape(()))
        locations.append(("log_z", len(values)))
        values.append(log_z.reshape(()).float())
        host_values = torch.stack(values).detach().to(device="cpu").tolist()
        resolved: dict[str, float] = {
            name: float(host_values[index]) for name, index in locations
        }
        return (
            resolved.get("online", online_loss),
            resolved.get("replay", replay_loss),
            resolved.get("auxiliary", auxiliary_loss),
            resolved["log_z"],
        )

    def _canonicalize_batch(
        self,
        smiles: Any,
        *,
        fidelity_indices: Tensor | None,
        molecule_chem: Any,
    ) -> tuple[tuple[str, ...], Tensor | None]:
        """Canonicalize a batch while converting fidelity actions once."""
        self._pending_fidelity_resolution = None
        if fidelity_indices is None:
            if len(self.fidelities) > 1 and smiles:
                raise ValueError(
                    "Multi-fidelity generation must return fidelity action indices."
                )
            cpu_indices = None
        else:
            if fidelity_indices.ndim != 1 or fidelity_indices.shape[0] != len(smiles):
                raise ValueError(
                    "Fidelity indices must align with generated molecules."
                )
            if fidelity_indices.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("Fidelity indices must contain integer action indices.")
            cpu_indices = (
                fidelity_indices.detach()
                .to(
                    device="cpu",
                    dtype=torch.long,
                )
                .tolist()
            )
            if any(index < 0 or index >= len(self.fidelities) for index in cpu_indices):
                raise ValueError(
                    "Generated fidelity indices contain an unknown action."
                )
        canonical_smiles: list[str] = []
        retained_indices: list[int] = []
        for index, generated_smiles in enumerate(smiles):
            canonical = _canonicalize_to_smiles(
                generated_smiles,
                molecule_chem=molecule_chem,
            )
            if canonical is not None:
                canonical_smiles.append(canonical)
                if cpu_indices is not None:
                    retained_indices.append(int(cpu_indices[index]))
        return (
            tuple(canonical_smiles),
            self._build_canonical_fidelity_tensor(
                fidelity_indices,
                retained_indices,
            ),
        )

    def _build_canonical_fidelity_tensor(
        self,
        fidelity_indices: Tensor | None,
        retained_indices: list[int],
    ) -> Tensor | None:
        """Build aligned action tensors and retain their host resolution."""
        if fidelity_indices is None:
            return None
        canonical_indices = torch.tensor(
            retained_indices,
            dtype=torch.long,
            device=fidelity_indices.device,
        )
        self._pending_fidelity_resolution = (
            id(canonical_indices),
            tuple(self.fidelities[index] for index in retained_indices),
        )
        return canonical_indices

    def _resolve_fidelity_values(
        self,
        fidelity_indices: Tensor | None,
        *,
        count: int,
    ) -> tuple[int, ...]:
        """Reuse one host fidelity conversion for canonicalized batches."""
        pending = self._pending_fidelity_resolution
        if pending is not None and pending[0] == id(fidelity_indices):
            self._pending_fidelity_resolution = None
            if len(pending[1]) != count:
                raise ValueError(
                    "Fidelity indices must align with generated molecules."
                )
            return pending[1]
        return super()._resolve_fidelity_values(
            fidelity_indices,
            count=count,
        )

    def _update_generated_batch(
        self,
        *,
        model: OptimizedS3GFNModel,
        prepared: _PreparedMoleculeBatch,
        positive_buffer: ReplayBuffer,
        negative_buffer: ReplayBuffer | None,
        optimizer: torch.optim.Optimizer,
    ) -> float | _DeferredScalar | None:
        """Store trajectories and apply one online update."""
        positive_mask = torch.tensor(
            prepared.synthesizable,
            dtype=torch.bool,
            device=prepared.input_ids.device,
        )
        negative_mask = ~positive_mask
        positive_smiles = tuple(
            smile
            for smile, is_positive in zip(
                prepared.smiles,
                prepared.synthesizable,
            )
            if is_positive
        )
        online_loss: Tensor | None = None
        prior_scores: Tensor | None = None
        if positive_smiles:
            positive_input_ids = prepared.input_ids[positive_mask]
            positive_reward_scores = prepared.reward_scores[positive_mask]
            positive_fidelity_indices = (
                None
                if prepared.fidelity_indices is None
                else prepared.fidelity_indices[positive_mask]
            )
            if self.carried_prior_scores:
                prior_scores = model.prior_sequence_log_probabilities_uncached(
                    positive_input_ids
                )
                positive_buffer.add_batch(
                    positive_input_ids,
                    positive_smiles,
                    positive_reward_scores,
                    fidelity_indices=positive_fidelity_indices,
                    prior_log_probabilities=prior_scores,
                )
                online_loss = model.on_policy_loss(
                    positive_input_ids,
                    positive_reward_scores,
                    self.beta,
                    fidelity_indices=positive_fidelity_indices,
                    precomputed_sequence_log_probabilities=prior_scores,
                )
            elif self.overlap_online_prior:
                positive_buffer.add_batch(
                    positive_input_ids,
                    positive_smiles,
                    positive_reward_scores,
                    fidelity_indices=positive_fidelity_indices,
                )
                online_loss = self._overlapped_online_loss(
                    model=model,
                    input_ids=positive_input_ids,
                    reward_scores=positive_reward_scores,
                    fidelity_indices=positive_fidelity_indices,
                )
            else:
                positive_buffer.add_batch(
                    positive_input_ids,
                    positive_smiles,
                    positive_reward_scores,
                    fidelity_indices=positive_fidelity_indices,
                )
                online_loss = model.on_policy_loss(
                    positive_input_ids,
                    positive_reward_scores,
                    self.beta,
                    fidelity_indices=positive_fidelity_indices,
                )
            optimized_loss = self._optimize(optimizer, online_loss, model)
        else:
            optimized_loss = None

        negative_smiles = tuple(
            smile
            for smile, is_positive in zip(
                prepared.smiles,
                prepared.synthesizable,
            )
            if not is_positive
        )
        if negative_buffer is not None and negative_smiles:
            negative_buffer.add_batch(
                prepared.input_ids[negative_mask],
                negative_smiles,
                prepared.reward_scores[negative_mask],
                fidelity_indices=(
                    None
                    if prepared.fidelity_indices is None
                    else prepared.fidelity_indices[negative_mask]
                ),
            )
        return optimized_loss

    def _overlapped_online_loss(
        self,
        *,
        model: OptimizedS3GFNModel,
        input_ids: Tensor,
        reward_scores: Tensor,
        fidelity_indices: Tensor | None,
    ) -> Tensor:
        """Overlap independent frozen-prior and policy forwards on CUDA."""
        if self.device.type != "cuda":
            return model.on_policy_loss(
                input_ids,
                reward_scores,
                self.beta,
                fidelity_indices=fidelity_indices,
            )
        prior_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(prior_stream):
            prior_sequence_scores = model.prior_sequence_log_probabilities_uncached(
                input_ids
            )
        policy_scores = model.policy_trajectory_log_probabilities(
            input_ids,
            fidelity_indices=fidelity_indices,
        )
        event = torch.cuda.Event()
        event.record(prior_stream)
        torch.cuda.current_stream(device=self.device).wait_event(event)
        prior_scores = model.prior_trajectory_log_probabilities(
            input_ids,
            fidelity_indices=fidelity_indices,
            precomputed_sequence_log_probabilities=prior_sequence_scores,
        )
        return relative_trajectory_balance_loss(
            policy_log_probabilities=policy_scores,
            prior_log_probabilities=prior_scores,
            reward_scores=reward_scores.to(
                device=self.device,
                dtype=model.log_z.dtype,
            ),
            log_z=model.log_z.float(),
            beta=self.beta,
        )

    def _update_replay_batch(
        self,
        *,
        model: OptimizedS3GFNModel,
        positive_buffer: ReplayBuffer,
        negative_buffer: ReplayBuffer | None,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float | _DeferredScalar | None, float | _DeferredScalar | None]:
        """Apply the replay update while reusing carried prior scores."""
        if len(positive_buffer) < self.replay_batch_size:
            return None, None
        positive_replay = positive_buffer.sample(
            count=min(self.replay_batch_size, len(positive_buffer)),
            device=self.device,
            dtype=self.dtype,
            reward_prioritized=True,
            replace=True,
        )
        negative_replay = None
        if (
            negative_buffer is not None
            and len(negative_buffer) >= self.replay_batch_size
        ):
            negative_replay = negative_buffer.sample(
                count=self.replay_batch_size,
                device=self.device,
                dtype=self.dtype,
            )
        prior_scores = None
        if isinstance(positive_replay, OptimizedReplayBatch):
            if (
                positive_replay.prior_log_probabilities is not None
                and positive_replay.prior_score_mask is not None
            ):
                prior_scores = positive_replay.prior_log_probabilities.clone()
                prior_scores.masked_fill_(
                    ~positive_replay.prior_score_mask,
                    float("nan"),
                )
        loss = model.replay_loss(
            positive_input_ids=positive_replay.input_ids,
            reward_scores=positive_replay.reward_scores,
            beta=self.beta,
            negative_input_ids=(
                None if negative_replay is None else negative_replay.input_ids
            ),
            aux_coefficient=self.aux_coefficient,
            positive_fidelity_indices=positive_replay.fidelity_indices,
            negative_fidelity_indices=(
                None if negative_replay is None else negative_replay.fidelity_indices
            ),
            precomputed_positive_prior_log_probabilities=prior_scores,
        )
        optimized_loss = self._optimize(optimizer, loss, model)
        auxiliary_tensor = getattr(model, "_last_auxiliary_loss", None)
        if isinstance(auxiliary_tensor, Tensor):
            auxiliary_loss: float | _DeferredScalar | None
            if self.deferred_sync:
                auxiliary_loss = _DeferredScalar(auxiliary_tensor.detach())
            else:
                auxiliary_loss = float(auxiliary_tensor.detach().cpu().item())
        else:
            auxiliary_loss = None
        return optimized_loss, auxiliary_loss


__all__ = ["OptimizedS3GFNSampler"]

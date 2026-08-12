"""Acquisition-guided S3-GFN sampler for molecular active learning.

The sampler trains a fresh GP-MoLFormer policy from a pretrained checkpoint
for each active-learning round. It canonicalizes generated SMILES, scores them
with the active-learning acquisition function, trains with Relative Trajectory
Balance (RTB), and returns unique connected molecules with an explicitly
sampled fidelity.

Canonicalization parses each generated string with RDKit, rejects invalid or
disconnected molecules, and rewrites each valid molecule as RDKit's
deterministic canonical SMILES representation. This gives equivalent SMILES a
single representation for deduplication and downstream scoring.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import Tensor

from activelearning.acquisition.cost_utility import cost_weighting_from_cost_fn
from activelearning.sampler.sampler import Sampler
from activelearning.sampler.s3gfn._optional import require_rdkit
from activelearning.sampler.s3gfn.model import S3GFNModel
from activelearning.sampler.s3gfn.replay_buffer import ReplayBuffer
from activelearning.sampler.s3gfn.synthesizability import SAScoreSynthesizability
from activelearning.utils.types import Candidate, Observation

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedMoleculeBatch:
    """RDKit-canonicalized molecules and their acquisition-derived rewards.

    Canonicalization parses each string, rejects invalid or disconnected
    molecules, and rewrites valid molecules into one deterministic SMILES
    representation.

    Attributes
    ----------
    smiles : tuple[str, ...]
        Canonical connected SMILES strings retained from one generated batch.
    input_ids : Tensor
        Padded token ids for ``smiles`` on the model device.
    reward_scores : Tensor
        Scaled reward scores associated with ``smiles``. The tensor uses the
        sampler's bound runtime floating-point dtype.
    fidelity_indices : Tensor or None
        Optional terminal fidelity-action indices aligned with ``smiles``.
    synthesizable : tuple[bool, ...]
        Synthetic accessibility score (SA-score) threshold results aligned with ``smiles``.
    """

    smiles: tuple[str, ...]
    input_ids: Tensor
    reward_scores: Tensor
    synthesizable: tuple[bool, ...]
    fidelity_indices: Tensor | None = None


class S3GFNSampler(Sampler):
    """Generate canonical SMILES with an acquisition-guided S3-GFN policy.

    Generated strings are canonicalized by RDKit before scoring: invalid or
    disconnected molecules are discarded, and valid molecules are rewritten
    into one deterministic canonical SMILES representation.

    Each :meth:`sample` call creates a fresh trainable policy from the
    pretrained GP-MoLFormer checkpoint. The policy is optimized with RTB using
    acquisition-derived reward scores. A reward-prioritized replay buffer keeps
    synthesizable, chemically diverse trajectories, while an optional FIFO
    buffer supplies negative trajectories to the auxiliary contrastive loss.

    The sampler models molecules and, when multiple fidelities are configured,
    adds one categorical fidelity action after each terminal molecule.
    The frozen prior, tokenizer, and pretrained policy weights are cached
    across rounds; learned policy weights are not carried between rounds.
    """

    def __init__(
        self,
        n_samples: int,
        fidelities: Sequence[int],
        model_name_or_path: str = "ibm-research/GP-MoLFormer-Uniq",
        tokenizer_name_or_path: str = "ibm-research/MoLFormer-XL-both-10pct",
        *,
        trust_remote_code: bool = True,
        cache_dir: str | None = None,
        max_length: int = 140,
        batch_size: int = 64,
        replay_batch_size: int = 64,
        n_train_steps: int = 5000,
        num_warmup_steps: int = 100,
        learning_rate: float = 1.0e-4,
        log_z_learning_rate: float = 1.0e-3,
        beta: float = 25.0,
        aux_coefficient: float = 1.0e-3,
        buffer_size: int = 6400,
        sa_threshold: float = 4.0,
        sampling_temperature: float = 1.0,
        gradient_clip_norm: float = 10.0,
        seed: int = 42,
    ) -> None:
        """Initialize an acquisition-guided S3-GFN sampler.

        Parameters
        ----------
        n_samples : int
            Number of unique candidates returned by each :meth:`sample` call.
        fidelities : Sequence[int]
            Fidelity levels available to the terminal action. One level keeps
            the molecule-only trajectory; multiple levels add one final action.
        model_name_or_path : str, optional
            Hugging Face identifier or local path for the pretrained
            GP-MoLFormer causal language model.
        tokenizer_name_or_path : str, optional
            Hugging Face identifier or local path for its tokenizer.
        trust_remote_code : bool, optional
            Whether Transformers may load custom model and tokenizer code.
        cache_dir : str or None, optional
            Directory used for Hugging Face downloads and cache files.
        max_length : int, optional
            Maximum tokenized or generated sequence length, including special
            tokens.
        batch_size : int, optional
            Number of molecules generated for each training step and final
            generation attempt.
        replay_batch_size : int, optional
            Maximum number of positive and negative trajectories sampled for a
            replay update.
        n_train_steps : int, optional
            Number of policy-training iterations per active-learning round.
        num_warmup_steps : int, optional
            Number of linear learning-rate warmup steps. Zero disables the
            Transformers scheduler.
        learning_rate : float, optional
            AdamW learning rate for policy parameters.
        log_z_learning_rate : float, optional
            AdamW learning rate for the trainable RTB normalizer ``log Z``.
        beta : float, optional
            Reward inverse temperature used to convert acquisition scores into
            RTB log rewards.
        aux_coefficient : float, optional
            Weight of the negative replay contrastive loss. Zero disables the
            negative replay buffer and auxiliary loss.
        buffer_size : int, optional
            Capacity of each replay buffer.
        sa_threshold : float, optional
            Maximum synthetic accessibility score accepted as synthesizable.
        sampling_temperature : float, optional
            Temperature used when sampling policy tokens.
        gradient_clip_norm : float, optional
            Maximum global norm for policy and ``log Z`` gradients.
        seed : int, optional
            Base seed. The active-learning round index is added to it before
            seeding Python, PyTorch, and CUDA generators.

        Raises
        ------
        ValueError
            If a count, hyperparameter, fidelity list, or seed is invalid.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if not fidelities:
            raise ValueError("fidelities must contain at least one level.")
        if len(set(fidelities)) != len(fidelities):
            raise ValueError("fidelities must not contain duplicates.")
        if max_length < 2:
            raise ValueError("max_length must be at least two.")
        if batch_size <= 0 or replay_batch_size <= 0:
            raise ValueError("batch_size and replay_batch_size must be positive.")
        if n_train_steps <= 0:
            raise ValueError("n_train_steps must be positive.")
        if num_warmup_steps < 0:
            raise ValueError("num_warmup_steps must be nonnegative.")
        if learning_rate <= 0.0 or log_z_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive.")
        if beta <= 0.0 or not math.isfinite(beta):
            raise ValueError("beta must be a finite positive value.")
        if aux_coefficient < 0.0 or not math.isfinite(aux_coefficient):
            raise ValueError("aux_coefficient must be finite and nonnegative.")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        if sa_threshold < 0.0 or not math.isfinite(sa_threshold):
            raise ValueError("sa_threshold must be finite and nonnegative.")
        if sampling_temperature <= 0.0 or not math.isfinite(sampling_temperature):
            raise ValueError("sampling_temperature must be finite and positive.")
        if gradient_clip_norm <= 0.0 or not math.isfinite(gradient_clip_norm):
            raise ValueError("gradient_clip_norm must be finite and positive.")
        if seed < 0:
            raise ValueError("seed must be nonnegative.")

        self.n_samples = n_samples
        self.fidelities = tuple(int(fidelity) for fidelity in fidelities)
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.replay_batch_size = replay_batch_size
        self.n_train_steps = n_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.learning_rate = learning_rate
        self.log_z_learning_rate = log_z_learning_rate
        self.beta = beta
        self.aux_coefficient = aux_coefficient
        self.buffer_size = buffer_size
        self.sa_threshold = sa_threshold
        self.sampling_temperature = sampling_temperature
        self.gradient_clip_norm = gradient_clip_norm
        self.seed = seed
        self._pretrained_model: S3GFNModel | None = None
        self._round_index = 0

    def sample(
        self,
        acquisition: Any | None = None,
        observations: Iterable[Observation] | None = None,
        cost_fn: Callable[[Sequence[Candidate]], list[float]] | None = None,
    ) -> Sequence[Candidate]:
        """Train a fresh policy and return unique acquisition-weighted candidates.

        Parameters
        ----------
        acquisition : Any, optional
            Singleton-scoring acquisition function used to turn generated
            molecule-fidelity pairs into RTB reward scores. S3-GFN requires
            ``supports_singleton_scoring`` to be true.
        observations : Iterable[Observation], optional
            Current active-learning observations. Accepted for the common
            sampler interface but intentionally unused: each round starts from
            the pretrained policy and does not warm-start from observations.
        cost_fn : callable, optional
            Function returning one positive cost per candidate. When provided,
            acquisition scores are inverse-cost weighted before becoming
            reward scores.

        Returns
        -------
        Sequence[Candidate]
            Exactly ``n_samples`` unique candidates with canonical connected
            SMILES and sampled fidelity levels.

        Raises
        ------
        ValueError
            If no acquisition is supplied or it does not support singleton
            scoring.
        RuntimeError
            If bounded final-generation retries cannot produce enough unique
            valid molecules.
        """
        _ = observations
        self._validate_acquisition(acquisition)

        self._set_round_seed()
        round_number = self._round_index + 1
        _logger.info(
            "S3-GFN round %d started: target=%d candidate(s), train_steps=%d, "
            "batch_size=%d, device=%s.",
            round_number,
            self.n_samples,
            self.n_train_steps,
            self.batch_size,
            self.device,
        )
        _logger.info("S3-GFN round %d: loading molecule dependencies.", round_number)
        molecule_chem, _, _ = require_rdkit()
        synthesizability = SAScoreSynthesizability(threshold=self.sa_threshold)
        model = self._new_round_model()
        positive_buffer, negative_buffer = self._create_replay_buffers(
            pad_token_id=model.pad_token_id
        )

        self._train_round(
            model=model,
            synthesizability=synthesizability,
            positive_buffer=positive_buffer,
            negative_buffer=negative_buffer,
            molecule_chem=molecule_chem,
            acquisition=acquisition,
            cost_fn=cost_fn,
        )
        _logger.info("S3-GFN round %d: policy training complete.", round_number)
        candidates = self._generate_final_candidates(
            model=model,
            molecule_chem=molecule_chem,
        )
        _logger.info(
            "S3-GFN round %d complete: generated %d candidate(s).",
            round_number,
            len(candidates),
        )
        self._round_index += 1
        return candidates

    def _new_round_model(self) -> S3GFNModel:
        """Create a fresh policy from cached pretrained model components."""
        if self._pretrained_model is None:
            _logger.info(
                "S3-GFN pretrained model cache is empty; loading policy and prior."
            )
            self._pretrained_model = S3GFNModel.from_pretrained(
                policy_model_name_or_path=self.model_name_or_path,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                device=self.device,
                n_fidelities=len(self.fidelities),
            )
        else:
            _logger.info("Reusing cached S3-GFN pretrained model components.")
            self._pretrained_model.to(self.device)

        policy = copy.deepcopy(self._pretrained_model.policy)
        model = S3GFNModel(
            policy=policy,
            prior=self._pretrained_model.prior,
            tokenizer=self._pretrained_model.tokenizer,
            fidelity_head=copy.deepcopy(self._pretrained_model.fidelity_head),
        ).to(self.device)
        model.policy.train()
        model.prior.eval()
        _logger.info("Fresh trainable S3-GFN policy initialized.")
        return model

    def _validate_acquisition(self, acquisition: Any | None) -> None:
        """Validate the singleton-scoring acquisition contract."""
        if acquisition is None:
            raise ValueError("S3GFNSampler requires an acquisition function.")
        if not getattr(acquisition, "supports_singleton_scoring", True):
            raise ValueError(
                f"{type(acquisition).__name__} does not support singleton scoring. "
                "S3GFNSampler requires an acquisition with score()."
            )

    def _create_replay_buffers(
        self,
        *,
        pad_token_id: int,
    ) -> tuple[ReplayBuffer, ReplayBuffer | None]:
        """Create the positive and optional negative replay buffers."""
        seed = self.seed + self._round_index
        positive_buffer = ReplayBuffer(
            pad_token_id=pad_token_id,
            capacity=self.buffer_size,
            policy="reward",
            seed=seed,
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

    def _train_round(
        self,
        *,
        model: S3GFNModel,
        synthesizability: SAScoreSynthesizability,
        positive_buffer: ReplayBuffer,
        negative_buffer: ReplayBuffer | None,
        molecule_chem: Any,
        acquisition: Any,
        cost_fn: Callable[[Sequence[Candidate]], list[float]] | None,
    ) -> None:
        """Run the configured training steps and advance the scheduler."""
        policy_parameters = list(model.policy.parameters())
        if model.fidelity_head is not None:
            policy_parameters.extend(model.fidelity_head.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": policy_parameters, "lr": self.learning_rate},
                {"params": [model.log_z], "lr": self.log_z_learning_rate},
            ]
        )
        scheduler = self._build_scheduler(optimizer)
        progress_interval = max(1, self.n_train_steps // 10)
        _logger.info(
            "S3-GFN policy training started: %d step(s), batch_size=%d.",
            self.n_train_steps,
            self.batch_size,
        )

        for step_index in range(self.n_train_steps):
            generated_count, valid_count, synthesizable_count = self._train_step(
                model=model,
                synthesizability=synthesizability,
                positive_buffer=positive_buffer,
                negative_buffer=negative_buffer,
                molecule_chem=molecule_chem,
                acquisition=acquisition,
                cost_fn=cost_fn,
                optimizer=optimizer,
            )
            if scheduler is not None:
                scheduler.step()
            self._log_training_progress(
                step_number=step_index + 1,
                progress_interval=progress_interval,
                generated_count=generated_count,
                valid_count=valid_count,
                synthesizable_count=synthesizable_count,
                positive_buffer=positive_buffer,
                negative_buffer=negative_buffer,
            )

    def _train_step(
        self,
        *,
        model: S3GFNModel,
        synthesizability: SAScoreSynthesizability,
        positive_buffer: ReplayBuffer,
        negative_buffer: ReplayBuffer | None,
        molecule_chem: Any,
        acquisition: Any,
        cost_fn: Callable[[Sequence[Candidate]], list[float]] | None,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[int, int, int]:
        """Generate one batch and apply on-policy and replay updates."""
        generated = model.generate(
            count=self.batch_size,
            max_length=self.max_length,
            temperature=self.sampling_temperature,
        )
        prepared = self._prepare_batch(
            model=model,
            smiles=generated.smiles,
            fidelity_indices=generated.fidelity_indices,
            synthesizability=synthesizability,
            molecule_chem=molecule_chem,
            acquisition=acquisition,
            cost_fn=cost_fn,
        )
        self._update_generated_batch(
            model=model,
            prepared=prepared,
            positive_buffer=positive_buffer,
            negative_buffer=negative_buffer,
            optimizer=optimizer,
        )
        self._update_replay_batch(
            model=model,
            positive_buffer=positive_buffer,
            negative_buffer=negative_buffer,
            optimizer=optimizer,
        )
        return (
            len(generated.smiles),
            len(prepared.smiles),
            sum(prepared.synthesizable),
        )

    def _update_generated_batch(
        self,
        *,
        model: S3GFNModel,
        prepared: _PreparedMoleculeBatch,
        positive_buffer: ReplayBuffer,
        negative_buffer: ReplayBuffer | None,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Store generated trajectories and apply the on-policy update."""
        positive_mask = torch.tensor(
            prepared.synthesizable,
            dtype=torch.bool,
            device=prepared.input_ids.device,
        )
        negative_mask = ~positive_mask
        positive_smiles = tuple(
            smiles
            for smiles, is_positive in zip(
                prepared.smiles,
                prepared.synthesizable,
            )
            if is_positive
        )
        if positive_smiles:
            positive_input_ids = prepared.input_ids[positive_mask]
            positive_reward_scores = prepared.reward_scores[positive_mask]
            positive_fidelity_indices = (
                None
                if prepared.fidelity_indices is None
                else prepared.fidelity_indices[positive_mask]
            )
            positive_buffer.add_batch(
                positive_input_ids,
                positive_smiles,
                positive_reward_scores,
                fidelity_indices=positive_fidelity_indices,
            )
            self._optimize(
                optimizer,
                model.on_policy_loss(
                    positive_input_ids,
                    positive_reward_scores,
                    self.beta,
                    fidelity_indices=positive_fidelity_indices,
                ),
                model,
            )

        negative_smiles = tuple(
            smiles
            for smiles, is_positive in zip(
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

    def _update_replay_batch(
        self,
        *,
        model: S3GFNModel,
        positive_buffer: ReplayBuffer,
        negative_buffer: ReplayBuffer | None,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Sample replay data and apply the replay loss when available."""
        if not positive_buffer:
            return

        positive_replay = positive_buffer.sample(
            count=min(self.replay_batch_size, len(positive_buffer)),
            device=self.device,
            dtype=self.dtype,
            reward_prioritized=True,
            replace=True,
        )
        negative_replay = None
        if negative_buffer is not None and len(negative_buffer) > 0:
            negative_replay = negative_buffer.sample(
                count=min(self.replay_batch_size, len(negative_buffer)),
                device=self.device,
                dtype=self.dtype,
            )
        self._optimize(
            optimizer,
            model.replay_loss(
                positive_input_ids=positive_replay.input_ids,
                reward_scores=positive_replay.reward_scores,
                beta=self.beta,
                negative_input_ids=(
                    None if negative_replay is None else negative_replay.input_ids
                ),
                aux_coefficient=self.aux_coefficient,
                positive_fidelity_indices=positive_replay.fidelity_indices,
                negative_fidelity_indices=(
                    None
                    if negative_replay is None
                    else negative_replay.fidelity_indices
                ),
            ),
            model,
        )

    def _log_training_progress(
        self,
        *,
        step_number: int,
        progress_interval: int,
        generated_count: int,
        valid_count: int,
        synthesizable_count: int,
        positive_buffer: ReplayBuffer,
        negative_buffer: ReplayBuffer | None,
    ) -> None:
        """Log periodic training counts."""
        if not (
            step_number == 1
            or step_number % progress_interval == 0
            or step_number == self.n_train_steps
        ):
            return
        _logger.info(
            "S3-GFN training step %d/%d: generated=%d, valid=%d, "
            "invalid=%d, synthesizable=%d, positive_buffer=%d, "
            "negative_buffer=%d.",
            step_number,
            self.n_train_steps,
            generated_count,
            valid_count,
            generated_count - valid_count,
            synthesizable_count,
            len(positive_buffer),
            len(negative_buffer) if negative_buffer is not None else 0,
        )

    def _prepare_batch(
        self,
        *,
        model: S3GFNModel,
        smiles: Sequence[str],
        fidelity_indices: Tensor | None,
        synthesizability: SAScoreSynthesizability,
        molecule_chem: Any,
        acquisition: Any,
        cost_fn: Callable[[Sequence[Candidate]], list[float]] | None,
    ) -> _PreparedMoleculeBatch:
        """Canonicalize, score, encode, and classify one generated batch."""
        canonical_smiles, canonical_fidelity_indices = self._canonicalize_batch(
            smiles,
            fidelity_indices=fidelity_indices,
            molecule_chem=molecule_chem,
        )

        if not canonical_smiles:
            empty_ids = model.encode_smiles([])
            return _PreparedMoleculeBatch(
                smiles=(),
                input_ids=empty_ids,
                reward_scores=torch.empty(
                    0,
                    dtype=self.dtype,
                    device=empty_ids.device,
                ),
                synthesizable=(),
                fidelity_indices=canonical_fidelity_indices,
            )

        candidates = [
            Candidate(x=canonical, fidelity=fidelity)
            for canonical, fidelity in zip(
                canonical_smiles,
                self._resolve_fidelity_values(
                    canonical_fidelity_indices,
                    count=len(canonical_smiles),
                ),
            )
        ]
        scores = _score_candidates(acquisition, candidates, cost_fn=cost_fn)

        input_ids = model.encode_smiles(canonical_smiles, max_length=self.max_length)
        labels = synthesizability.classify_batch(canonical_smiles)
        return _PreparedMoleculeBatch(
            smiles=tuple(canonical_smiles),
            input_ids=input_ids,
            reward_scores=torch.tensor(
                scores,
                # Reward scores are derived floating-point data, so use the
                # bound runtime dtype while tokenized inputs remain integer.
                dtype=self.dtype,
                device=input_ids.device,
            ),
            synthesizable=tuple(labels),
            fidelity_indices=canonical_fidelity_indices,
        )

    def _canonicalize_batch(
        self,
        smiles: Sequence[str],
        *,
        fidelity_indices: Tensor | None,
        molecule_chem: Any,
    ) -> tuple[tuple[str, ...], Tensor | None]:
        """Return valid canonical SMILES and aligned action indices."""
        self._resolve_fidelity_values(fidelity_indices, count=len(smiles))
        canonical_smiles: list[str] = []
        retained_fidelity_indices: list[int] = []
        for index, generated_smiles in enumerate(smiles):
            canonical = _canonicalize_to_smiles(
                generated_smiles,
                molecule_chem=molecule_chem,
            )
            if canonical is not None:
                canonical_smiles.append(canonical)
                if fidelity_indices is not None:
                    retained_fidelity_indices.append(
                        int(fidelity_indices[index].item())
                    )
        return (
            tuple(canonical_smiles),
            (
                None
                if fidelity_indices is None
                else torch.tensor(
                    retained_fidelity_indices,
                    dtype=torch.long,
                    device=fidelity_indices.device,
                )
            ),
        )

    def _resolve_fidelity_values(
        self,
        fidelity_indices: Tensor | None,
        *,
        count: int,
    ) -> tuple[int, ...]:
        """Map optional action indices to configured fidelity values."""
        if count == 0 and fidelity_indices is None:
            return ()
        if fidelity_indices is None:
            if len(self.fidelities) > 1:
                raise ValueError(
                    "Multi-fidelity generation must return fidelity action indices."
                )
            return (self.fidelities[0],) * count
        if fidelity_indices.ndim != 1 or fidelity_indices.shape[0] != count:
            raise ValueError("Fidelity indices must align with generated molecules.")
        if fidelity_indices.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("Fidelity indices must contain integer action indices.")
        indices = fidelity_indices.detach().to(device="cpu", dtype=torch.long)
        if indices.numel() and (
            bool(torch.any(indices < 0))
            or bool(torch.any(indices >= len(self.fidelities)))
        ):
            raise ValueError("Generated fidelity indices contain an unknown action.")
        return tuple(self.fidelities[int(index)] for index in indices)

    def _generate_final_candidates(
        self,
        *,
        model: S3GFNModel,
        molecule_chem: Any,
    ) -> list[Candidate]:
        """Generate exactly ``n_samples`` unique canonical candidates."""
        model.policy.eval()
        model.prior.eval()
        candidates: list[Candidate] = []
        seen_smiles: set[str] = set()
        generated_attempts = 0
        max_attempts = max(self.n_samples * 20, self.batch_size * 2)
        batch_index = 0
        _logger.info(
            "Generating %d final candidate(s), with at most %d model attempts.",
            self.n_samples,
            max_attempts,
        )

        while len(candidates) < self.n_samples and generated_attempts < max_attempts:
            batch_index += 1
            requested = min(self.batch_size, self.n_samples - len(candidates))
            generated = model.generate(
                count=requested,
                max_length=self.max_length,
                temperature=self.sampling_temperature,
            )
            generated_attempts += max(requested, len(generated.smiles))
            valid_count, invalid_count, duplicate_count = self._process_candidate_batch(
                smiles=generated.smiles,
                fidelity_indices=generated.fidelity_indices,
                candidates=candidates,
                seen_smiles=seen_smiles,
                molecule_chem=molecule_chem,
            )
            self._log_final_batch_progress(
                batch_index=batch_index,
                generated_attempts=generated_attempts,
                max_attempts=max_attempts,
                valid_count=valid_count,
                invalid_count=invalid_count,
                duplicate_count=duplicate_count,
                candidate_count=len(candidates),
            )

        if len(candidates) != self.n_samples:
            _logger.error(
                "Final candidate generation stopped at %d/%d unique candidate(s) "
                "after %d attempts.",
                len(candidates),
                self.n_samples,
                generated_attempts,
            )
            raise RuntimeError(
                f"S3GFNSampler generated {len(candidates)} valid unique molecules "
                f"after {generated_attempts} attempts; requested {self.n_samples}."
            )
        _logger.info(
            "Final candidate generation complete: %d unique candidate(s).",
            len(candidates),
        )
        return candidates

    def _process_candidate_batch(
        self,
        *,
        smiles: Sequence[str],
        fidelity_indices: Tensor | None,
        candidates: list[Candidate],
        seen_smiles: set[str],
        molecule_chem: Any,
    ) -> tuple[int, int, int]:
        """Validate a generation batch and append new candidates.

        Returns ``(valid, invalid, duplicate)`` counts for progress logging.
        """
        invalid_count = 0
        duplicate_count = 0
        valid_count = 0
        fidelity_values = self._resolve_fidelity_values(
            fidelity_indices,
            count=len(smiles),
        )
        for index, generated_smiles in enumerate(smiles):
            canonical = _canonicalize_to_smiles(
                generated_smiles,
                molecule_chem=molecule_chem,
            )
            if canonical is None:
                invalid_count += 1
                continue
            if canonical in seen_smiles:
                duplicate_count += 1
                continue

            valid_count += 1
            candidates.append(Candidate(x=canonical, fidelity=fidelity_values[index]))
            seen_smiles.add(canonical)
            if len(candidates) == self.n_samples:
                break

        return valid_count, invalid_count, duplicate_count

    def _log_final_batch_progress(
        self,
        *,
        batch_index: int,
        generated_attempts: int,
        max_attempts: int,
        valid_count: int,
        invalid_count: int,
        duplicate_count: int,
        candidate_count: int,
    ) -> None:
        """Log final-generation progress at bounded intervals."""
        if not (
            batch_index <= 3
            or batch_index % 5 == 0
            or candidate_count == self.n_samples
        ):
            return
        _logger.info(
            "Final candidate batch %d: %d/%d unique candidate(s), "
            "attempts=%d/%d, valid=%d, invalid=%d, duplicate=%d.",
            batch_index,
            candidate_count,
            self.n_samples,
            generated_attempts,
            max_attempts,
            valid_count,
            invalid_count,
            duplicate_count,
        )

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LambdaLR | None:
        """Build the optional linear warmup/decay scheduler."""
        if self.num_warmup_steps == 0:
            return None
        from activelearning.sampler.s3gfn._optional import (
            S3GFNOptionalDependencyError,
        )

        try:
            from transformers import get_linear_schedule_with_warmup
        except ImportError as error:  # pragma: no cover - optional dependency
            raise S3GFNOptionalDependencyError(
                "S3-GFN training requires Transformers. "
                "Install it with: uv sync --extra molecules"
            ) from error
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.n_train_steps,
        )

    def _optimize(
        self,
        optimizer: torch.optim.Optimizer,
        loss: Tensor | None,
        model: S3GFNModel,
    ) -> None:
        """Apply one guarded optimizer update with gradient clipping."""
        if loss is None or not loss.requires_grad:
            return
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.policy.parameters()) + [model.log_z],
            self.gradient_clip_norm,
        )
        optimizer.step()

    def _set_round_seed(self) -> None:
        """Seed Python, PyTorch, and CUDA for the current round."""
        seed = self.seed + self._round_index
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _canonicalize_to_smiles(
    smiles: str,
    *,
    molecule_chem: Any,
) -> str | None:
    """Return a canonical connected SMILES, or ``None`` if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = molecule_chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        return None
    try:
        canonical = molecule_chem.MolToSmiles(molecule, canonical=True)
        if "." in canonical:
            return None
    except (TypeError, ValueError, RuntimeError):
        return None
    if not canonical:
        return None
    return canonical


def _score_candidates(
    acquisition: Any,
    candidates: Sequence[Candidate],
    *,
    cost_fn: Callable[[Sequence[Candidate]], list[float]] | None,
) -> list[float]:
    """Score candidates and optionally apply inverse-cost weighting."""
    if not getattr(acquisition, "supports_singleton_scoring", True):
        raise ValueError(
            f"{type(acquisition).__name__} does not support singleton scoring. "
            "S3GFNSampler requires an acquisition with score()."
        )
    scores = (
        acquisition.score(candidates)
        if cost_fn is None
        else acquisition.score(
            candidates,
            cost_weighting=cost_weighting_from_cost_fn(cost_fn),
        )
    )
    if len(scores) != len(candidates):
        raise ValueError("Acquisition returned a score count that does not align.")
    return scores

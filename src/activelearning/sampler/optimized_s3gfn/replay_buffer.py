"""Replay storage extensions used by the optimized S3-GFN sampler."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from activelearning.sampler.s3gfn.replay_buffer import ReplayBatch, ReplayBuffer


@dataclass(frozen=True)
class OptimizedReplayBatch(ReplayBatch):
    """Replay data with optional detached frozen-prior sequence scores."""

    prior_log_probabilities: Tensor | None = None
    prior_score_mask: Tensor | None = None


class OptimizedReplayBuffer(ReplayBuffer):
    """Reference replay buffer that carries deterministic prior scores.

    The base retention and sampling policies remain unchanged. Scores are
    indexed by canonical SMILES so replacement and eviction automatically
    follow the entries retained by the parent buffer.
    """

    def __init__(
        self,
        *args: object,
        prior_scores_enabled: bool = True,
        **kwargs: object,
    ) -> None:
        """Initialize optimized replay storage."""
        super().__init__(*args, **kwargs)
        self.prior_scores_enabled = bool(prior_scores_enabled)
        self._prior_scores: dict[str, float] = {}

    def add_batch(
        self,
        input_ids: Tensor,
        smiles: Sequence[str],
        reward_scores: Tensor | Sequence[float] | None = None,
        fidelity_indices: Tensor | Sequence[int] | None = None,
        prior_log_probabilities: Tensor | Sequence[float] | None = None,
    ) -> int:
        """Add trajectories and aligned detached sequence-prior scores."""
        if prior_log_probabilities is None:
            prior_values: list[float | None] = [None] * len(smiles)
        else:
            prior_tensor = torch.as_tensor(
                prior_log_probabilities,
                dtype=torch.float64,
            ).reshape(-1)
            if prior_tensor.numel() != len(smiles):
                raise ValueError(
                    "Replay prior scores and SMILES must have equal length."
                )
            prior_values = [float(value) for value in prior_tensor.tolist()]
            if not all(math.isfinite(value) for value in prior_values):
                raise ValueError("Replay prior scores must be finite.")

        added = super().add_batch(
            input_ids=input_ids,
            smiles=smiles,
            reward_scores=reward_scores,
            fidelity_indices=fidelity_indices,
        )
        if self.prior_scores_enabled:
            for smile, value in zip(smiles, prior_values, strict=True):
                normalized_smile = smile.strip()
                if normalized_smile in self._smiles and value is not None:
                    self._prior_scores[normalized_smile] = value
            self._prior_scores = {
                entry.smiles: self._prior_scores[entry.smiles]
                for entry in self._entries
                if entry.smiles in self._prior_scores
            }
        else:
            self._prior_scores.clear()
        return added

    def sample(
        self,
        count: int,
        device: str | torch.device,
        *,
        dtype: torch.dtype | None = None,
        reward_prioritized: bool = False,
        replace: bool = True,
    ) -> OptimizedReplayBatch:
        """Sample replay data and return scores for entries that have them."""
        batch = super().sample(
            count=count,
            device=device,
            dtype=dtype,
            reward_prioritized=reward_prioritized,
            replace=replace,
        )
        if not self.prior_scores_enabled or not batch.smiles:
            return OptimizedReplayBatch(
                input_ids=batch.input_ids,
                reward_scores=batch.reward_scores,
                smiles=batch.smiles,
                fidelity_indices=batch.fidelity_indices,
            )

        score_dtype = (
            batch.reward_scores.dtype
            if batch.reward_scores.is_floating_point()
            else torch.get_default_dtype()
        )
        scores = torch.zeros(
            len(batch.smiles),
            dtype=score_dtype,
            device=device,
        )
        mask = torch.zeros(
            len(batch.smiles),
            dtype=torch.bool,
            device=device,
        )
        for index, smile in enumerate(batch.smiles):
            value = self._prior_scores.get(smile)
            if value is not None:
                scores[index] = value
                mask[index] = True
        return OptimizedReplayBatch(
            input_ids=batch.input_ids,
            reward_scores=batch.reward_scores,
            smiles=batch.smiles,
            fidelity_indices=batch.fidelity_indices,
            prior_log_probabilities=scores,
            prior_score_mask=mask,
        )


__all__ = ["OptimizedReplayBatch", "OptimizedReplayBuffer"]

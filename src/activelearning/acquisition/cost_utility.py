"""Helpers for applying inverse-cost weighting to acquisition scores."""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from activelearning.utils.types import Candidate


def scale_by_cost(
    scores: Sequence[float], costs: Sequence[float], additive_offset: float = 0.0
) -> list[float]:
    """Divide scores by cost and add an optional fixed offset.

    Parameters
    ----------
    scores : Sequence[float]
        Acquisition scores to rescale.
    costs : Sequence[float]
        Positive per-item costs aligned with ``scores``.
    additive_offset : float, default=0.0
        Optional additive offset applied after inverse-cost scaling.
    """
    if len(scores) != len(costs):
        raise ValueError(
            "Scores and costs must have the same length for inverse-cost scaling."
        )
    scores_t = torch.tensor(scores, dtype=torch.float64)
    costs_t = torch.tensor(costs, dtype=torch.float64)
    if torch.any(costs_t <= 0):
        raise ValueError("Costs must be strictly positive for inverse-cost scaling.")
    return ((scores_t / costs_t) + additive_offset).tolist()


def cost_weighting_from_cost_fn(
    cost_fn: Callable[[Sequence[Candidate]], list[float]],
    additive_offset: float = 0.0,
) -> Callable[[list[float], list[Candidate]], list[float]]:
    """Build a score post-processor from a candidate-level cost function."""

    def weight_scores(scores: list[float], candidates: list[Candidate]) -> list[float]:
        return scale_by_cost(
            scores,
            cost_fn(candidates),
            additive_offset=additive_offset,
        )

    return weight_scores

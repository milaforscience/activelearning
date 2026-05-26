"""Helpers for applying inverse-cost weighting to acquisition scores."""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from activelearning.utils.types import Candidate


def scale_by_cost(
    values: torch.Tensor, costs: torch.Tensor, fixed_cost: float | torch.Tensor = 0.0
) -> torch.Tensor:
    """Divide scores by cost and add an optional fixed offset.

    Parameters
    ----------
    values : torch.Tensor
        Acquisition scores to rescale.
    costs : torch.Tensor
        Positive per-item costs aligned with ``values``.
    fixed_cost : float or torch.Tensor, default=0.0
        Optional additive offset applied after inverse-cost scaling.
    """
    if torch.any(costs <= 0):
        raise ValueError("Costs must be strictly positive for inverse-cost scaling.")
    return (values / costs) + fixed_cost


def normalize_scores_by_cost(
    scores: Sequence[float], costs: Sequence[float], fixed_cost: float = 0.0
) -> list[float]:
    """Apply inverse-cost weighting to a Python sequence of scores."""
    if len(scores) != len(costs):
        raise ValueError(
            "Scores and costs must have the same length for inverse-cost scaling."
        )
    scaled = scale_by_cost(
        torch.tensor(scores, dtype=torch.float64),
        torch.tensor(costs, dtype=torch.float64),
        fixed_cost=fixed_cost,
    )
    return scaled.tolist()


def cost_weighting_from_cost_fn(
    cost_fn: Callable[[Sequence[Candidate]], list[float]],
    fixed_cost: float = 0.0,
) -> Callable[[list[float], list[Candidate]], list[float]]:
    """Build a score post-processor from a candidate-level cost function."""

    def weight_scores(scores: list[float], candidates: list[Candidate]) -> list[float]:
        return normalize_scores_by_cost(
            scores,
            cost_fn(candidates),
            fixed_cost=fixed_cost,
        )

    return weight_scores

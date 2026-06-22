"""Pydantic model for the configuration of the budget schedules."""

import torch
from typing import Callable


def constant_schedule(value: float) -> Callable[[int], float]:
    """Return a schedule that allocates a fixed budget every round.

    Parameters
    ----------
    value : float
        Budget allocated per round, regardless of round index.

    Returns
    -------
    Callable[[int], float]
        A callable that accepts a round index and returns ``value``.
    """

    def schedule(_: int) -> float:
        return value

    return schedule


def sigmoid_iteration_schedule(
    total_budget: float,
    num_iterations: int,
    midpoint_fraction: float,
    steepness: float,
) -> Callable[[int], float]:
    """Return a schedule that distributes budget according to a sigmoid curve.

    The per-round allocation is derived from increments of the sigmoid CDF,
    so cumulative spend follows an S-curve. Early rounds receive a smaller
    share; spending accelerates around ``midpoint_fraction * num_iterations``
    and tapers off thereafter. All allocations sum exactly to ``total_budget``.

    Uses ``torch.sigmoid`` internally, which is numerically stable for any
    finite steepness value.

    Parameters
    ----------
    total_budget : float
        Total budget to distribute across all rounds.
    num_iterations : int
        Number of rounds over which the budget is distributed.
    midpoint_fraction : float
        Fraction of total iterations at which the sigmoid is centred,
        i.e. where spending rate is highest. Must be in ``(0, 1)``.
    steepness : float
        Controls how sharply spending accelerates around the midpoint.
        Higher values produce a more step-like allocation.

    Returns
    -------
    Callable[[int], float]
        A callable that accepts a round index and returns the pre-computed
        budget allocation for that round. Returns ``0.0`` for out-of-range
        indices.

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.
    """
    if total_budget <= 0:
        raise ValueError(f"total_budget must be positive, got {total_budget}.")
    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}.")
    if not (0 < midpoint_fraction < 1):
        raise ValueError(
            f"midpoint_fraction must be in (0, 1), got {midpoint_fraction}."
        )
    if steepness <= 0:
        raise ValueError(f"steepness must be positive, got {steepness}.")

    def _sigmoid(x: float) -> float:
        return torch.sigmoid(torch.tensor(steepness * (x - midpoint_fraction))).item()

    # Use CDF increments so cumulative spend follows a sigmoid.
    weights = [
        _sigmoid((i + 1) / num_iterations) - _sigmoid(i / num_iterations)
        for i in range(num_iterations)
    ]
    weight_sum = sum(weights)

    # Fallback: if all increments are numerically zero (e.g. steepness so small
    # that the sigmoid is flat over the iteration range), distribute uniformly.
    if weight_sum == 0.0:
        allocations = [total_budget / num_iterations] * num_iterations
    else:
        allocations = [total_budget * weight / weight_sum for weight in weights]

    def schedule(current_round: int) -> float:
        if current_round < 0 or current_round >= num_iterations:
            return 0.0
        return allocations[current_round]

    return schedule

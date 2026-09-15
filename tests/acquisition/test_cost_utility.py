import pytest

from activelearning.acquisition.cost_utility import (
    cost_weighting_from_cost_fn,
    scale_by_cost,
)
from activelearning.utils.types import Candidate


def test_scale_by_cost_scales_scores() -> None:
    scaled = scale_by_cost([4.0, 10.0, 6.0], [2.0, 5.0, 2.0], 0.25)
    assert scaled == [2.25, 2.25, 3.25]


def test_cost_weighting_from_cost_fn_uses_candidate_costs() -> None:
    candidates = [
        Candidate(x=[0.0], fidelity=1),
        Candidate(x=[1.0], fidelity=2),
        Candidate(x=[2.0], fidelity=1),
    ]

    def cost_fn(cands):
        return [2.0 if candidate.fidelity == 1 else 5.0 for candidate in cands]

    weight_scores = cost_weighting_from_cost_fn(cost_fn, additive_offset=0.25)

    scaled = weight_scores([4.0, 10.0, 6.0], candidates)

    assert scaled == [2.25, 2.25, 3.25]


def test_non_positive_costs_raise_during_scaling() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        scale_by_cost([1.0], [0.0])


def test_cost_weighting_from_cost_fn_checks_lengths() -> None:
    def cost_fn(cands):
        return [1.0]

    weight_scores = cost_weighting_from_cost_fn(cost_fn)

    with pytest.raises(ValueError, match="same length"):
        weight_scores([1.0, 2.0], [Candidate(x=1), Candidate(x=2)])

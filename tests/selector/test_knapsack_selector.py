from unittest.mock import Mock

import pytest

from activelearning.selector import knapsack_selector as knapsack_selector_module
from activelearning.selector.knapsack_selector import (
    KnapsackSelector,
    greedy_knapsack_indices,
)
from activelearning.utils.types import Candidate


def test_greedy_knapsack_indices_selects_best_ratios_with_zero_cost():
    """Test greedy helper prioritizes zero-cost items and respects budget."""
    values = [10.0, 20.0, 30.0, 40.0]
    costs = [5.0, 0.0, 6.0, 2.0]

    selected_indices = greedy_knapsack_indices(values, costs, budget=7.0)

    assert selected_indices == [1, 3, 0]


def test_greedy_knapsack_indices_raises_on_length_mismatch():
    """Test greedy helper validates aligned value and cost arrays."""
    with pytest.raises(ValueError, match="same length"):
        greedy_knapsack_indices([1.0], [1.0, 2.0], budget=3.0)


def test_greedy_knapsack_indices_breaks_ratio_ties_by_value_then_index():
    """Test greedy helper preserves the documented tie-break order."""
    values = [10.0, 8.0, 10.0]
    costs = [5.0, 4.0, 5.0]

    selected_indices = greedy_knapsack_indices(values, costs, budget=10.0)

    assert selected_indices == [0, 2]


@pytest.mark.parametrize("warm_start", [False, True])
def test_knapsack_selector_finds_exact_solution_when_greedy_misses(warm_start):
    """Test exact knapsack solve beats the greedy warm-start heuristic."""
    candidates = [Candidate(x=0), Candidate(x=1)]
    acquisition = Mock()
    acquisition.score.return_value = [10.0, 9.0]

    def cost_fn(_candidates):
        return [6.0, 5.0]

    selector = KnapsackSelector(warm_start=warm_start)
    selected = selector(
        candidates,
        acquisition=acquisition,
        cost_fn=cost_fn,
        round_budget=10.0,
    )

    assert [candidate.x for candidate in selected] == [0]


@pytest.mark.parametrize(("warm_start", "expected_calls"), [(False, 0), (True, 1)])
def test_knapsack_selector_uses_greedy_helper_only_for_warm_start(
    monkeypatch, warm_start, expected_calls
):
    """Test greedy helper is consulted only when warm-starting is enabled."""
    helper_calls = []
    original_helper = knapsack_selector_module.greedy_knapsack_indices

    def recording_helper(values, costs, budget):
        helper_calls.append((list(values), list(costs), budget))
        return original_helper(values, costs, budget)

    monkeypatch.setattr(
        knapsack_selector_module,
        "greedy_knapsack_indices",
        recording_helper,
    )

    acquisition = Mock()
    acquisition.score.return_value = [8.0, 5.0]

    selector = KnapsackSelector(warm_start=warm_start)
    selector(
        [Candidate(x=0), Candidate(x=1)],
        acquisition=acquisition,
        cost_fn=lambda _candidates: [4.0, 3.0],
        round_budget=4.0,
    )

    assert len(helper_calls) == expected_calls


def test_knapsack_selector_returns_empty_for_empty_candidates():
    """Test selector returns an empty list when the candidate pool is empty."""
    acquisition = Mock()
    acquisition.score.return_value = []
    selector = KnapsackSelector()

    selected = selector(
        [],
        acquisition=acquisition,
        cost_fn=lambda _candidates: [],
        round_budget=1.0,
    )

    assert selected == []

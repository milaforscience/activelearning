import random
import warnings
from unittest.mock import Mock

import pulp
import pytest

from activelearning.acquisition.dummy_acquisition import DummyAcquisition
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


def test_greedy_knapsack_indices_misses_optimal_solution():
    """Test greedy helper is suboptimal when a lower-ratio item dominates.

    Greedy sorts by value/cost ratio and picks:
      - item 1 (value=6, cost=5, ratio=1.20) first,
      - then item 0 (value=7, cost=6, ratio=1.17) is too expensive,
      - then item 2 (value=10, cost=9, ratio=1.11) is too expensive,
    yielding total value 6. The true optimum is item 2 alone (value=10).
    """
    values = [7.0, 6.0, 10.0]
    costs = [6.0, 5.0, 9.0]

    selected_indices = greedy_knapsack_indices(values, costs, budget=10.0)

    # Greedy picks only item 1 (best ratio), missing the better single-item solution
    assert selected_indices == [1]
    assert sum(values[i] for i in selected_indices) < values[2]


def test_greedy_knapsack_indices_misses_optimal_multi_item_solution():
    """Test greedy helper is suboptimal when filling with high-ratio cheap items
    blocks a higher-value combination that includes a lower-ratio expensive item.

    Greedy sorts by value/cost ratio and picks:
      - items 0, 1, 2 (each value=3, cost=2, ratio=1.50) — exhausting 6 units of budget,
      - item 3 (value=8, cost=6, ratio=1.33) is then too expensive for the remaining 4,
    yielding total value 9.

    The optimal solution is items 0 + 1 + 3 (cost=10, value=14).
    """
    values = [3.0, 3.0, 3.0, 8.0]
    costs = [2.0, 2.0, 2.0, 6.0]

    selected_indices = greedy_knapsack_indices(values, costs, budget=10.0)

    # Greedy fills with three cheap items and misses the optimal three-item set
    assert selected_indices == [0, 1, 2]
    greedy_value = sum(values[i] for i in selected_indices)
    optimal_value = values[0] + values[1] + values[3]  # items 0, 1, 3
    assert greedy_value < optimal_value


def test_greedy_knapsack_indices_breaks_ratio_ties_by_value_then_index():
    """Test greedy helper preserves the documented tie-break order."""
    values = [10.0, 8.0, 10.0]
    costs = [5.0, 4.0, 5.0]

    selected_indices = greedy_knapsack_indices(values, costs, budget=10.0)

    assert selected_indices == [0, 2]


@pytest.mark.parametrize("warm_start", [False, True])
def test_knapsack_selector_finds_exact_solution_when_greedy_misses(warm_start):
    """Test exact solving finds the optimum when the greedy MIP start does not."""
    candidates = [Candidate(x=0), Candidate(x=1)]
    acquisition = Mock()
    acquisition.score.return_value = [10.0, 9.0]

    def cost_fn(_candidates):
        return [6.0, 5.0]

    selector = KnapsackSelector(warm_start=warm_start)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selected = selector(
            candidates,
            acquisition=acquisition,
            cost_fn=cost_fn,
            round_budget=10.0,
        )

    assert [candidate.x for candidate in selected] == [0]
    acquisition.score.assert_called_once_with(candidates)


@pytest.mark.parametrize(("warm_start", "expected_calls"), [(False, 0), (True, 1)])
def test_knapsack_selector_uses_greedy_helper_only_for_warm_start(
    monkeypatch, warm_start, expected_calls
):
    """Test the greedy MIP-start helper runs only when warm-starting is enabled."""
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
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector(
            [Candidate(x=0), Candidate(x=1)],
            acquisition=acquisition,
            cost_fn=lambda _candidates: [4.0, 3.0],
            round_budget=4.0,
        )

    assert len(helper_calls) == expected_calls


def test_knapsack_selector_warns_when_warm_start_used_without_time_limit():
    """Test warm-starting without a time limit emits a UserWarning."""
    selector = KnapsackSelector(warm_start=True)
    acquisition = Mock()
    acquisition.score.return_value = [3.0, 2.0]

    with pytest.warns(UserWarning, match="no benefit without a time_limit"):
        selector(
            [Candidate(x=0), Candidate(x=1)],
            acquisition=acquisition,
            cost_fn=lambda _candidates: [1.0, 1.0],
            round_budget=2.0,
        )


def test_knapsack_selector_returns_empty_for_empty_candidates():
    """Test selector returns an empty list when the candidate pool is empty."""
    acquisition = Mock()
    acquisition.return_value = []
    selector = KnapsackSelector()

    selected = selector(
        [],
        acquisition=acquisition,
        cost_fn=lambda _candidates: [],
        round_budget=1.0,
    )

    assert selected == []


def test_knapsack_selector_rejects_negative_costs():
    """Test selector rejects negative per-candidate costs."""
    selector = KnapsackSelector()
    acquisition = Mock()
    acquisition.score.return_value = [3.0, 2.0]

    with pytest.raises(ValueError, match="negative cost"):
        selector(
            [Candidate(x=0), Candidate(x=1)],
            acquisition=acquisition,
            cost_fn=lambda _candidates: [1.0, -1.0],
            round_budget=2.0,
        )


def test_knapsack_selector_rejects_negative_acquisition_scores():
    """Test selector rejects scores that do not satisfy its utility contract."""
    selector = KnapsackSelector()
    acquisition = Mock()
    acquisition.score.return_value = [3.0, -0.5]

    with pytest.raises(
        ValueError,
        match="requires non-negative, additive acquisition scores",
    ):
        selector(
            [Candidate(x=0), Candidate(x=1)],
            acquisition=acquisition,
            cost_fn=lambda _candidates: [1.0, 1.0],
            round_budget=2.0,
        )


def test_knapsack_selector_uses_acquisition_callable_interface():
    """Test selector works with a real Acquisition implementation."""
    selector = KnapsackSelector()
    acquisition = DummyAcquisition()

    selected = selector(
        [Candidate(x=0), Candidate(x=1)],
        acquisition=acquisition,
        cost_fn=lambda _candidates: [2.0, 1.0],
        round_budget=1.0,
    )

    assert [candidate.x for candidate in selected] == [1]


def test_knapsack_selector_raises_for_unusable_solver_status(monkeypatch):
    """Test selector raises cleanly on infeasible solver outcomes."""
    selector = KnapsackSelector()
    acquisition = Mock()
    acquisition.score.return_value = [3.0, 2.0]

    def fake_solve(self, _solver):
        return pulp.LpStatusInfeasible

    monkeypatch.setattr(knapsack_selector_module.pulp.LpProblem, "solve", fake_solve)

    with pytest.raises(ValueError, match="unusable status 'Infeasible'"):
        selector(
            [Candidate(x=0), Candidate(x=1)],
            acquisition=acquisition,
            cost_fn=lambda _candidates: [1.0, 1.0],
            round_budget=1.0,
        )


def test_knapsack_selector_raises_when_not_solved_has_no_incumbent(monkeypatch):
    """Test selector rejects time-limited runs without variable assignments."""
    selector = KnapsackSelector()
    acquisition = Mock()
    acquisition.score.return_value = [3.0, 2.0]

    def fake_solve(self, _solver):
        return pulp.LpStatusNotSolved

    monkeypatch.setattr(knapsack_selector_module.pulp.LpProblem, "solve", fake_solve)

    with pytest.raises(ValueError, match="without a usable incumbent solution"):
        selector(
            [Candidate(x=0), Candidate(x=1)],
            acquisition=acquisition,
            cost_fn=lambda _candidates: [1.0, 1.0],
            round_budget=1.0,
        )


def test_knapsack_selector_uses_incumbent_when_not_solved(monkeypatch, capsys):
    """Test selector can use CBC incumbents from a non-optimal solve."""
    selector = KnapsackSelector()
    acquisition = Mock()
    acquisition.score.return_value = [3.0, 2.0]

    def fake_solve(self, _solver):
        for variable, value in zip(self.variables(), [1.0, 0.0]):
            variable.varValue = value
        return pulp.LpStatusNotSolved

    monkeypatch.setattr(knapsack_selector_module.pulp.LpProblem, "solve", fake_solve)

    selected = selector(
        [Candidate(x=0), Candidate(x=1)],
        acquisition=acquisition,
        cost_fn=lambda _candidates: [1.0, 1.0],
        round_budget=1.0,
    )

    assert [candidate.x for candidate in selected] == [0]
    assert "Using the best available solution found so far" in capsys.readouterr().out


def test_knapsack_selector_early_stops_on_solver_time_limit(monkeypatch, capsys):
    """Test CBC returns a feasible incumbent when its time limit is reached."""
    time_limit = 0.05
    item_count = 5_000
    candidates = [Candidate(x=index) for index in range(item_count)]
    random_generator = random.Random(0)
    costs = [random_generator.randint(1, 1_000) for _ in range(item_count)]
    values = [random_generator.randint(1, 1_000) for _ in range(item_count)]
    budget = sum(costs) // 2
    selector = KnapsackSelector(time_limit=time_limit, warm_start=True)
    acquisition = Mock()
    acquisition.score.return_value = values

    recorded_time_limits = []
    recorded_statuses = []
    original_solve = knapsack_selector_module.pulp.LpProblem.solve

    def recording_solve(self, solver, **kwargs):
        """Run CBC and retain its actual status values for assertions."""
        recorded_time_limits.append(solver.timeLimit)
        status = original_solve(self, solver, **kwargs)
        recorded_statuses.append((status, self.sol_status))
        return status

    monkeypatch.setattr(
        knapsack_selector_module.pulp.LpProblem,
        "solve",
        recording_solve,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Constructing LpVariable",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="PULP_CBC_CMD is deprecated",
            category=DeprecationWarning,
        )
        selected = selector(
            candidates,
            acquisition=acquisition,
            cost_fn=lambda _candidates: costs,
            round_budget=budget,
        )

    assert recorded_time_limits == [time_limit]
    # PuLP records a CBC time-limited incumbent as a feasible, non-optimal
    # solution even though its coarse problem status is ``LpStatusOptimal``.
    assert recorded_statuses == [(pulp.LpStatusOptimal, pulp.LpSolutionIntegerFeasible)]
    assert selected
    assert sum(costs[candidate.x] for candidate in selected) <= budget
    assert "Using the best available solution found so far" in capsys.readouterr().out


def test_knapsack_selector_raises_when_cbc_is_unavailable(monkeypatch):
    """Test selector fails early with a clear CBC availability error."""
    selector = KnapsackSelector()
    acquisition = Mock()
    acquisition.score.return_value = [3.0, 2.0]

    monkeypatch.setattr(
        knapsack_selector_module.pulp.PULP_CBC_CMD,
        "available",
        lambda self: False,
    )

    with pytest.raises(ValueError, match="CBC solver is not available"):
        selector(
            [Candidate(x=0), Candidate(x=1)],
            acquisition=acquisition,
            cost_fn=lambda _candidates: [1.0, 1.0],
            round_budget=1.0,
        )

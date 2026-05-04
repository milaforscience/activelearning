from unittest.mock import Mock

import pytest

from activelearning.selector.cost_aware_selector import CostAwareSelector
from activelearning.utils.types import Candidate


@pytest.fixture
def selector():
    """Create a CostAwareSelector instance."""
    return CostAwareSelector()


@pytest.fixture
def candidates():
    """Create a list of test candidates."""
    return [Candidate(x=i) for i in range(5)]


@pytest.fixture
def mock_acquisition():
    """Create a mock acquisition function."""
    acquisition = Mock()
    acquisition.supports_singleton_scoring = True
    acquisition.score.return_value = [10.0, 20.0, 30.0, 40.0, 50.0]
    return acquisition


@pytest.fixture
def uniform_cost_fn():
    """Cost function returning uniform costs."""

    def cost_fn(candidates):
        return [5.0 for _ in candidates]

    return cost_fn


@pytest.fixture
def varying_cost_fn():
    """Cost function returning varying costs."""

    def cost_fn(candidates):
        return [float(i + 1) for i in range(len(candidates))]

    return cost_fn


def test_raises_without_acquisition(selector, candidates, uniform_cost_fn):
    """Test that selector raises ValueError when acquisition is None."""
    with pytest.raises(ValueError, match="Acquisition function is required"):
        selector(
            candidates, acquisition=None, cost_fn=uniform_cost_fn, round_budget=100.0
        )


def test_raises_without_cost_fn(selector, candidates, mock_acquisition):
    """Test that selector raises ValueError when cost_fn is None."""
    with pytest.raises(ValueError, match="Cost function is required"):
        selector(
            candidates, acquisition=mock_acquisition, cost_fn=None, round_budget=100.0
        )


def test_raises_without_budget(selector, candidates, mock_acquisition, uniform_cost_fn):
    """Test that selector raises ValueError when round_budget is None."""
    with pytest.raises(ValueError, match="Budget is required"):
        selector(
            candidates,
            acquisition=mock_acquisition,
            cost_fn=uniform_cost_fn,
            round_budget=None,
        )


def test_empty_candidates(selector, mock_acquisition, uniform_cost_fn):
    """Test selector returns empty list for empty candidates."""
    result = selector(
        [], acquisition=mock_acquisition, cost_fn=uniform_cost_fn, round_budget=100.0
    )
    assert result == []


def test_selects_by_utility_cost_ratio(selector, candidates):
    """Test that selector ranks candidates by acquisition value/cost ratio."""
    # Acquisition values: [10, 20, 30, 40, 50]
    # Costs:              [10, 5,  2,  1,  1]
    # Ratios:             [1,  4,  15, 40, 50] <- descending order: 4, 3, 2, 1, 0
    acquisition = Mock()
    acquisition.supports_singleton_scoring = True
    acquisition.score.return_value = [10.0, 20.0, 30.0, 40.0, 50.0]

    def cost_fn(c):
        return [10.0, 5.0, 2.0, 1.0, 1.0]

    selected = selector(
        candidates, acquisition=acquisition, cost_fn=cost_fn, round_budget=100.0
    )

    # With budget 100, all should be selected, sorted by bang-for-buck
    assert len(selected) == 5
    # Candidate 4 has ratio 50/1=50 (best)
    # Candidate 3 has ratio 40/1=40
    # Candidate 2 has ratio 30/2=15
    # Candidate 1 has ratio 20/5=4
    # Candidate 0 has ratio 10/10=1
    assert selected[0].x == 4
    assert selected[1].x == 3
    assert selected[2].x == 2
    assert selected[3].x == 1
    assert selected[4].x == 0


def test_stops_when_budget_exhausted(selector, candidates):
    """Test that selector stops selecting when budget is exhausted."""
    # Acquisition values: [50, 40, 30, 20, 10]
    # Costs:              [1,  1,  2,  5,  10]
    # Ratios:             [50, 40, 15, 4,  1] <- sorted by ratio descending
    acquisition = Mock()
    acquisition.supports_singleton_scoring = True
    acquisition.score.return_value = [50.0, 40.0, 30.0, 20.0, 10.0]

    def cost_fn(c):
        return [1.0, 1.0, 2.0, 5.0, 10.0]

    # Budget is 3.0 - should select candidates 0 (cost 1) and 1 (cost 1), total 2.0
    # Cannot add candidate 2 (cost 2) as 2+2=4 > 3
    selected = selector(
        candidates, acquisition=acquisition, cost_fn=cost_fn, round_budget=3.0
    )

    assert len(selected) == 2
    assert selected[0].x == 0  # Best ratio: 50/1
    assert selected[1].x == 1  # Second best: 40/1


def test_uniform_costs_varying_utilities(selector, candidates, uniform_cost_fn):
    """Test selection with uniform costs and varying acquisition values."""
    # All costs are 5.0, so ranking is purely by acquisition value
    # Acquisition values: [10, 20, 30, 40, 50]
    acquisition = Mock()
    acquisition.supports_singleton_scoring = True
    acquisition.score.return_value = [10.0, 20.0, 30.0, 40.0, 50.0]

    # Budget 12.0 allows 2 candidates (2 * 5.0 = 10.0)
    selected = selector(
        candidates, acquisition=acquisition, cost_fn=uniform_cost_fn, round_budget=12.0
    )

    assert len(selected) == 2
    assert selected[0].x == 4  # Highest acquisition value
    assert selected[1].x == 3  # Second highest


def test_varying_costs_uniform_utilities(selector, candidates, varying_cost_fn):
    """Test selection with varying costs and uniform acquisition values."""
    # All acquisition values are 100.0
    # Costs: [1, 2, 3, 4, 5]
    # Ratios: [100, 50, 33.3, 25, 20] <- descending
    acquisition = Mock()
    acquisition.supports_singleton_scoring = True
    acquisition.score.return_value = [100.0, 100.0, 100.0, 100.0, 100.0]

    # Budget 6.0 allows candidates 0 (cost 1), 1 (cost 2), 2 (cost 3) = 6.0 total
    selected = selector(
        candidates, acquisition=acquisition, cost_fn=varying_cost_fn, round_budget=6.0
    )

    assert len(selected) == 3
    assert selected[0].x == 0  # Lowest cost, best ratio
    assert selected[1].x == 1
    assert selected[2].x == 2


def test_zero_cost_candidate(selector):
    """Test handling of candidate with zero cost (infinite ratio)."""
    candidates = [Candidate(x=0), Candidate(x=1), Candidate(x=2)]
    acquisition = Mock()
    acquisition.supports_singleton_scoring = True
    acquisition.score.return_value = [10.0, 20.0, 30.0]

    def cost_fn(c):
        return [0.0, 5.0, 10.0]  # First candidate has zero cost

    selected = selector(
        candidates, acquisition=acquisition, cost_fn=cost_fn, round_budget=15.0
    )

    # Zero-cost candidate should be selected first (infinite ratio)
    assert len(selected) == 3
    assert selected[0].x == 0  # Zero cost = infinite ratio


def test_negative_cost_candidate(selector):
    """Test that negative candidate costs raise ValueError."""
    candidates = [Candidate(x=0), Candidate(x=1), Candidate(x=2)]
    acquisition = Mock()
    acquisition.supports_singleton_scoring = True
    acquisition.score.return_value = [10.0, 20.0, 30.0]

    def cost_fn(c):
        return [-5.0, 5.0, 10.0]  # First candidate has negative cost

    with pytest.raises(ValueError, match="negative cost"):
        selector(
            candidates, acquisition=acquisition, cost_fn=cost_fn, round_budget=15.0
        )


def test_zero_budget(selector, candidates, mock_acquisition, uniform_cost_fn):
    """Test that zero budget results in no selections."""
    selected = selector(
        candidates,
        acquisition=mock_acquisition,
        cost_fn=uniform_cost_fn,
        round_budget=0.0,
    )

    assert len(selected) == 0


def test_exact_budget_fit(selector):
    """Test selection when candidates exactly fit budget."""
    candidates = [Candidate(x=i) for i in range(3)]
    acquisition = Mock()
    acquisition.supports_singleton_scoring = True
    acquisition.score.return_value = [30.0, 20.0, 10.0]

    def cost_fn(c):
        return [5.0, 5.0, 5.0]

    # Budget exactly 15.0 = 3 candidates * 5.0
    selected = selector(
        candidates, acquisition=acquisition, cost_fn=cost_fn, round_budget=15.0
    )

    assert len(selected) == 3


def test_greedy_not_optimal(selector):
    """Test that greedy utility/cost ranking can miss the max-utility feasible set."""
    # Classic knapsack counter-example
    candidates = [Candidate(x=0), Candidate(x=1)]
    acquisition = Mock()
    acquisition.supports_singleton_scoring = True
    acquisition.score.return_value = [10.0, 9.0]  # Candidate 0 slightly better utility

    def cost_fn(c):
        return [6.0, 5.0]  # Candidate 0 costs more

    # Budget 10.0 - greedy by ratio picks candidate 1 first (ratio 9/5=1.8),
    # then cannot fit candidate 0 (cost 6.0) in the remaining budget.
    # The true max-utility feasible set is candidate 0 alone (utility 10.0 > 9.0).
    selected = selector(
        candidates, acquisition=acquisition, cost_fn=cost_fn, round_budget=10.0
    )

    # Greedy will select candidate 1 first (better ratio), then can't fit candidate 0
    assert len(selected) == 1
    assert selected[0].x == 1

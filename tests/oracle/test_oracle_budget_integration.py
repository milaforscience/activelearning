from unittest.mock import Mock

import pytest

from activelearning.budget.budget import Budget
from activelearning.oracle.dummy_oracle import DummyOracle
from activelearning.utils.types import Candidate


@pytest.fixture
def mock_budget():
    """Create a mock Budget for testing."""
    budget = Mock(spec=Budget)
    budget.available_budget = 100.0
    return budget


@pytest.fixture
def oracle():
    """Create a DummyOracle with cost_per_sample=2.0."""
    return DummyOracle(cost_per_sample=2.0, score_fn=lambda x: float(x) * 2)


@pytest.fixture
def candidates():
    """Create a list of test candidates."""
    return [Candidate(x=i) for i in range(5)]


def test_get_costs_returns_list_of_costs(oracle, candidates):
    """Test that get_costs returns a list with one cost per candidate."""
    costs = oracle.get_costs(candidates)

    assert isinstance(costs, list)
    assert len(costs) == len(candidates)
    assert all(cost == 2.0 for cost in costs)


def test_get_costs_empty_candidates(oracle):
    """Test get_costs with empty candidate list."""
    costs = oracle.get_costs([])

    assert isinstance(costs, list)
    assert len(costs) == 0


def test_get_costs_single_candidate(oracle):
    """Test get_costs with single candidate."""
    candidates = [Candidate(x=10)]
    costs = oracle.get_costs(candidates)

    assert len(costs) == 1
    assert costs[0] == 2.0


def test_query_calls_budget_consume_with_correct_total(oracle, candidates, mock_budget):
    """Test that query calls budget.consume with the sum of costs."""
    oracle.query(candidates, mock_budget)

    # Verify consume was called once with total cost
    mock_budget.consume.assert_called_once_with(10.0)  # 5 candidates * 2.0 each


def test_query_returns_correct_observations(oracle, candidates):
    """Test that query returns correct observations despite budget integration."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)

    observations = oracle.query(candidates, budget)

    assert len(observations) == len(candidates)
    for i, obs in enumerate(observations):
        assert obs.x == i
        assert obs.y == i * 2.0
        assert obs.fidelity is None


def test_query_consumes_budget_correctly(oracle, candidates):
    """Test that query correctly consumes budget."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)

    oracle.query(candidates, budget)

    # 5 candidates * 2.0 cost each = 10.0 consumed
    assert budget.available_budget == 90.0


def test_query_raises_when_budget_insufficient(oracle, candidates):
    """Test that query raises ValueError when budget is insufficient."""
    # Budget only has 5.0, but 5 candidates cost 10.0
    budget = Budget(available_budget=5.0, schedule=lambda r: 10.0)

    with pytest.raises(ValueError, match="Cost .* exceeds available budget"):
        oracle.query(candidates, budget)

    # Budget should remain unchanged
    assert budget.available_budget == 5.0


def test_query_multiple_calls_deplete_budget(oracle):
    """Test multiple query calls correctly deplete budget."""
    budget = Budget(available_budget=50.0, schedule=lambda r: 10.0)

    # First query: 3 candidates * 2.0 = 6.0
    oracle.query([Candidate(x=i) for i in range(3)], budget)
    assert budget.available_budget == 44.0

    # Second query: 5 candidates * 2.0 = 10.0
    oracle.query([Candidate(x=i) for i in range(5)], budget)
    assert budget.available_budget == 34.0

    # Third query: 10 candidates * 2.0 = 20.0
    oracle.query([Candidate(x=i) for i in range(10)], budget)
    assert budget.available_budget == 14.0


def test_query_with_varying_cost_oracle():
    """Test query with oracle that has varying costs per candidate."""

    class VaryingCostOracle(DummyOracle):
        def get_costs(self, candidates):
            # Cost increases with candidate index
            return [float(i + 1) for i in range(len(candidates))]

    oracle = VaryingCostOracle(score_fn=lambda x: float(x))
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
    candidates = [Candidate(x=i) for i in range(4)]

    # Costs will be [1.0, 2.0, 3.0, 4.0] = 10.0 total
    oracle.query(candidates, budget)

    assert budget.available_budget == 90.0


def test_query_with_fidelity_candidates(oracle):
    """Test query with candidates that have fidelity levels."""
    budget = Budget(available_budget=100.0, schedule=lambda r: 10.0)
    candidates = [
        Candidate(x=1, fidelity=0),
        Candidate(x=2, fidelity=1),
        Candidate(x=3, fidelity=0),
    ]

    observations = oracle.query(candidates, budget)

    # 3 candidates * 2.0 = 6.0 consumed
    assert budget.available_budget == 94.0
    assert len(observations) == 3
    assert observations[0].fidelity == 0
    assert observations[1].fidelity == 1
    assert observations[2].fidelity == 0

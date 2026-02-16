import pytest

from activelearning.budget.budget import Budget
from activelearning.oracle.dummy_oracle import DummyOracle
from activelearning.utils.types import Candidate, Observation


@pytest.fixture(params=[1.0, 2.5])
def cost_per_sample(request):
    return request.param


@pytest.fixture
def default_score_fn():
    def score_fn(x):
        return float(x)

    return score_fn


@pytest.fixture
def custom_score_fn():
    def score_fn(x):
        return x * 2.0

    return score_fn


@pytest.fixture
def candidate_x_values():
    return [5, 10, 15]


@pytest.fixture
def budget():
    """Create a Budget with sufficient funds for testing."""
    return Budget(available_budget=1000.0, schedule=lambda r: 100.0)


def test_get_costs_single_candidate(cost_per_sample):
    """Test cost calculation for a single candidate."""
    oracle = DummyOracle(cost_per_sample=cost_per_sample)
    candidates = [Candidate(x=10)]
    costs = oracle.get_costs(candidates)
    assert isinstance(costs, list)
    assert len(costs) == 1
    assert costs[0] == cost_per_sample


def test_get_costs_multiple_candidates(cost_per_sample, candidate_x_values):
    """Test cost calculation for multiple candidates."""
    oracle = DummyOracle(cost_per_sample=cost_per_sample)
    candidates = [Candidate(x=x) for x in candidate_x_values]
    costs = oracle.get_costs(candidates)
    assert isinstance(costs, list)
    assert len(costs) == len(candidate_x_values)
    assert all(cost == cost_per_sample for cost in costs)


def test_query_with_default_score_fn(default_score_fn, candidate_x_values, budget):
    """Test querying candidates with the default score function."""
    oracle = DummyOracle(score_fn=default_score_fn)
    candidates = [Candidate(x=x) for x in candidate_x_values]
    observations = oracle.query(candidates, budget)

    assert len(observations) == len(candidate_x_values)
    for obs, x_val in zip(observations, candidate_x_values):
        assert isinstance(obs, Observation)
        assert obs.x == x_val
        assert obs.y == float(x_val)
        assert obs.fidelity is None


def test_query_with_custom_score_fn(custom_score_fn, budget):
    """Test querying candidates with a custom score function."""
    oracle = DummyOracle(score_fn=custom_score_fn)
    candidates = [Candidate(x=3), Candidate(x=7)]
    observations = oracle.query(candidates, budget)

    assert len(observations) == 2
    assert isinstance(observations[0], Observation)
    assert observations[0].x == 3
    assert observations[0].y == 6.0
    assert observations[0].fidelity is None
    assert isinstance(observations[1], Observation)
    assert observations[1].x == 7
    assert observations[1].y == 14.0
    assert observations[1].fidelity is None


def test_query_with_fidelity_candidates(cost_per_sample, budget):
    """Test querying candidates with fidelity levels."""
    oracle = DummyOracle(
        cost_per_sample=cost_per_sample, score_fn=lambda x: float(x) + 0.5
    )
    candidates = [Candidate(x=10, fidelity=0), Candidate(x=20, fidelity=1)]

    costs = oracle.get_costs(candidates)
    observations = oracle.query(candidates, budget)

    assert all(cost == cost_per_sample for cost in costs)
    assert len(observations) == 2
    assert isinstance(observations[0], Observation)
    assert observations[0].x == 10
    assert observations[0].y == 10.5
    assert observations[0].fidelity == 0
    assert isinstance(observations[1], Observation)
    assert observations[1].x == 20
    assert observations[1].y == 20.5
    assert observations[1].fidelity == 1

import pytest

from activelearning.oracle.dummy_oracle import DummyOracle
from activelearning.utils.types import Candidate


@pytest.fixture(params=[1.0, 2.5])
def cost_per_sample(request):
    return request.param


@pytest.fixture
def default_score_fn():
    return lambda x: float(x)


@pytest.fixture
def custom_score_fn():
    return lambda x: x * 2.0


@pytest.fixture
def candidate_x_values():
    return [5, 10, 15]


def test_get_cost_single_candidate(cost_per_sample):
    """Test cost calculation for a single candidate."""
    oracle = DummyOracle(cost_per_sample=cost_per_sample)
    candidates = [Candidate(x=10)]
    cost = oracle.get_cost(candidates)
    assert cost == cost_per_sample


def test_get_cost_multiple_candidates(cost_per_sample, candidate_x_values):
    """Test cost calculation for multiple candidates."""
    oracle = DummyOracle(cost_per_sample=cost_per_sample)
    candidates = [Candidate(x=x) for x in candidate_x_values]
    cost = oracle.get_cost(candidates)
    assert cost == cost_per_sample * len(candidate_x_values)


def test_query_with_default_score_fn(default_score_fn, candidate_x_values):
    """Test querying candidates with the default score function."""
    oracle = DummyOracle(score_fn=default_score_fn)
    candidates = [Candidate(x=x) for x in candidate_x_values]
    scores = oracle.query(candidates)

    assert len(scores) == len(candidate_x_values)
    assert scores == [float(x) for x in candidate_x_values]


def test_query_with_custom_score_fn(custom_score_fn):
    """Test querying candidates with a custom score function."""
    oracle = DummyOracle(score_fn=custom_score_fn)
    candidates = [Candidate(x=3), Candidate(x=7)]
    scores = oracle.query(candidates)

    assert len(scores) == 2
    assert scores == [6.0, 14.0]


def test_query_with_fidelity_candidates(cost_per_sample):
    """Test querying candidates with fidelity levels."""
    oracle = DummyOracle(
        cost_per_sample=cost_per_sample, score_fn=lambda x: float(x) + 0.5
    )
    candidates = [Candidate(x=10, fidelity=0), Candidate(x=20, fidelity=1)]

    cost = oracle.get_cost(candidates)
    scores = oracle.query(candidates)

    assert cost == cost_per_sample * 2
    assert scores == [10.5, 20.5]

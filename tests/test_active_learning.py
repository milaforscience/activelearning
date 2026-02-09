import pytest

from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.dataset.dummy_dataset import DummyDataset
from activelearning.oracle.dummy_oracle import DummyOracle
from activelearning.sampler.pool_score_sampler import PoolScoreSampler
from activelearning.selector.score_selector import ScoreSelector
from activelearning.surrogate.dummy_surrogate import DummySurrogate
from activelearning.active_learning import active_learning, get_best_candidates
from activelearning.utils.types import Candidate


@pytest.fixture
def dataset():
    """Create a dummy dataset for testing."""
    return DummyDataset()


@pytest.fixture
def surrogate():
    """Create a dummy surrogate model for testing."""
    return DummySurrogate()


@pytest.fixture
def acquisition():
    """Create a dummy acquisition function for testing."""
    return DummyAcquisition()


@pytest.fixture
def sampler():
    """Create a pool score sampler with multi-fidelity candidates."""
    candidate_pool = [Candidate(i, 0) for i in range(100)] + [
        Candidate(i, 1) for i in range(100)
    ]
    return PoolScoreSampler(candidate_pool=candidate_pool, num_samples=100)


@pytest.fixture
def selector():
    """Create a score-based selector for testing."""
    return ScoreSelector(num_samples=5)


@pytest.fixture
def oracles():
    """Create multi-fidelity oracles with different costs and scoring functions."""
    return {
        0: DummyOracle(cost_per_sample=1.0, score_fn=lambda s: float(s)),
        1: DummyOracle(cost_per_sample=2.0, score_fn=lambda s: float(s) + 0.5),
    }


@pytest.fixture
def budget():
    """Set the budget for active learning loop."""
    return 100.0


@pytest.fixture
def top_k():
    """Set the number of top candidates to retrieve."""
    return 3


def test_active_learning_loop(
    dataset, surrogate, acquisition, sampler, selector, oracles, budget, top_k
):
    """Test that the active learning loop completes and returns expected types."""
    dataset_out, cost, num_iter = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracles=oracles,
        budget=budget,
    )
    best = get_best_candidates(dataset_out, k=top_k)
    assert isinstance(best, list)
    assert isinstance(cost, float)
    assert isinstance(num_iter, int)

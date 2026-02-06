import pytest

from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.dataset.dummy_dataset import DummyDataset
from activelearning.oracle.dummy_oracle import DummyOracle
from activelearning.sampler.pool_score_sampler import PoolScoreSampler
from activelearning.selector.score_selector import ScoreSelector
from activelearning.surrogate.dummy_surrogate import DummySurrogate
from activelearning.active_learning import active_learning
from activelearning.utils.types import Candidate


@pytest.fixture
def dataset():
    return DummyDataset()


@pytest.fixture
def surrogate():
    return DummySurrogate()


@pytest.fixture
def acquisition():
    return DummyAcquisition()


@pytest.fixture
def sampler(acquisition):
    candidate_pool = [Candidate(i, 0) for i in range(100)] + [
        Candidate(i, 1) for i in range(100)
    ]
    return PoolScoreSampler(
        candidate_pool=candidate_pool, num_samples=100, score_fn=acquisition
    )


@pytest.fixture
def selector(acquisition):
    return ScoreSelector(score_fn=acquisition, num_samples=5)


@pytest.fixture
def oracles():
    return {
        0: DummyOracle(cost_per_sample=1.0, score_fn=lambda s: float(s)),
        1: DummyOracle(cost_per_sample=2.0, score_fn=lambda s: float(s) + 0.5),
    }


@pytest.fixture
def budget():
    return 100.0


@pytest.fixture
def top_k():
    return 3


def test_active_learning_loop(
    dataset, surrogate, acquisition, sampler, selector, oracles, budget, top_k
):
    best, cost, num_iter = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracles=oracles,
        budget=budget,
        top_k=top_k,
    )
    assert isinstance(best, list)
    assert isinstance(cost, float)
    assert isinstance(num_iter, int)

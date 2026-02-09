import pytest

from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.sampler.pool_score_sampler import PoolScoreSampler
from activelearning.surrogate.dummy_surrogate import DummySurrogate
from activelearning.utils.types import Candidate, Observation


@pytest.fixture(params=[10, 20])
def pool_size(request):
    return request.param


@pytest.fixture
def candidate_pool(pool_size):
    return [Candidate(x=i) for i in range(pool_size)]


@pytest.fixture
def acquisition_with_surrogate():
    surrogate = DummySurrogate()
    observations = [Observation(x=i, y=float(i * 10)) for i in range(10)]
    surrogate.fit(observations)

    acquisition = DummyAcquisition(beta=1.0)
    acquisition.update(surrogate)
    return acquisition


@pytest.fixture(params=[5, 10])
def num_samples(request):
    return request.param


def test_sampling_with_acquisition_scores(
    candidate_pool, acquisition_with_surrogate, num_samples
):
    """Test that sampling uses acquisition scores as weights."""
    if num_samples > len(candidate_pool):
        pytest.skip("num_samples exceeds pool size for this test")

    sampler = PoolScoreSampler(candidate_pool=candidate_pool, num_samples=num_samples)
    samples = sampler.sample(acquisition=acquisition_with_surrogate)

    assert len(samples) == num_samples
    assert all(isinstance(s, Candidate) for s in samples)
    assert all(s in candidate_pool for s in samples)


def test_num_samples_exceeds_pool_size(acquisition_with_surrogate):
    """Test that requesting more samples than pool size returns entire pool."""
    pool = [Candidate(x=i) for i in range(5)]
    sampler = PoolScoreSampler(candidate_pool=pool, num_samples=100)
    samples = sampler.sample(acquisition=acquisition_with_surrogate)

    assert len(samples) == len(pool)
    assert set(samples) == set(pool)


def test_sampling_with_multi_fidelity_pool(acquisition_with_surrogate, num_samples):
    """Test sampling from a multi-fidelity candidate pool."""
    pool = [Candidate(x=i, fidelity=0) for i in range(10)] + [
        Candidate(x=i, fidelity=1) for i in range(10)
    ]
    sampler = PoolScoreSampler(candidate_pool=pool, num_samples=num_samples)
    samples = sampler.sample(acquisition=acquisition_with_surrogate)

    assert len(samples) == num_samples
    assert all(s in pool for s in samples)

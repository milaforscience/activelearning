import pytest

from activelearning.sampler.pool_uniform_sampler import PoolUniformSampler
from activelearning.utils.types import Candidate


@pytest.fixture(params=[10, 20])
def pool_size(request):
    return request.param


@pytest.fixture
def candidate_pool(pool_size):
    return [Candidate(x=i) for i in range(pool_size)]


@pytest.fixture(params=[5, 10])
def num_samples(request):
    return request.param


def test_uniform_sampling(candidate_pool, num_samples):
    """Test that sampling returns correct number of samples."""
    if num_samples > len(candidate_pool):
        pytest.skip("num_samples exceeds pool size for this test")

    sampler = PoolUniformSampler(candidate_pool=candidate_pool, num_samples=num_samples)
    samples = sampler.sample()

    assert len(samples) == num_samples
    assert all(isinstance(s, Candidate) for s in samples)
    assert all(s in candidate_pool for s in samples)


def test_num_samples_exceeds_pool_size():
    """Test that requesting more samples than pool size returns entire pool."""
    pool = [Candidate(x=i) for i in range(5)]
    sampler = PoolUniformSampler(candidate_pool=pool, num_samples=100)
    samples = sampler.sample()

    assert len(samples) == len(pool)
    assert set(samples) == set(pool)


def test_samples_are_unique(candidate_pool, num_samples):
    """Test that uniform sampling without replacement produces unique samples."""
    if num_samples > len(candidate_pool):
        pytest.skip("num_samples exceeds pool size for this test")

    sampler = PoolUniformSampler(candidate_pool=candidate_pool, num_samples=num_samples)
    samples = sampler.sample()

    # random.sample samples without replacement, so all should be unique
    assert len(samples) == len(set(samples))

import pytest

from activelearning.dataset.dummy_dataset import DummyDataset
from activelearning.utils.types import Candidate, Observation


@pytest.fixture
def dataset():
    return DummyDataset()


@pytest.fixture(params=[None, 0, 1])
def fidelity(request):
    return request.param


@pytest.fixture
def sample_x_values():
    return [1, 2, 3]


@pytest.fixture
def sample_scores():
    return [10.0, 20.0, 30.0]


def test_empty_dataset(dataset):
    """Test that a newly created dataset has no observations."""
    observations = dataset.get_observations()
    assert len(observations) == 0
    assert isinstance(observations, list)


def test_add_samples(dataset, sample_x_values, sample_scores, fidelity):
    """Test adding samples with different fidelity levels."""
    samples = [Candidate(x=x, fidelity=fidelity) for x in sample_x_values]

    dataset.add_samples(samples, sample_scores)
    observations = dataset.get_observations()

    assert len(observations) == len(sample_x_values)
    for i, (x, score) in enumerate(zip(sample_x_values, sample_scores)):
        assert observations[i] == Observation(x=x, y=score, fidelity=fidelity)


def test_add_samples_multiple_times(dataset):
    """Test that adding samples multiple times accumulates observations."""
    dataset.add_samples([Candidate(x=1)], [100.0])
    dataset.add_samples([Candidate(x=2)], [200.0])
    dataset.add_samples([Candidate(x=3)], [300.0])

    observations = dataset.get_observations()
    assert len(observations) == 3
    assert observations[0].x == 1
    assert observations[1].x == 2
    assert observations[2].x == 3

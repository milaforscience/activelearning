import pytest

from activelearning.dataset.dummy_dataset import DummyDataset
from activelearning.utils.types import Observation


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
def sample_y_values():
    return [10.0, 20.0, 30.0]


def test_empty_dataset(dataset):
    """Test that a newly created dataset has no observations."""
    observations = dataset.get_observations_iterable()
    assert len(observations) == 0
    assert isinstance(observations, list)


def test_add_observations(dataset, sample_x_values, sample_y_values, fidelity):
    """Test adding observations with different fidelity levels."""
    observations_to_add = [
        Observation(x=x, y=y, fidelity=fidelity)
        for x, y in zip(sample_x_values, sample_y_values)
    ]

    dataset.add_observations(observations_to_add)
    observations = dataset.get_observations_iterable()

    assert len(observations) == len(sample_x_values)
    for i, (x, y) in enumerate(zip(sample_x_values, sample_y_values)):
        assert observations[i] == Observation(x=x, y=y, fidelity=fidelity)


def test_add_observations_multiple_times(dataset):
    """Test that adding observations multiple times accumulates them."""
    dataset.add_observations([Observation(x=1, y=10.0)])
    dataset.add_observations([Observation(x=2, y=20.0)])
    dataset.add_observations([Observation(x=3, y=30.0)])

    observations = dataset.get_observations_iterable()
    assert len(observations) == 3
    assert observations[0].x == 1
    assert observations[0].y == 10.0
    assert observations[1].x == 2
    assert observations[1].y == 20.0
    assert observations[2].x == 3
    assert observations[2].y == 30.0

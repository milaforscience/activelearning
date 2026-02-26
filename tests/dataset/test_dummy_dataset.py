import pytest

from activelearning.dataset.list_dataset import ListDataset
from activelearning.utils.types import Observation


@pytest.fixture
def dataset():
    return ListDataset()


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


def test_get_latest_observations_empty_dataset(dataset):
    """Test that get_latest_observations_iterable returns empty list for new dataset."""
    latest = dataset.get_latest_observations_iterable()
    assert len(latest) == 0
    assert isinstance(latest, list)


def test_get_latest_observations_single_add(dataset, sample_x_values, sample_y_values):
    """Test that get_latest_observations_iterable returns all observations after single add."""
    observations_to_add = [
        Observation(x=x, y=y, fidelity=None)
        for x, y in zip(sample_x_values, sample_y_values)
    ]

    dataset.add_observations(observations_to_add)
    latest = dataset.get_latest_observations_iterable()

    assert len(latest) == len(sample_x_values)
    for i, (x, y) in enumerate(zip(sample_x_values, sample_y_values)):
        assert latest[i] == Observation(x=x, y=y, fidelity=None)


def test_get_latest_observations_multiple_adds(dataset):
    """Test that get_latest_observations_iterable only returns most recent batch."""
    # First batch
    dataset.add_observations([Observation(x=1, y=10.0), Observation(x=2, y=20.0)])
    latest = dataset.get_latest_observations_iterable()
    assert len(latest) == 2
    assert latest[0].x == 1
    assert latest[1].x == 2

    # Second batch - should replace latest
    dataset.add_observations([Observation(x=3, y=30.0)])
    latest = dataset.get_latest_observations_iterable()
    assert len(latest) == 1
    assert latest[0].x == 3

    # Third batch - should again replace latest
    dataset.add_observations(
        [Observation(x=4, y=40.0), Observation(x=5, y=50.0), Observation(x=6, y=60.0)]
    )
    latest = dataset.get_latest_observations_iterable()
    assert len(latest) == 3
    assert latest[0].x == 4
    assert latest[1].x == 5
    assert latest[2].x == 6

    # But all observations should still be in the full dataset
    all_obs = dataset.get_observations_iterable()
    assert len(all_obs) == 6


def test_get_latest_observations_iterable_freshness(dataset):
    """Test that get_latest_observations_iterable can be consumed multiple times."""
    dataset.add_observations([Observation(x=1, y=10.0), Observation(x=2, y=20.0)])

    # Get the iterable multiple times and consume each
    latest1 = list(dataset.get_latest_observations_iterable())
    latest2 = list(dataset.get_latest_observations_iterable())

    assert latest1 == latest2
    assert len(latest1) == 2


def test_get_latest_observations_independence(dataset):
    """Test that modifying returned list doesn't affect internal state."""
    dataset.add_observations([Observation(x=1, y=10.0)])

    latest = dataset.get_latest_observations_iterable()
    latest.append(Observation(x=999, y=999.0))

    # Get latest again - should not include the appended observation
    latest_again = dataset.get_latest_observations_iterable()
    assert len(latest_again) == 1
    assert latest_again[0].x == 1

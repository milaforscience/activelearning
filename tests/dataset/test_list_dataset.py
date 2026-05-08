import math

import pytest

from activelearning.dataset.config import CSVInitialDataConfig, ListDatasetConfig
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


def test_add_observations_filters_nan_targets(dataset):
    """NaN-valued observations are dropped at insertion time."""
    dataset.add_observations(
        [
            Observation(x=1, y=10.0),
            Observation(x=2, y=float("nan")),
            Observation(x=3, y=30.0),
        ]
    )

    observations = dataset.get_observations_iterable()
    latest = dataset.get_latest_observations_iterable()

    assert len(observations) == 2
    assert [obs.x for obs in observations] == [1, 3]
    assert latest == observations


def test_add_observations_filters_infinite_targets(dataset):
    """Infinite scalar targets are dropped for the same reason as NaNs."""
    dataset.add_observations(
        [
            Observation(x=1, y=float("inf")),
            Observation(x=2, y=float("-inf")),
        ]
    )

    assert dataset.get_observations_iterable() == []
    assert dataset.get_latest_observations_iterable() == []


def test_add_observations_preserves_non_scalar_targets(dataset):
    """Structured targets are kept because the dataset only filters scalar failures."""
    dataset.add_observations(
        [
            Observation(x=1, y=[float("nan"), 1.0]),
            Observation(x=2, y={"score": float("inf")}),
        ]
    )

    observations = dataset.get_observations_iterable()

    assert len(observations) == 2
    assert len(observations[0].y) == 2
    assert math.isnan(observations[0].y[0])
    assert observations[0].y[1] == 1.0
    assert observations[1].y == {"score": float("inf")}


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


def test_list_dataset_config_loads_numeric_csv_initial_data(tmp_path):
    """ListDatasetConfig should preload numeric observations from CSV."""

    csv_path = tmp_path / "initial.csv"
    csv_path.write_text(
        "x1,x2,y,fidelity,source\n1.0,2.0,3.0,1,seed\n4.0,5.0,6.0,2,seed\n",
        encoding="utf-8",
    )
    config = ListDatasetConfig(
        initial_data=CSVInitialDataConfig(
            path=csv_path,
            x_columns=["x1", "x2"],
            y_column="y",
            fidelity_column="fidelity",
        )
    )

    observations = config.build().get_observations_iterable()

    assert observations == [
        Observation(x=[1.0, 2.0], y=3.0, fidelity=1, metadata={"source": "seed"}),
        Observation(x=[4.0, 5.0], y=6.0, fidelity=2, metadata={"source": "seed"}),
    ]


def test_list_dataset_config_loads_string_csv_for_minimization(tmp_path):
    """ListDatasetConfig can negate seeded targets before storing them."""

    csv_path = tmp_path / "initial.csv"
    csv_path.write_text("selfies,y\n[C],7.5\n", encoding="utf-8")
    config = ListDatasetConfig(
        negate_initial_targets=True,
        initial_data=CSVInitialDataConfig(
            path=csv_path,
            x_columns="selfies",
            y_column="y",
            metadata_columns=None,
        ),
    )

    observations = config.build().get_observations_iterable()

    assert observations == [Observation(x="[C]", y=-7.5)]

import pytest

from activelearning.surrogate.dummy_surrogate import DummySurrogate
from activelearning.utils.types import Candidate, Observation


@pytest.fixture
def surrogate():
    return DummySurrogate()


@pytest.fixture
def observations():
    return [
        Observation(x=1, y=10.0),
        Observation(x=2, y=20.0),
        Observation(x=3, y=30.0),
    ]


@pytest.fixture(params=[None, 0, 1])
def fidelity(request):
    return request.param


@pytest.fixture
def observation_x_values():
    return [5, 10]


@pytest.fixture
def observation_y_values():
    return [50.0, 100.0]


def test_fit_stores_observations(surrogate, observations):
    """Test that fit stores observations in the internal model."""
    surrogate.fit(observations)

    # Check internal state
    assert len(surrogate._model) == len(observations)
    for obs in observations:
        assert surrogate._model[(obs.x, obs.fidelity)] == obs.y


def test_predict_known_candidates(
    surrogate, observation_x_values, observation_y_values
):
    """Test predictions for candidates that were in the training data."""
    observations = [
        Observation(x=x, y=y)
        for x, y in zip(observation_x_values, observation_y_values)
    ]
    surrogate.fit(observations)

    candidates = [Candidate(x=x) for x in observation_x_values]
    predictions = surrogate.predict(candidates)

    assert "mean" in predictions
    assert "std" in predictions
    assert predictions["mean"] == observation_y_values
    assert predictions["std"] == [0.1] * len(observation_x_values)


def test_predict_unknown_candidates(surrogate, observations):
    """Test predictions for candidates not in the training data."""
    surrogate.fit(observations)

    candidates = [Candidate(x=999)]
    predictions = surrogate.predict(candidates)

    expected_mean = sum(obs.y for obs in observations) / len(observations)
    assert predictions["mean"] == [expected_mean]
    assert predictions["std"] == [1.0]


def test_predict_mixed_candidates(surrogate):
    """Test predictions for a mix of known and unknown candidates."""
    observations = [Observation(x=1, y=100.0)]
    surrogate.fit(observations)

    candidates = [Candidate(x=1), Candidate(x=999)]
    predictions = surrogate.predict(candidates)

    assert predictions["mean"] == [100.0, 100.0]
    assert predictions["std"] == [0.1, 1.0]


def test_fit_with_multi_fidelity_observations(surrogate):
    """Test that fit handles multi-fidelity observations correctly."""
    observations = [
        Observation(x=1, y=10.0, fidelity=0),
        Observation(x=1, y=12.0, fidelity=1),
        Observation(x=2, y=20.0, fidelity=0),
    ]
    surrogate.fit(observations)

    # Different fidelities are stored separately
    assert surrogate._model[(1, 0)] == 10.0
    assert surrogate._model[(1, 1)] == 12.0
    assert surrogate._model[(2, 0)] == 20.0

    # Mean score is average of all observations
    expected_mean = (10.0 + 12.0 + 20.0) / 3
    assert surrogate._mean_score == pytest.approx(expected_mean)


def test_predict_with_fidelity(surrogate, fidelity):
    """Test predictions consider fidelity levels."""
    observations = [
        Observation(x=5, y=50.0, fidelity=0),
        Observation(x=5, y=55.0, fidelity=1),
    ]
    surrogate.fit(observations)

    candidate = Candidate(x=5, fidelity=fidelity)
    predictions = surrogate.predict([candidate])

    if fidelity == 0:
        assert predictions["mean"] == [50.0]
        assert predictions["std"] == [0.1]
    elif fidelity == 1:
        assert predictions["mean"] == [55.0]
        assert predictions["std"] == [0.1]
    else:  # None - unknown fidelity
        expected_mean = (50.0 + 55.0) / 2
        assert predictions["mean"] == [expected_mean]
        assert predictions["std"] == [1.0]

import pytest

from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.utils.types import Candidate, Observation


@pytest.fixture(params=[1.0, 2.5, 3.0])
def beta(request):
    return request.param


@pytest.fixture
def surrogate():
    return DummyMeanSurrogate()


@pytest.fixture
def observations():
    return [
        Observation(x=1, y=10.0),
        Observation(x=2, y=20.0),
    ]


@pytest.fixture
def test_candidates():
    return [Candidate(x=1), Candidate(x=2), Candidate(x=3)]


def test_initialization(beta):
    """Test initialization with different beta values."""
    acq = DummyAcquisition(beta=beta)
    assert acq._beta == beta


def test_scoring_without_surrogate(test_candidates):
    """Test that scoring without a surrogate returns zeros."""
    acquisition = DummyAcquisition()
    acq_values = acquisition.score(test_candidates)

    assert len(acq_values) == len(test_candidates)
    assert all(acq_value == 0.0 for acq_value in acq_values)


def test_scoring_with_surrogate(beta, surrogate, observations):
    """Test scoring with mean and std predictions."""
    acquisition = DummyAcquisition(beta=beta)
    surrogate.fit(observations)
    acquisition.update(surrogate)

    # Query known candidates
    candidates = [Candidate(x=1), Candidate(x=2)]
    acq_values = acquisition.score(candidates)

    assert len(acq_values) == 2
    # Known candidates: mean + beta * low_std (0.1)
    assert acq_values[0] == pytest.approx(10.0 + beta * 0.1)
    assert acq_values[1] == pytest.approx(20.0 + beta * 0.1)


def test_scoring_unknown_candidates(beta, surrogate, observations):
    """Test scoring for unknown candidates with higher uncertainty."""
    acquisition = DummyAcquisition(beta=beta)
    surrogate.fit(observations)
    acquisition.update(surrogate)

    # Query unknown candidate (higher std)
    candidates = [Candidate(x=999)]
    acq_values = acquisition.score(candidates)

    # Unknown candidates get mean_score + beta * 1.0
    expected_mean = 15.0  # (10 + 20) / 2
    assert len(acq_values) == 1
    assert acq_values[0] == pytest.approx(expected_mean + beta * 1.0)


def test_surrogate_update(surrogate):
    """Test that update method correctly sets the surrogate."""
    acq = DummyAcquisition()
    assert acq.surrogate is None

    acq.update(surrogate)
    assert acq.surrogate is surrogate


def test_supports_singleton_scoring():
    """Test that DummyAcquisition declares singleton scoring support."""
    acq = DummyAcquisition()
    assert acq.supports_singleton_scoring() is True

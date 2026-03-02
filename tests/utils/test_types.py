import pytest

from activelearning.utils.types import Candidate, Observation, label_candidates


@pytest.fixture(params=[None, 1, 2])
def fidelity(request):
    return request.param


@pytest.fixture
def candidate_x():
    return 42


@pytest.fixture
def observation_x():
    return 10


@pytest.fixture
def observation_y():
    return 5.5


def test_candidate_creation(candidate_x, fidelity):
    """Test creating a Candidate with different fidelity levels."""
    candidate = Candidate(x=candidate_x, fidelity=fidelity)
    assert candidate.x == candidate_x
    assert candidate.fidelity == fidelity


def test_observation_creation(observation_x, observation_y, fidelity):
    """Test creating an Observation with different fidelity levels."""
    observation = Observation(x=observation_x, y=observation_y, fidelity=fidelity)
    assert observation.x == observation_x
    assert observation.y == observation_y
    assert observation.fidelity == fidelity


def test_candidate_immutability(candidate_x, fidelity):
    """Test that Candidate instances are frozen and cannot be modified."""
    candidate = Candidate(x=candidate_x, fidelity=fidelity)
    with pytest.raises(AttributeError):
        candidate.x = 99  # type: ignore[misc]


def test_observation_immutability(observation_x, observation_y, fidelity):
    """Test that Observation instances are frozen and cannot be modified."""
    observation = Observation(x=observation_x, y=observation_y, fidelity=fidelity)
    with pytest.raises(AttributeError):
        observation.y = 100.0  # type: ignore[misc]


def test_label_candidates(fidelity):
    """Test converting candidates and labels to observations."""
    candidates = [
        Candidate(x=1, fidelity=fidelity),
        Candidate(x=2, fidelity=fidelity),
        Candidate(x=3, fidelity=fidelity),
    ]
    labels = [10.0, 20.0, 30.0]

    observations = label_candidates(candidates, labels)

    assert len(observations) == 3
    assert observations[0] == Observation(x=1, y=10.0, fidelity=fidelity)
    assert observations[1] == Observation(x=2, y=20.0, fidelity=fidelity)
    assert observations[2] == Observation(x=3, y=30.0, fidelity=fidelity)

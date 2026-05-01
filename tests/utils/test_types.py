import pytest

from activelearning.utils.types import (
    Candidate,
    Observation,
    candidates_to_tensor,
    label_candidates,
    observations_to_tensors,
)


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


def test_candidate_creation_with_metadata():
    """Metadata should be stored unchanged on Candidate."""
    metadata = {"raw": "[C][O]"}
    candidate = Candidate(x=42, metadata=metadata)
    assert candidate.metadata == metadata


def test_observation_creation_with_metadata():
    """Metadata should be stored unchanged on Observation."""
    metadata = {"raw": "[C][O]"}
    observation = Observation(x=10, y=5.5, metadata=metadata)
    assert observation.metadata == metadata


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


def test_label_candidates_preserves_metadata():
    """Candidate metadata should be copied to the resulting observations."""
    metadata = {"raw": "[C][O]"}
    observations = label_candidates([Candidate(x=1, metadata=metadata)], [10.0])
    assert observations == [Observation(x=1, y=10.0, metadata=metadata)]


def test_observations_to_tensors_empty_mapping_raises_key_error():
    """Test that an explicit empty fidelity map does not mask missing entries."""
    observations = [Observation(x=1, y=10.0, fidelity=0)]

    with pytest.raises(KeyError, match="0"):
        observations_to_tensors(observations, fidelity_confidences={})


def test_observations_to_tensors_missing_mapping_raises_value_error():
    """Test that fidelity-bearing observations require an explicit mapping."""
    observations = [Observation(x=1, y=10.0, fidelity=0)]

    with pytest.raises(ValueError, match="no fidelity_confidences mapping"):
        observations_to_tensors(observations)


def test_candidates_to_tensor_empty_mapping_raises_key_error():
    """Test that candidate tensor conversion honors explicit empty fidelity maps."""
    candidates = [Candidate(x=1, fidelity=0)]

    with pytest.raises(KeyError, match="0"):
        candidates_to_tensor(candidates, fidelity_confidences={})


def test_candidates_to_tensor_missing_mapping_raises_value_error():
    """Test that fidelity-bearing candidates require an explicit mapping."""
    candidates = [Candidate(x=1, fidelity=0)]

    with pytest.raises(ValueError, match="no fidelity_confidences mapping"):
        candidates_to_tensor(candidates)


def test_observations_to_tensors_ignores_metadata():
    """Metadata should not affect tensor conversion."""
    X, y, fidelities = observations_to_tensors(
        [Observation(x=1, y=10.0, metadata={"raw": "[C][O]"})]
    )
    assert X.tolist() == [1.0]
    assert y.tolist() == [10.0]
    assert fidelities == []


def test_candidates_to_tensor_ignores_metadata():
    """Metadata should not affect candidate tensor conversion."""
    X, fidelities = candidates_to_tensor([Candidate(x=1, metadata={"raw": "[C][O]"})])
    assert X.tolist() == [1.0]
    assert fidelities == []

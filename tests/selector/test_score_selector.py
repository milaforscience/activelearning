import pytest

from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.utils.types import Candidate, Observation


@pytest.fixture(params=[2, 3, 5])
def num_samples(request):
    return request.param


@pytest.fixture
def selector(num_samples):
    return TopKAcquisitionSelector(num_samples=num_samples)


@pytest.fixture
def acquisition_with_surrogate():
    surrogate = DummyMeanSurrogate()
    observations = [
        Observation(x=1, y=10.0),
        Observation(x=5, y=50.0),
        Observation(x=10, y=100.0),
    ]
    surrogate.fit(observations)

    acquisition = DummyAcquisition(beta=1.0)
    acquisition.update(surrogate)
    return acquisition


@pytest.fixture
def test_candidates():
    return [Candidate(x=i) for i in [1, 5, 10, 2, 8]]


def test_top_k_selection_by_score(
    selector, acquisition_with_surrogate, test_candidates
):
    """Test that selector returns top-k candidates by acquisition score."""
    selected = selector(test_candidates, acquisition=acquisition_with_surrogate)

    expected_length = min(selector.num_samples, len(test_candidates))
    assert len(selected) == expected_length
    assert all(isinstance(c, Candidate) for c in selected)


def test_correct_ordering_highest_first(acquisition_with_surrogate):
    """Test that selected candidates are ordered by score (highest first)."""
    selector = TopKAcquisitionSelector(num_samples=3)
    candidates = [Candidate(x=1), Candidate(x=5), Candidate(x=10)]
    selected = selector(candidates, acquisition=acquisition_with_surrogate)

    # Get acquisition values to verify ordering
    acq_values = acquisition_with_surrogate(selected)

    assert len(selected) == 3
    # Acquisition values should be in descending order
    assert acq_values[0] >= acq_values[1] >= acq_values[2]


def test_num_samples_exceeds_candidates_length(acquisition_with_surrogate):
    """Test that requesting more samples than candidates returns all."""
    selector = TopKAcquisitionSelector(num_samples=10)
    candidates = [Candidate(x=i) for i in range(5)]
    selected = selector(candidates, acquisition=acquisition_with_surrogate)

    assert len(selected) == 5


def test_selection_with_varied_scores():
    """Test selection with clearly differentiated scores."""
    surrogate = DummyMeanSurrogate()
    observations = [
        Observation(x=1, y=100.0),
        Observation(x=2, y=50.0),
        Observation(x=3, y=200.0),
        Observation(x=4, y=25.0),
    ]
    surrogate.fit(observations)

    acquisition = DummyAcquisition(beta=0.0)  # Only use mean
    acquisition.update(surrogate)

    selector = TopKAcquisitionSelector(num_samples=2)
    candidates = [Candidate(x=1), Candidate(x=2), Candidate(x=3), Candidate(x=4)]
    selected = selector(candidates, acquisition=acquisition)

    # Top 2 should be x=3 (200.0) and x=1 (100.0)
    assert len(selected) == 2
    assert selected[0].x == 3
    assert selected[1].x == 1

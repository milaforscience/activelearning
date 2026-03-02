from unittest.mock import Mock

import pytest

from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.budget.budget import Budget
from activelearning.dataset.list_dataset import ListDataset
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.sampler.pool_score_sampler import PoolScoreSampler
from activelearning.selector.score_selector import ScoreSelector
from activelearning.surrogate.dummy_surrogate import DummySurrogate
from activelearning.active_learning import active_learning
from activelearning.utils.types import Candidate


class ConfidenceAwareDummySurrogate(DummySurrogate):
    """Dummy surrogate variant that records fidelity confidences."""

    def __init__(self) -> None:
        super().__init__()
        self.fidelity_confidences: dict[int, float] | None = None

    def set_fidelity_confidences(self, confidences: dict[int, float]) -> None:
        self.fidelity_confidences = dict(confidences)


@pytest.fixture
def dataset():
    """Create a dummy dataset for testing."""
    return ListDataset()


@pytest.fixture
def surrogate():
    """Create a dummy surrogate model for testing."""
    return DummySurrogate()


@pytest.fixture
def acquisition():
    """Create a dummy acquisition function for testing."""
    return DummyAcquisition()


@pytest.fixture
def sampler():
    """Create a pool score sampler with multi-fidelity candidates."""
    candidate_pool = [Candidate(i, 0) for i in range(100)] + [
        Candidate(i, 1) for i in range(100)
    ]
    return PoolScoreSampler(candidate_pool=candidate_pool, num_samples=100)


@pytest.fixture
def selector():
    """Create a score-based selector for testing."""
    return ScoreSelector(num_samples=5)


@pytest.fixture
def oracle():
    """Create multi-fidelity oracle with different costs and scoring functions."""

    def score_fn_0(s):
        return float(s)

    def score_fn_1(s):
        return float(s) + 0.5

    return MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": score_fn_0,
                "fidelity_confidence": 1.0,
            },
            1: {
                "cost_per_sample": 2.0,
                "score_fn": score_fn_1,
                "fidelity_confidence": 1.0,
            },
        }
    )


@pytest.fixture
def budget():
    """Set the budget for active learning loop."""
    return Budget(available_budget=100.0, schedule=lambda r: 20.0)


@pytest.fixture
def top_k():
    """Set the number of top candidates to retrieve."""
    return 3


def test_active_learning_loop(
    dataset, surrogate, acquisition, sampler, selector, oracle, budget, top_k
):
    """Test that the active learning loop completes and returns expected types."""
    dataset_out, cost, num_iter = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=oracle,
        budget=budget,
    )
    best = dataset_out.get_best_candidates(k=top_k)
    assert isinstance(best, list)
    assert isinstance(cost, float)
    assert isinstance(num_iter, int)


def test_active_learning_passes_fidelity_confidences_to_surrogate(
    dataset, acquisition, sampler, selector, oracle, budget
):
    """Test the active learning loop passes oracle confidences to surrogate."""
    surrogate = ConfidenceAwareDummySurrogate()
    active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=oracle,
        budget=budget,
    )
    assert surrogate.fidelity_confidences == oracle.get_fidelity_confidences()


def test_active_learning_stops_when_selector_returns_empty(
    dataset, surrogate, acquisition, sampler, oracle, budget
):
    """Test loop terminates when selector returns no candidates."""
    empty_selector = Mock()
    empty_selector.return_value = []

    dataset_out, cost, num_iter = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=empty_selector,
        oracle=oracle,
        budget=budget,
    )

    assert dataset_out.get_observations_iterable() == []
    assert cost == 0.0
    assert num_iter == 0

import pytest
from unittest.mock import Mock
from typing import Callable, Iterable, Optional, Sequence

from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.budget.budget import Budget
from activelearning.dataset.list_dataset import ListDataset
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.sampler.pool_score_sampler import PoolScoreSampler
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.active_learning import active_learning
from activelearning.utils.types import Candidate, Observation
from activelearning.logger.logger import ConsoleLogger
from activelearning.runtime import RuntimeContext


class ConfidenceAwareDummyMeanSurrogate(DummyMeanSurrogate):
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
    """Create a BoTorch surrogate for testing the full AL loop end-to-end."""
    return BoTorchGPSurrogate()


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
    return TopKAcquisitionSelector(num_samples=5)


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


def test_active_learning_logs_metrics_with_console_logger(
    dataset, surrogate, acquisition, sampler, selector, oracle, budget, capsys
):
    """Test that the active learning loop integrates with a logger."""
    logger = ConsoleLogger(project_name="test_project", run_name="console_test_run")
    capsys.readouterr()

    _, cost, num_iter = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=oracle,
        budget=budget,
        runtime_context=RuntimeContext(logger=logger),
    )
    out = capsys.readouterr().out

    assert num_iter > 0
    assert cost > 0.0
    assert out.count("[Step ") == num_iter
    assert "round=" in out
    assert "num_new_samples=" in out
    assert "round_cost=" in out
    assert "total_cost=" in out
    assert "budget_remaining=" in out
    assert "[Logger] Run 'console_test_run' finished." in out


def test_active_learning_passes_fidelity_confidences_to_surrogate(
    dataset, acquisition, sampler, selector, oracle, budget
):
    """Test the active learning loop passes oracle confidences to surrogate."""
    surrogate = ConfidenceAwareDummyMeanSurrogate()
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


def test_active_learning_cold_start_with_botorch_surrogate(
    acquisition, sampler, selector, oracle, budget
):
    """End-to-end test: BoTorchGPSurrogate starts from an empty dataset.

    The first round must use random selection (surrogate unfitted); subsequent
    rounds must use acquisition-driven selection (surrogate fitted on round-1 data).
    """
    dataset = ListDataset()
    surrogate = BoTorchGPSurrogate()

    assert not surrogate.is_fitted(), "Surrogate must be unfitted before any data."

    dataset_out, cost, num_iter = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=oracle,
        budget=budget,
    )

    assert surrogate.is_fitted(), "Surrogate must be fitted after at least one round."
    assert num_iter >= 1, "Loop must complete at least one round."
    assert cost > 0.0, "At least one oracle query must have been made."
    observations = list(dataset_out.get_observations_iterable())
    assert len(observations) > 0, "Dataset must contain observations after the loop."


def test_botorch_surrogate_fit_empty_is_noop():
    """fit([]) must not raise and must leave the surrogate in an unfitted state."""
    surrogate = BoTorchGPSurrogate()
    surrogate.fit([])  # Must not raise
    assert not surrogate.is_fitted()
    assert surrogate.model is None


# ---------------------------------------------------------------------------
# Test runtime integration with active learning components
# ---------------------------------------------------------------------------


class RuntimeLoggingDataset(ListDataset):
    """Dataset test double that emits metrics through the bound runtime logger."""

    def add_observations(self, observations: Sequence[Observation]) -> None:
        super().add_observations(observations)
        if self.logger is not None:
            self.logger.log_metric("dataset_records", len(self._records))


class RuntimeLoggingSurrogate(DummyMeanSurrogate):
    """Surrogate test double that emits metrics through the bound runtime logger."""

    def fit(self, observations: Iterable[Observation]) -> None:
        super().fit(observations)
        if self.logger is not None:
            self.logger.log_metric("surrogate_fit_calls", 1)


class RuntimeLoggingSampler(PoolScoreSampler):
    """Sampler test double that emits metrics through the bound runtime logger."""

    def sample(
        self,
        acquisition: Optional[DummyAcquisition] = None,
        observations: Optional[Iterable[Observation]] = None,
    ) -> list[Candidate]:
        samples = super().sample(acquisition=acquisition, observations=observations)
        if self.logger is not None:
            self.logger.log_metric("sampler_num_samples", len(samples))
        return samples


class RuntimeLoggingSelector(TopKAcquisitionSelector):
    """Selector test double that emits metrics through the bound runtime logger."""

    def __call__(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[DummyAcquisition] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        round_budget: Optional[float] = None,
    ) -> list[Candidate]:
        selected = super().__call__(
            candidates,
            acquisition=acquisition,
            cost_fn=cost_fn,
            round_budget=round_budget,
        )
        if self.logger is not None:
            self.logger.log_metric("selector_selected", len(selected))
        return selected


class RuntimeLoggingOracle(MultiFidelityOracle):
    """Oracle test double that emits metrics through the bound runtime logger."""

    def query(self, candidates: Sequence[Candidate]) -> list[Observation]:
        observations = super().query(candidates)
        if self.logger is not None:
            self.logger.log_metric("oracle_queries", len(observations))
        return observations


class RuntimeLoggingBudget(Budget):
    """Budget test double that emits metrics through the bound runtime logger."""

    def get_round_budget(self, current_round: int) -> float:
        round_budget = super().get_round_budget(current_round)
        if self.logger is not None:
            self.logger.log_metric("budget_round_limit", round_budget)
        return round_budget


def test_active_learning_binds_runtime_context_to_modules_for_logging(capsys):
    """Bound modules should log through the runtime context logger in the loop."""
    logger = ConsoleLogger(project_name="test_project", run_name="runtime_test_run")
    runtime_context = RuntimeContext(logger=logger)
    capsys.readouterr()

    dataset = RuntimeLoggingDataset()
    surrogate = RuntimeLoggingSurrogate()
    acquisition = DummyAcquisition()
    candidate_pool = [Candidate(i, 0) for i in range(50)] + [
        Candidate(i, 1) for i in range(50)
    ]
    sampler = RuntimeLoggingSampler(candidate_pool=candidate_pool, num_samples=20)
    selector = RuntimeLoggingSelector(num_samples=5)

    def score_fn_0(value):
        return float(value)

    def score_fn_1(value):
        return float(value) + 0.5

    oracle = RuntimeLoggingOracle(
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
    budget = RuntimeLoggingBudget(
        available_budget=100.0, schedule=lambda round_num: 20.0
    )

    _, cost, num_iter = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=oracle,
        budget=budget,
        runtime_context=runtime_context,
    )
    out = capsys.readouterr().out

    assert num_iter > 0
    assert cost > 0.0
    assert dataset.runtime_context is runtime_context
    assert surrogate.runtime_context is runtime_context
    assert acquisition.runtime_context is runtime_context
    assert sampler.runtime_context is runtime_context
    assert selector.runtime_context is runtime_context
    assert oracle.runtime_context is runtime_context
    assert budget.runtime_context is runtime_context
    assert "surrogate_fit_calls=" in out
    assert "sampler_num_samples=" in out
    assert "selector_selected=" in out
    assert "oracle_queries=" in out
    assert "dataset_records=" in out
    assert "budget_round_limit=" in out
    assert "[Logger] Run 'runtime_test_run' finished." in out

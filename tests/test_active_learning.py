import pytest
from unittest.mock import Mock
from typing import Callable, Iterable, Optional, Sequence

from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.budget.budget import Budget
from activelearning.dataset.list_dataset import ListDataset
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.run_writer import RunWriter
from activelearning.sampler.pool_score_sampler import PoolScoreSampler
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.surrogate.surrogate import MultiFidelitySurrogate
from activelearning.active_learning import active_learning
from activelearning.utils.types import Candidate, Observation
from activelearning.logger.logger import ConsoleLogger
from activelearning.runtime import RuntimeContext


class RecordingRunWriter(RunWriter):
    """In-memory run writer for asserting active-learning integration."""

    def __init__(self) -> None:
        self.start_metadata: dict | None = None
        self.rounds: list[dict] = []
        self.summary: dict | None = None

    def start_run(self, metadata: dict) -> None:
        """Store the run-start metadata."""
        self.start_metadata = metadata

    def record_round(
        self,
        *,
        round_index: int,
        sampled_candidates: Sequence[Candidate],
        sampled_scores: Sequence[float],
        selected_candidates: Sequence[Candidate],
        selected_scores: Sequence[float],
        selected_costs: Sequence[float],
        observations: Sequence[Observation],
        cumulative_cost: float,
        remaining_budget: float,
    ) -> None:
        """Store one round payload."""
        self.rounds.append(
            {
                "round_index": round_index,
                "sampled_candidates": list(sampled_candidates),
                "sampled_scores": list(sampled_scores),
                "selected_candidates": list(selected_candidates),
                "selected_scores": list(selected_scores),
                "selected_costs": list(selected_costs),
                "observations": list(observations),
                "cumulative_cost": cumulative_cost,
                "remaining_budget": remaining_budget,
            }
        )

    def end_run(self, summary: dict) -> None:
        """Store the final run summary."""
        self.summary = summary


@pytest.fixture
def dataset():
    """Create a dummy dataset for testing."""
    return ListDataset()


@pytest.fixture
def surrogate(oracle):
    """Create a BoTorch surrogate for testing the full AL loop end-to-end."""
    return BoTorchGPSurrogate(is_multi_fidelity=True)


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
    assert surrogate.get_fidelity_confidences() == oracle.get_fidelity_confidences()


def test_active_learning_rejects_unsupported_multi_fidelity_surrogate(
    dataset, acquisition, sampler, selector, oracle, budget
):
    """Multi-fidelity oracle metadata requires a compatible surrogate."""
    with pytest.raises(ValueError, match="does not support multi-fidelity"):
        active_learning(
            dataset=dataset,
            surrogate=DummyMeanSurrogate(),
            acquisition=acquisition,
            sampler=sampler,
            selector=selector,
            oracle=oracle,
            budget=budget,
        )


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


class RuntimeLoggingSurrogate(DummyMeanSurrogate, MultiFidelitySurrogate):
    """Surrogate test double that emits metrics through the bound runtime logger."""

    def set_fidelity_confidences(self, confidences: dict[int, float]) -> None:
        """Accept fidelity metadata without using it."""

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
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
    ) -> list[Candidate]:
        samples = super().sample(
            acquisition=acquisition,
            observations=observations,
            cost_fn=cost_fn,
        )
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


# ---------------------------------------------------------------------------
# Oracle invalid-output filtering at the AL boundary
# ---------------------------------------------------------------------------


def _make_nan_oracle() -> MultiFidelityOracle:
    """Return a single-fidelity oracle whose score function always returns NaN."""
    return MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": lambda _x: float("nan"),
                "fidelity_confidence": 1.0,
            }
        }
    )


def _make_mixed_oracle() -> MultiFidelityOracle:
    """Return a single-fidelity oracle that returns NaN for odd candidates."""
    return MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": lambda x: float("nan") if int(x) % 2 == 1 else float(x),
                "fidelity_confidence": 1.0,
            }
        }
    )


def _make_pool_sampler_single_fidelity(
    n_candidates: int, n_samples: int
) -> PoolScoreSampler:
    pool = [Candidate(i, fidelity=0) for i in range(n_candidates)]
    return PoolScoreSampler(candidate_pool=pool, num_samples=n_samples)


def test_al_loop_drops_nan_oracle_observations():
    """Oracle observations with NaN targets are not added to the dataset."""
    nan_oracle = _make_nan_oracle()
    dataset = ListDataset()
    surrogate = DummyMeanSurrogate()
    acquisition = DummyAcquisition()
    sampler = _make_pool_sampler_single_fidelity(50, 20)
    selector = TopKAcquisitionSelector(num_samples=5)
    budget = Budget(available_budget=20.0, schedule=lambda _: 20.0)

    dataset_out, _cost, _num_iter = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=nan_oracle,
        budget=budget,
    )

    # All oracle outputs were NaN → dataset must remain empty
    assert dataset_out.get_observations_iterable() == []


def test_al_loop_keeps_valid_from_mixed_oracle():
    """Only the valid observations from a mixed oracle batch enter the dataset."""
    mixed_oracle = _make_mixed_oracle()
    dataset = ListDataset()
    surrogate = DummyMeanSurrogate()
    acquisition = DummyAcquisition()
    # Use a pool of even-indexed candidates only → all scores will be finite
    pool = [Candidate(i * 2, fidelity=0) for i in range(50)]
    sampler = PoolScoreSampler(candidate_pool=pool, num_samples=20)
    selector = TopKAcquisitionSelector(num_samples=5)
    budget = Budget(available_budget=10.0, schedule=lambda _: 10.0)

    dataset_out, _cost, _num_iter = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=mixed_oracle,
        budget=budget,
    )

    observations = list(dataset_out.get_observations_iterable())
    # All stored observations must have finite y values
    assert all(o.y is not None for o in observations)
    import math

    assert all(math.isfinite(float(o.y)) for o in observations)


def test_al_loop_all_invalid_oracle_does_not_crash():
    """A round where all oracle outputs are invalid does not crash the loop."""
    nan_oracle = _make_nan_oracle()
    dataset = ListDataset()
    surrogate = DummyMeanSurrogate()
    acquisition = DummyAcquisition()
    sampler = _make_pool_sampler_single_fidelity(20, 10)
    selector = TopKAcquisitionSelector(num_samples=3)
    # Budget for exactly one round
    budget = Budget(available_budget=3.0, schedule=lambda _: 3.0)

    dataset_out, cost, num_iter = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=nan_oracle,
        budget=budget,
    )

    # One round ran, budget was consumed, but dataset has nothing valid
    assert num_iter == 1
    assert cost > 0.0
    assert dataset_out.get_observations_iterable() == []


def test_al_loop_warns_when_invalid_observations_dropped(caplog):
    """The AL loop emits a warning when oracle observations are dropped."""
    import logging

    nan_oracle = _make_nan_oracle()
    dataset = ListDataset()
    surrogate = DummyMeanSurrogate()
    acquisition = DummyAcquisition()
    sampler = _make_pool_sampler_single_fidelity(20, 10)
    selector = TopKAcquisitionSelector(num_samples=3)
    budget = Budget(available_budget=3.0, schedule=lambda _: 3.0)

    with caplog.at_level(logging.WARNING, logger="activelearning.active_learning"):
        active_learning(
            dataset=dataset,
            surrogate=surrogate,
            acquisition=acquisition,
            sampler=sampler,
            selector=selector,
            oracle=nan_oracle,
            budget=budget,
        )

    assert any("Dropped" in record.message for record in caplog.records)


def test_active_learning_run_writer_records_only_valid_observations():
    """Run-writer round records must match the filtered observations added."""
    mixed_oracle = _make_mixed_oracle()
    run_writer = RecordingRunWriter()

    dataset_out, cost, num_iter = active_learning(
        dataset=ListDataset(),
        surrogate=DummyMeanSurrogate(),
        acquisition=DummyAcquisition(),
        sampler=PoolScoreSampler(
            candidate_pool=[Candidate(i, fidelity=0) for i in range(5)],
            num_samples=5,
        ),
        selector=TopKAcquisitionSelector(num_samples=3),
        oracle=mixed_oracle,
        budget=Budget(available_budget=3.0, schedule=lambda _: 3.0),
        run_writer=run_writer,
    )

    stored_observations = list(dataset_out.get_observations_iterable())

    assert num_iter == 1
    assert cost == 3.0
    assert run_writer.start_metadata == {
        "initial_budget": 3.0,
        "initial_data": {"initial_observations": []},
    }
    assert run_writer.summary == {
        "num_rounds": 1,
        "total_cost": 3.0,
        "budget_remaining": 0.0,
    }
    assert len(run_writer.rounds) == 1
    assert run_writer.rounds[0]["observations"] == stored_observations
    assert [observation.x for observation in stored_observations] == [0, 2]

import pytest
from itertools import count
from unittest.mock import Mock
from typing import Callable, Iterable, Optional, Sequence

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.budget.budget import Budget
from activelearning.dataset.list_dataset import ListDataset
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.monitoring.run_writer import RoundRecord, RunWriter
from activelearning.sampler.pool_score_sampler import PoolScoreSampler
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.surrogate.surrogate import MultiFidelitySurrogate
from activelearning.active_learning import active_learning
from activelearning.active_learning import _validate_oracle_results
from activelearning.utils.types import Candidate, Observation
from activelearning.logger.logger import ConsoleLogger
from activelearning.monitoring.diagnostics_config import DiagnosticsConfig
from activelearning.runtime import RuntimeContext
import activelearning.active_learning as active_learning_module


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
        record: RoundRecord,
        figures=None,
    ) -> None:
        """Store one round payload."""
        self.rounds.append(
            {
                "round_index": record.round_index,
                "sampled_candidates": list(record.sampled_candidates),
                "selected_candidates": list(record.selected_candidates),
                "selected_costs": list(record.selected_costs),
                "valid_observations": list(record.valid_observations),
                "cumulative_cost": record.cumulative_cost,
                "remaining_budget": record.remaining_budget,
                "metrics": dict(record.metrics),
                "profiling": dict(record.profiling),
                "diagnostics": dict(record.diagnostics),
            }
        )

    def end_run(self, summary: dict) -> None:
        """Store the final run summary."""
        self.summary = summary


class DiagnosticPoolSampler(PoolScoreSampler):
    """Pool sampler exposing one implementation-specific diagnostics payload."""

    def __init__(self, candidate_pool: Sequence[Candidate], num_samples: int) -> None:
        super().__init__(candidate_pool=candidate_pool, num_samples=num_samples)
        self._figure: Figure | None = None
        self.created_figure: Figure | None = None

    def sample(
        self,
        acquisition=None,
        observations=None,
        cost_fn=None,
    ) -> Sequence[Candidate]:
        """Create one specialized plot after standard pool sampling."""
        candidates = super().sample(acquisition, observations, cost_fn)
        self._figure = plt.figure()
        self._figure.add_subplot(1, 1, 1).plot([0.0, 1.0], [0.0, 1.0])
        self.created_figure = self._figure
        return candidates

    def drain_round_diagnostics(
        self,
        *,
        include_figures: bool,
        max_points: int,
    ) -> tuple[dict[str, int | float], dict[str, Figure]]:
        """Return the pending sampler-specific metric and optional figure."""
        _ = max_points
        figure = self._figure
        self._figure = None
        return (
            {"sampler/test/generated": 1},
            {"sampler/test/trajectory": figure}
            if include_figures and figure is not None
            else {},
        )


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


def test_active_learning_stops_at_configured_max_rounds(
    dataset, surrogate, acquisition, sampler, selector, oracle
):
    """A round cap stops the loop while unused total budget remains."""
    budget = Budget(
        available_budget=100.0,
        schedule=lambda _: 20.0,
        max_rounds=1,
    )

    _, cost, num_iter = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=oracle,
        budget=budget,
    )

    assert num_iter == 1
    assert cost > 0.0
    assert budget.available_budget > 0.0


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


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        ([], "one observation"),
        ([Observation(x=2, y=2.0, fidelity=0)], "input identity"),
        ([Observation(x=1, y=1.0, fidelity=1)], "fidelity"),
    ],
)
def test_validate_oracle_results_rejects_broken_positional_contract(
    observations, message
):
    """Oracle result count, input identity, and fidelity must remain aligned."""
    candidates = [Candidate(x=1, fidelity=0)]

    with pytest.raises(ValueError, match=message):
        _validate_oracle_results(candidates, observations)


def test_validate_oracle_results_accepts_sequence_representation_changes():
    """Equivalent list, tuple, and tensor inputs should retain candidate identity."""
    candidates = [Candidate(x=(1.0, 2.0), fidelity=0)]
    observations = [Observation(x=[1.0, 2.0], y=3.0, fidelity=0)]

    _validate_oracle_results(candidates, observations)


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
    assert "active_learning/observations/new=" in out
    assert "active_learning/cost/round=" in out
    assert "active_learning/cost/cumulative=" in out
    assert "active_learning/budget/remaining=" in out
    assert "profiling/sampler/sample_s=" in out
    assert "[Logger] Run 'console_test_run' finished." in out


def test_active_learning_accumulates_prequential_surrogate_metrics() -> None:
    """The loop should aggregate held-out predictions from multiple rounds."""
    run_writer = RecordingRunWriter()
    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": lambda value: float(value),
                "fidelity_confidence": 1.0,
            }
        }
    )

    active_learning(
        dataset=ListDataset(),
        surrogate=DummyMeanSurrogate(),
        acquisition=DummyAcquisition(),
        sampler=PoolScoreSampler(
            candidate_pool=[Candidate(value, fidelity=0) for value in range(3)],
            num_samples=1,
        ),
        selector=TopKAcquisitionSelector(num_samples=1),
        oracle=oracle,
        budget=Budget(available_budget=3.0, schedule=lambda _: 1.0),
        run_writer=run_writer,
    )

    completed_diagnostics = [
        round_record["diagnostics"]
        for round_record in run_writer.rounds
        if "surrogate/general/held_out/rolling/count" in round_record["diagnostics"]
    ]
    assert len(completed_diagnostics) >= 2
    assert completed_diagnostics[0]["surrogate/general/held_out/rolling/count"] == 1
    assert completed_diagnostics[-1]["surrogate/general/held_out/rolling/count"] >= 2


def test_active_learning_fans_diagnostics_out_to_independent_sinks() -> None:
    """Writer persistence and tracker submission should receive one shared round."""
    runtime_logger = Mock()
    run_writer = RecordingRunWriter()
    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": lambda value: float(value),
                "fidelity_confidence": 1.0,
            }
        }
    )
    sampler = DiagnosticPoolSampler(
        candidate_pool=[Candidate(1, fidelity=0)],
        num_samples=1,
    )

    _, _, num_rounds = active_learning(
        dataset=ListDataset(),
        surrogate=DummyMeanSurrogate(),
        acquisition=DummyAcquisition(),
        sampler=sampler,
        selector=TopKAcquisitionSelector(num_samples=1),
        oracle=oracle,
        budget=Budget(available_budget=1.0, schedule=lambda _: 1.0),
        runtime_context=RuntimeContext(logger=runtime_logger),
        run_writer=run_writer,
    )

    assert num_rounds == 1
    assert run_writer.rounds[0]["metrics"]["active_learning/round"] == 1
    diagnostics = run_writer.rounds[0]["diagnostics"]
    assert diagnostics["acquisition/general/sampled/count"] == 1
    assert diagnostics["acquisition/general/selected/count"] == 1
    assert diagnostics["selector/general/ranking/mean"] == 1.0
    assert diagnostics["selector/general/selected_ranking/mean"] == 1.0
    logged_metrics = {
        call.args[0]: call.args[1] for call in runtime_logger.log_metric.call_args_list
    }
    assert logged_metrics["sampler/test/generated"] == 1
    assert logged_metrics["acquisition/general/sampled/count"] == 1
    assert logged_metrics["selector/general/ranking/mean"] == 1.0
    assert any(
        call.args == ("sampler/test/trajectory", sampler.created_figure)
        for call in runtime_logger.log_figure.call_args_list
    )
    assert any(
        call.args[0] == "acquisition/general/score_distribution"
        for call in runtime_logger.log_figure.call_args_list
    )
    assert sampler.created_figure.get_label() == "sampler/test/trajectory"
    runtime_logger.log_step.assert_called_once_with(1)
    assert sampler._figure is None
    assert sampler.created_figure is not None
    assert not plt.fignum_exists(sampler.created_figure.number)


def test_active_learning_discards_selector_scores_when_diagnostics_disabled() -> None:
    """Disabled diagnostics should drain transient scores without retaining them."""
    selector = TopKAcquisitionSelector(num_samples=1)
    run_writer = RecordingRunWriter()
    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": lambda value: float(value),
                "fidelity_confidence": 1.0,
            }
        }
    )

    active_learning(
        dataset=ListDataset(),
        surrogate=DummyMeanSurrogate(),
        acquisition=DummyAcquisition(),
        sampler=PoolScoreSampler(
            candidate_pool=[Candidate(1, fidelity=0)],
            num_samples=1,
        ),
        selector=selector,
        oracle=oracle,
        budget=Budget(available_budget=1.0, schedule=lambda _: 1.0),
        run_writer=run_writer,
        diagnostics_config=DiagnosticsConfig(enabled=False),
    )

    recorded_round = run_writer.rounds[0]
    assert recorded_round["diagnostics"] == {}
    assert recorded_round["metrics"]["active_learning/round"] == 1
    assert "profiling/round/total_s" in recorded_round["profiling"]
    assert selector.drain_selection_scores() is None


def test_active_learning_accepts_list_returning_selector_without_score_hook() -> None:
    """A custom selector only needs to implement the list-returning call API."""

    class ListReturningSelector:
        """Select the first candidate without exposing score telemetry."""

        def __call__(
            self,
            candidates: Sequence[Candidate],
            acquisition=None,
            cost_fn=None,
            round_budget=None,
        ) -> list[Candidate]:
            """Return one candidate from the sampled pool."""
            _ = acquisition, cost_fn, round_budget
            return list(candidates[:1])

    run_writer = RecordingRunWriter()
    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": lambda value: float(value),
                "fidelity_confidence": 1.0,
            }
        }
    )

    active_learning(
        dataset=ListDataset(),
        surrogate=DummyMeanSurrogate(),
        acquisition=DummyAcquisition(),
        sampler=PoolScoreSampler(
            candidate_pool=[Candidate(1, fidelity=0)],
            num_samples=1,
        ),
        selector=ListReturningSelector(),
        oracle=oracle,
        budget=Budget(available_budget=1.0, schedule=lambda _: 1.0),
        run_writer=run_writer,
    )

    assert run_writer.rounds[0]["diagnostics"]
    assert not any(
        key.startswith(("acquisition/", "selector/"))
        for key in run_writer.rounds[0]["diagnostics"]
    )


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


def test_active_learning_finalizes_run_writer_on_zero_round_exit():
    """Early termination still writes a complete zero-round summary."""
    run_writer = RecordingRunWriter()
    runtime_logger = Mock()
    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": lambda value: float(value),
                "fidelity_confidence": 1.0,
            }
        }
    )

    _, cost, num_rounds = active_learning(
        dataset=ListDataset(),
        surrogate=DummyMeanSurrogate(),
        acquisition=DummyAcquisition(),
        sampler=PoolScoreSampler(
            candidate_pool=[Candidate(index, fidelity=0) for index in range(3)],
            num_samples=3,
        ),
        selector=Mock(return_value=[]),
        oracle=oracle,
        budget=Budget(available_budget=3.0, schedule=lambda _: 3.0),
        runtime_context=RuntimeContext(logger=runtime_logger),
        run_writer=run_writer,
    )

    assert cost == 0.0
    assert num_rounds == 0
    assert run_writer.rounds == []
    assert run_writer.summary is not None
    assert run_writer.summary["num_rounds"] == 0
    assert run_writer.summary["total_cost"] == 0.0
    assert run_writer.summary["budget_remaining"] == 3.0
    assert run_writer.summary["elapsed_time_s"] >= 0.0
    assert run_writer.summary["num_observations"] == 0
    runtime_logger.end.assert_called_once()


def test_active_learning_records_phase_profiling_with_deterministic_clock(
    monkeypatch,
):
    """Each timed phase records a positive duration at the round boundary."""
    clock = count()
    monkeypatch.setattr(
        active_learning_module.time,
        "perf_counter",
        lambda: float(next(clock)),
    )
    run_writer = RecordingRunWriter()
    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": lambda value: float(value),
                "fidelity_confidence": 1.0,
            }
        }
    )

    _, cost, num_rounds = active_learning_module.active_learning(
        dataset=ListDataset(),
        surrogate=DummyMeanSurrogate(),
        acquisition=DummyAcquisition(),
        sampler=PoolScoreSampler(
            candidate_pool=[Candidate(index, fidelity=0) for index in range(5)],
            num_samples=5,
        ),
        selector=TopKAcquisitionSelector(num_samples=2),
        oracle=oracle,
        budget=Budget(available_budget=2.0, schedule=lambda _: 2.0),
        runtime_context=RuntimeContext(logger=Mock()),
        run_writer=run_writer,
    )

    assert cost == 2.0
    assert num_rounds == 1
    profiling = run_writer.rounds[0]["profiling"]
    assert set(profiling) == {
        "profiling/dataset/get_observations_s",
        "profiling/surrogate/fit_s",
        "profiling/acquisition/update_s",
        "profiling/sampler/sample_s",
        "profiling/budget/get_round_budget_s",
        "profiling/selector/select_s",
        "profiling/oracle/get_costs_s",
        "profiling/budget/can_afford_s",
        "profiling/budget/consume_s",
        "profiling/oracle/query_s",
        "profiling/oracle/filter_observations_s",
        "profiling/dataset/add_observations_s",
        "profiling/diagnostics/total_s",
        "profiling/round/total_s",
    }
    assert profiling["profiling/dataset/get_observations_s"] == 1.0
    assert profiling["profiling/surrogate/fit_s"] == 1.0
    assert profiling["profiling/sampler/sample_s"] == 1.0
    assert profiling["profiling/selector/select_s"] == 1.0
    assert profiling["profiling/oracle/query_s"] == 1.0
    assert profiling["profiling/diagnostics/total_s"] > 0.0
    assert profiling["profiling/round/total_s"] > 1.0


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
            self.logger.log_metric("dataset/general/records", len(self._records))


class RuntimeLoggingSurrogate(DummyMeanSurrogate, MultiFidelitySurrogate):
    """Surrogate test double that emits metrics through the bound runtime logger."""

    def set_fidelity_confidences(self, confidences: dict[int, float]) -> None:
        """Accept fidelity metadata without using it."""

    def fit(self, observations: Iterable[Observation]) -> None:
        super().fit(observations)
        if self.logger is not None:
            self.logger.log_metric("surrogate/general/fit_calls", 1)


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
            self.logger.log_metric("sampler/general/num_samples", len(samples))
        return samples


class RuntimeLoggingSelector(TopKAcquisitionSelector):
    """Selector test double that emits metrics through the bound runtime logger."""

    def __call__(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[DummyAcquisition] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        round_budget: Optional[float] = None,
    ):
        result = super().__call__(
            candidates,
            acquisition=acquisition,
            cost_fn=cost_fn,
            round_budget=round_budget,
        )
        if self.logger is not None:
            self.logger.log_metric("selector/general/selected", len(result))
        return result


class RuntimeLoggingOracle(MultiFidelityOracle):
    """Oracle test double that emits metrics through the bound runtime logger."""

    def query(self, candidates: Sequence[Candidate]) -> list[Observation]:
        observations = super().query(candidates)
        if self.logger is not None:
            self.logger.log_metric("oracle/general/queries", len(observations))
        return observations


class RuntimeLoggingBudget(Budget):
    """Budget test double that emits metrics through the bound runtime logger."""

    def get_round_budget(self, current_round: int) -> float:
        round_budget = super().get_round_budget(current_round)
        if self.logger is not None:
            self.logger.log_metric("budget/general/round_limit", round_budget)
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
    assert "surrogate/general/fit_calls=" in out
    assert "sampler/general/num_samples=" in out
    assert "selector/general/selected=" in out
    assert "oracle/general/queries=" in out
    assert "dataset/general/records=" in out
    assert "budget/general/round_limit=" in out
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
    assert run_writer.summary is not None
    assert run_writer.summary["num_rounds"] == 1
    assert run_writer.summary["total_cost"] == 3.0
    assert run_writer.summary["budget_remaining"] == 0.0
    assert run_writer.summary["num_observations"] == 2
    assert run_writer.summary["elapsed_time_s"] >= 0.0
    assert len(run_writer.rounds) == 1
    assert run_writer.rounds[0]["valid_observations"] == stored_observations
    assert [observation.x for observation in stored_observations] == [0, 2]

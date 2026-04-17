import json
import re
from unittest.mock import MagicMock, patch

import pytest

from activelearning.logger.logger import (
    AimLogger,
    CometLogger,
    MultiLogger,
    ConsoleLogger,
    Logger,
    WandbLogger,
)
from activelearning.runtime import RuntimeContext


def _override_parent_names(
    self: Logger, project_name: str, run_name: str | None = None, **kwargs: object
) -> None:
    """Simulate a parent initializer that normalizes the stored names."""

    del kwargs
    self.project_name = f"{project_name}_normalized"
    self.run_name = run_name or "normalized_run"


# ---------------------------------------------------------------------------
# ConsoleLogger tests
# ---------------------------------------------------------------------------


class TestConsoleLogger:
    """Tests for ConsoleLogger."""

    @pytest.fixture
    def logger(self, capsys):
        """Create a ConsoleLogger and clear the init output."""
        logger = ConsoleLogger(project_name="test_project", run_name="run_01")
        capsys.readouterr()  # discard __init__ output
        return logger

    def test_init_prints_project_and_run(self, capsys):
        ConsoleLogger(project_name="proj", run_name="run_x")
        out = capsys.readouterr().out
        assert "proj" in out
        assert "run_x" in out

    def test_init_default_run_name(self, capsys):
        logger = ConsoleLogger(project_name="proj")
        out = capsys.readouterr().out
        assert "proj" in out
        assert logger.run_name is not None
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{6}", logger.run_name
        )

    def test_init_uses_stored_project_name(self, capsys):
        with patch.object(Logger, "__init__", new=_override_parent_names):
            ConsoleLogger(project_name="proj", run_name="run_x")
        out = capsys.readouterr().out
        assert "proj_normalized" in out
        assert "proj'" not in out

    def test_log_config(self, logger, capsys):
        logger.log_config({"lr": 0.01, "epochs": 10})
        out = capsys.readouterr().out
        assert "lr" in out
        assert "0.01" in out
        assert "epochs" in out

    def test_log_config_valid_json(self, logger, capsys):
        config = {"lr": 0.01, "nested": {"a": 1}}
        logger.log_config(config)
        out = capsys.readouterr().out
        # Should contain valid JSON output
        assert json.loads(out.split("[Config]\n")[1]) == config

    def test_log_metric_buffered(self, logger, capsys):
        logger.log_metric("loss", 0.5)
        out = capsys.readouterr().out
        assert out == ""  # nothing printed until log_step

    def test_log_step_flushes_metrics(self, logger, capsys):
        logger.log_metric("loss", 0.5)
        logger.log_metric("acc", 0.9)
        logger.log_step(1)
        out = capsys.readouterr().out
        assert "loss" in out
        assert "acc" in out
        assert "Step 1" in out

    def test_log_step_clears_buffer(self, logger, capsys):
        logger.log_metric("loss", 0.5)
        logger.log_step(1)
        capsys.readouterr()
        logger.log_step(2)
        out = capsys.readouterr().out
        assert "loss" not in out  # buffer was cleared

    def test_log_figure_prints_key(self, logger, capsys):
        mock_fig = MagicMock()
        logger.log_figure("my_plot", mock_fig)
        out = capsys.readouterr().out
        assert "my_plot" in out

    def test_end_prints_message(self, logger, capsys):
        logger.end()
        out = capsys.readouterr().out
        assert "run_01" in out


# ---------------------------------------------------------------------------
# WandbLogger tests
# ---------------------------------------------------------------------------


class TestWandbLogger:
    """Tests for WandbLogger (wandb is mocked)."""

    @pytest.fixture
    def mock_wandb(self):
        """Provide a mock wandb module."""
        mock = MagicMock()
        mock_run = MagicMock()
        mock.init.return_value = mock_run
        return mock, mock_run

    @pytest.fixture
    def logger(self, mock_wandb):
        """Create a WandbLogger with mocked wandb."""
        mock, mock_run = mock_wandb
        with patch.dict("sys.modules", {"wandb": mock}):
            logger = WandbLogger(project_name="test_project", run_name="run_01")
        return logger, mock, mock_run

    def test_init_calls_wandb_init(self, mock_wandb):
        mock, mock_run = mock_wandb
        with patch.dict("sys.modules", {"wandb": mock}):
            WandbLogger(project_name="proj", run_name="run_x")
        mock.init.assert_called_once_with(project="proj", name="run_x")

    def test_init_uses_stored_project_name(self, mock_wandb):
        mock, mock_run = mock_wandb
        with patch.dict("sys.modules", {"wandb": mock}):
            with patch.object(Logger, "__init__", new=_override_parent_names):
                WandbLogger(project_name="proj", run_name="run_x")
        mock.init.assert_called_once_with(project="proj_normalized", name="run_x")

    def test_log_config_updates_run_config(self, logger):
        wandb_logger, mock, mock_run = logger
        config = {"lr": 0.01}
        wandb_logger.log_config(config)
        mock_run.config.update.assert_called_once_with(config)

    def test_log_metric_buffered(self, logger):
        wandb_logger, mock, mock_run = logger
        wandb_logger.log_metric("loss", 0.5)
        mock_run.log.assert_not_called()

    def test_log_step_flushes_metrics(self, logger):
        wandb_logger, mock, mock_run = logger
        wandb_logger.log_metric("loss", 0.5)
        wandb_logger.log_metric("acc", 0.9)
        wandb_logger.log_step(3)
        mock_run.log.assert_called_once_with({"loss": 0.5, "acc": 0.9}, step=3)

    def test_log_step_clears_buffer(self, logger):
        wandb_logger, mock, mock_run = logger
        wandb_logger.log_metric("loss", 0.5)
        wandb_logger.log_step(1)
        mock_run.log.reset_mock()
        wandb_logger.log_step(2)
        mock_run.log.assert_called_once_with({}, step=2)

    def test_log_figure_buffers_image(self, logger):
        wandb_logger, mock, mock_run = logger
        mock_fig = MagicMock()
        wandb_logger.log_figure("plot", mock_fig)
        mock.Image.assert_called_once_with(mock_fig)
        assert "plot" in wandb_logger._buffer

    def test_end_calls_run_finish(self, logger):
        wandb_logger, mock, mock_run = logger
        wandb_logger.end()
        mock_run.finish.assert_called_once()


# ---------------------------------------------------------------------------
# CometLogger tests
# ---------------------------------------------------------------------------


class TestCometLogger:
    """Tests for CometLogger (comet_ml is mocked)."""

    @pytest.fixture
    def mock_comet(self):
        """Provide a mock comet_ml module."""
        mock = MagicMock()
        mock_experiment = MagicMock()
        mock.Experiment.return_value = mock_experiment
        return mock, mock_experiment

    @pytest.fixture
    def logger(self, mock_comet):
        """Create a CometLogger with mocked comet_ml."""
        mock, mock_experiment = mock_comet
        with patch.dict("sys.modules", {"comet_ml": mock}):
            logger = CometLogger(
                project_name="test_project",
                run_name="run_01",
                workspace="my_workspace",
            )
        return logger, mock, mock_experiment

    def test_init_creates_experiment(self, mock_comet):
        mock, mock_experiment = mock_comet
        with patch.dict("sys.modules", {"comet_ml": mock}):
            CometLogger(project_name="proj", run_name="run_x", workspace="ws")
        mock.Experiment.assert_called_once_with(
            project_name="proj", workspace="ws", api_key=None
        )
        mock_experiment.set_name.assert_called_once_with("run_x")

    def test_init_uses_stored_project_name(self, mock_comet):
        mock, mock_experiment = mock_comet
        with patch.dict("sys.modules", {"comet_ml": mock}):
            with patch.object(Logger, "__init__", new=_override_parent_names):
                CometLogger(project_name="proj", run_name="run_x", workspace="ws")
        mock.Experiment.assert_called_once_with(
            project_name="proj_normalized", workspace="ws", api_key=None
        )

    def test_log_config_calls_log_parameters(self, logger):
        comet_logger, mock, mock_experiment = logger
        config = {"lr": 0.01}
        comet_logger.log_config(config)
        mock_experiment.log_parameters.assert_called_once_with(config)

    def test_log_metric_buffered(self, logger):
        comet_logger, mock, mock_experiment = logger
        comet_logger.log_metric("loss", 0.5)
        mock_experiment.log_metric.assert_not_called()
        assert "loss" in comet_logger._buffer

    def test_log_figure_buffered(self, logger):
        comet_logger, mock, mock_experiment = logger
        mock_fig = MagicMock()
        comet_logger.log_figure("my_plot", mock_fig)
        mock_experiment.log_figure.assert_not_called()
        assert "my_plot" in comet_logger._figure_buffer

    def test_log_step_flushes_metrics(self, logger):
        comet_logger, mock, mock_experiment = logger
        comet_logger.log_metric("loss", 0.5)
        comet_logger.log_metric("acc", 0.9)
        comet_logger.log_step(3)
        mock_experiment.log_metric.assert_any_call("loss", 0.5, step=3)
        mock_experiment.log_metric.assert_any_call("acc", 0.9, step=3)

    def test_log_step_flushes_figures(self, logger):
        comet_logger, mock, mock_experiment = logger
        mock_fig = MagicMock()
        comet_logger.log_figure("my_plot", mock_fig)
        comet_logger.log_step(3)
        mock_experiment.log_figure.assert_called_once_with(
            figure_name="my_plot", figure=mock_fig
        )

    def test_log_step_clears_buffer(self, logger):
        comet_logger, mock, mock_experiment = logger
        comet_logger.log_metric("loss", 0.5)
        comet_logger.log_step(1)
        mock_experiment.reset_mock()
        comet_logger.log_step(2)
        mock_experiment.log_metric.assert_not_called()

    def test_end_calls_experiment_end(self, logger):
        comet_logger, mock, mock_experiment = logger
        comet_logger.end()
        mock_experiment.end.assert_called_once()


# ---------------------------------------------------------------------------
# AimLogger tests
# ---------------------------------------------------------------------------


class TestAimLogger:
    """Tests for AimLogger (aim is mocked)."""

    @pytest.fixture
    def mock_aim(self):
        """Provide a mock aim module."""
        mock = MagicMock()
        mock_run = MagicMock()
        mock.Run.return_value = mock_run
        return mock, mock_run

    @pytest.fixture
    def logger(self, mock_aim):
        """Create an AimLogger with mocked aim."""
        mock, mock_run = mock_aim
        with patch.dict("sys.modules", {"aim": mock}):
            logger = AimLogger(project_name="test_project", run_name="run_01")
        return logger, mock, mock_run

    def test_init_creates_run(self, mock_aim):
        mock, mock_run = mock_aim
        with patch.dict("sys.modules", {"aim": mock}):
            AimLogger(project_name="proj", run_name="run_x")
        mock.Run.assert_called_once_with(repo=None, experiment="proj")
        assert mock_run.name == "run_x"

    def test_init_uses_stored_project_name(self, mock_aim):
        mock, mock_run = mock_aim
        with patch.dict("sys.modules", {"aim": mock}):
            with patch.object(Logger, "__init__", new=_override_parent_names):
                AimLogger(project_name="proj", run_name="run_x")
        mock.Run.assert_called_once_with(repo=None, experiment="proj_normalized")

    def test_init_with_repo(self, mock_aim):
        mock, mock_run = mock_aim
        with patch.dict("sys.modules", {"aim": mock}):
            AimLogger(project_name="proj", run_name="run_x", repo="/tmp/aim_repo")
        mock.Run.assert_called_once_with(repo="/tmp/aim_repo", experiment="proj")

    def test_log_config_sets_run_config(self, logger):
        aim_logger, mock, mock_run = logger
        config = {"lr": 0.01}
        aim_logger.log_config(config)
        assert mock_run.__setitem__.call_args[0] == ("config", config)

    def test_log_metric_buffered(self, logger):
        aim_logger, mock, mock_run = logger
        aim_logger.log_metric("loss", 0.5)
        mock_run.track.assert_not_called()

    def test_log_step_flushes_metrics(self, logger):
        aim_logger, mock, mock_run = logger
        aim_logger.log_metric("loss", 0.5)
        aim_logger.log_metric("acc", 0.9)
        aim_logger.log_step(3)
        assert mock_run.track.call_count == 2
        mock_run.track.assert_any_call(0.5, name="loss", step=3)
        mock_run.track.assert_any_call(0.9, name="acc", step=3)

    def test_log_step_clears_buffer(self, logger):
        aim_logger, mock, mock_run = logger
        aim_logger.log_metric("loss", 0.5)
        aim_logger.log_step(1)
        mock_run.track.reset_mock()
        aim_logger.log_step(2)
        mock_run.track.assert_not_called()

    def test_log_figure_buffers_aim_figure(self, logger):
        aim_logger, mock, mock_run = logger
        mock_fig = MagicMock()
        aim_logger.log_figure("my_plot", mock_fig)
        mock.Figure.assert_called_once_with(mock_fig)
        assert "my_plot" in aim_logger._buffer

    def test_log_figure_tracked_on_step(self, logger):
        aim_logger, mock, mock_run = logger
        mock_fig = MagicMock()
        aim_logger.log_figure("my_plot", mock_fig)
        aim_logger.log_step(2)
        mock_run.track.assert_called_once_with(
            mock.Figure.return_value, name="my_plot", step=2
        )

    def test_end_closes_run(self, logger):
        aim_logger, mock, mock_run = logger
        aim_logger.end()
        mock_run.close.assert_called_once()


# ---------------------------------------------------------------------------
# MultiLogger tests
# ---------------------------------------------------------------------------


class TestMultiLogger:
    """Tests for MultiLogger delegation."""

    @pytest.fixture
    def child_loggers(self):
        return [MagicMock(spec=ConsoleLogger), MagicMock(spec=ConsoleLogger)]

    @pytest.fixture
    def composite(self, child_loggers):
        return MultiLogger(loggers=child_loggers)

    def test_log_config_delegates_to_all(self, composite, child_loggers):
        config = {"lr": 0.01}
        composite.log_config(config)
        for child in child_loggers:
            child.log_config.assert_called_once_with(config)

    def test_log_metric_delegates_to_all(self, composite, child_loggers):
        composite.log_metric("loss", 0.5)
        for child in child_loggers:
            child.log_metric.assert_called_once_with("loss", 0.5)

    def test_log_figure_delegates_to_all(self, composite, child_loggers):
        mock_fig = MagicMock()
        composite.log_figure("plot", mock_fig)
        for child in child_loggers:
            child.log_figure.assert_called_once_with("plot", mock_fig)

    def test_log_step_delegates_to_all(self, composite, child_loggers):
        composite.log_step(5)
        for child in child_loggers:
            child.log_step.assert_called_once_with(5)

    def test_end_delegates_to_all(self, composite, child_loggers):
        composite.end()
        for child in child_loggers:
            child.end.assert_called_once()


# ---------------------------------------------------------------------------
# Active learning loop integration test
# ---------------------------------------------------------------------------


class TestActiveLearningLoopIntegration:
    """Tests for logger integration in the active learning loop."""

    def _make_loop_components(self):
        """Build minimal components for running the active learning loop."""
        from activelearning.acquisition.dummy_acquisition import DummyAcquisition
        from activelearning.budget.budget import Budget
        from activelearning.dataset.list_dataset import ListDataset
        from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
        from activelearning.sampler.pool_score_sampler import PoolScoreSampler
        from activelearning.selector.score_selector import TopKAcquisitionSelector
        from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
        from activelearning.utils.types import Candidate

        dataset = ListDataset()
        surrogate = DummyMeanSurrogate()
        acquisition = DummyAcquisition()
        candidate_pool = [Candidate(float(i), 1) for i in range(20)]
        sampler = PoolScoreSampler(candidate_pool=candidate_pool, num_samples=10)
        selector = TopKAcquisitionSelector(num_samples=2)
        oracle = MultiFidelityOracle(
            fidelity_configs={
                1: {
                    "cost_per_sample": 1.0,
                    "score_fn": lambda x: float(x),
                    "fidelity_confidence": 1.0,
                }
            }
        )
        budget = Budget(available_budget=3.0, schedule=lambda r: 1.0)
        return dataset, surrogate, acquisition, sampler, selector, oracle, budget

    def test_loop_calls_logger_methods(self):
        """Logger methods are called during the active learning loop."""
        from activelearning.active_learning import active_learning

        components = self._make_loop_components()
        dataset, surrogate, acquisition, sampler, selector, oracle, budget = components
        mock_logger = MagicMock()

        active_learning(
            dataset=dataset,
            surrogate=surrogate,
            acquisition=acquisition,
            sampler=sampler,
            selector=selector,
            oracle=oracle,
            budget=budget,
            runtime_context=RuntimeContext(logger=mock_logger),
        )

        assert mock_logger.log_metric.called
        assert mock_logger.log_step.called
        mock_logger.end.assert_called_once()

    def test_loop_without_logger_runs_fine(self):
        """Active learning loop runs correctly when no logger is provided."""
        from activelearning.active_learning import active_learning

        components = self._make_loop_components()
        dataset, surrogate, acquisition, sampler, selector, oracle, budget = components

        dataset_result, total_cost, num_rounds = active_learning(
            dataset=dataset,
            surrogate=surrogate,
            acquisition=acquisition,
            sampler=sampler,
            selector=selector,
            oracle=oracle,
            budget=budget,
        )
        assert num_rounds >= 0
        assert total_cost >= 0.0

    def test_loop_logs_expected_metrics(self):
        """Active learning loop logs expected metric keys."""
        from activelearning.active_learning import active_learning

        components = self._make_loop_components()
        dataset, surrogate, acquisition, sampler, selector, oracle, budget = components

        logged_keys = []
        mock_logger = MagicMock()
        mock_logger.log_metric.side_effect = lambda key, value: logged_keys.append(key)

        active_learning(
            dataset=dataset,
            surrogate=surrogate,
            acquisition=acquisition,
            sampler=sampler,
            selector=selector,
            oracle=oracle,
            budget=budget,
            runtime_context=RuntimeContext(logger=mock_logger),
        )

        expected_keys = {
            "round",
            "round_cost",
            "total_cost",
            "budget_remaining",
        }
        assert expected_keys.issubset(set(logged_keys))

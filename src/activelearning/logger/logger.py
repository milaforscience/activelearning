import json
from abc import ABC, abstractmethod
from datetime import datetime
from numbers import Real
from typing import Any

from matplotlib.figure import Figure as MatplotlibFigure


class Logger(ABC):
    """Abstract base class for experiment loggers.

    Defines a unified interface for logging metrics, figures, and configuration
    across different backends (console, wandb, Comet ML, etc.).

    Parameters
    ----------
    project_name : str
        Name of the project or experiment group.
    run_name : str, optional
        Name of this specific run. Defaults to a timestamp with microseconds
        if not provided.
    """

    def __init__(
        self, project_name: str, run_name: str | None = None, **kwargs: Any
    ) -> None:
        self.project_name = project_name
        if run_name is None:
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.run_name = run_name

    @abstractmethod
    def log_config(self, config: dict[str, Any]) -> None:
        """Log experiment configuration/hyperparameters.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary of configuration keys and values to log.
        """

    @abstractmethod
    def log_metric(self, key: str, value: Any) -> None:
        """Log a scalar metric value.

        Parameters
        ----------
        key : str
            Name of the metric.
        value : Any
            Value of the metric (typically a float or int).
        """

    @abstractmethod
    def log_figure(self, key: str, figure: Any) -> None:
        """Log a figure or plot.

        Parameters
        ----------
        key : str
            Name or identifier for the figure.
        figure : Any
            The figure object to log (e.g., matplotlib Figure).
        """

    @abstractmethod
    def log_step(self, step: int) -> None:
        """Commit buffered metrics for the current step.

        Parameters
        ----------
        step : int
            The current step or iteration number.
        """

    @abstractmethod
    def end(self) -> None:
        """Finalize and close the logging session."""


class ConsoleLogger(Logger):
    """Logger that outputs experiment data to the console (stdout).

    Provides human-readable logging to stdout. Figures are not displayed
    but their keys are acknowledged. Metrics are buffered and flushed
    together at each step.

    Parameters
    ----------
    project_name : str
        Name of the project or experiment group.
    run_name : str, optional
        Name of this specific run. Defaults to a timestamp if not provided.
    """

    def __init__(
        self, project_name: str, run_name: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(project_name, run_name, **kwargs)
        self._buffer: dict[str, Any] = {}
        print(f"[Logger] Project: {self.project_name!r} | Run: {self.run_name!r}")

    def log_config(self, config: dict[str, Any]) -> None:
        """Log experiment configuration to the console as formatted JSON.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary of configuration keys and values to log.
        """
        print("[Config]")
        print(json.dumps(config, indent=2, default=str))

    def log_metric(self, key: str, value: Any) -> None:
        """Buffer a metric for display at the next log_step call.

        Parameters
        ----------
        key : str
            Name of the metric.
        value : Any
            Value of the metric.
        """
        self._buffer[key] = value

    def log_figure(self, key: str, figure: Any) -> None:
        """Acknowledge a figure (not displayable in console).

        Parameters
        ----------
        key : str
            Name or identifier for the figure.
        figure : Any
            The figure object (not rendered in console).
        """
        print(f"[Figure] {key!r} (not rendered in console)")

    def log_step(self, step: int) -> None:
        """Print all buffered metrics for this step and clear the buffer.

        Parameters
        ----------
        step : int
            The current step or iteration number.
        """
        metrics_str = " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in self._buffer.items()
        )
        print(f"[Step {step}] {metrics_str}")
        self._buffer = {}

    def end(self) -> None:
        """Print a run-end message to the console."""
        print(f"[Logger] Run {self.run_name!r} finished.")


class WandbLogger(Logger):
    """Logger that sends experiment data to Weights & Biases (wandb).

    Lazily imports wandb at construction time. Metrics and figures are
    buffered and flushed together on each ``log_step`` call.

    Parameters
    ----------
    project_name : str
        Name of the wandb project.
    run_name : str, optional
        Name of this specific run. Defaults to a timestamp if not provided.
    """

    def __init__(
        self, project_name: str, run_name: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(project_name, run_name, **kwargs)
        import wandb  # type: ignore[import-not-found]  # optional dependency

        self._wandb = wandb
        self.run = wandb.init(project=self.project_name, name=self.run_name)
        self._buffer: dict[str, Any] = {}

    def log_config(self, config: dict[str, Any]) -> None:
        """Update the wandb run configuration with experiment settings.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary of configuration keys and values to log.
        """
        self.run.config.update(config)

    def log_metric(self, key: str, value: Any) -> None:
        """Buffer a metric to be logged at the next log_step call.

        Parameters
        ----------
        key : str
            Name of the metric.
        value : Any
            Value of the metric.
        """
        self._buffer[key] = value

    def log_figure(self, key: str, figure: Any) -> None:
        """Buffer a figure as a wandb Image to be logged at the next log_step call.

        Parameters
        ----------
        key : str
            Name or identifier for the figure.
        figure : Any
            The figure object (e.g., matplotlib Figure).
        """
        self._buffer[key] = self._wandb.Image(figure)

    def log_step(self, step: int) -> None:
        """Flush all buffered metrics and figures to wandb for this step.

        Parameters
        ----------
        step : int
            The current step or iteration number.
        """
        self.run.log(self._buffer, step=step)
        self._buffer = {}

    def end(self) -> None:
        """Finish the wandb run."""
        self.run.finish()


class CometLogger(Logger):
    """Logger that sends experiment data to Comet ML.

    Lazily imports comet_ml at construction time. Comet ML logs
    metrics immediately (not buffered), but step tracking is maintained
    internally via ``log_step``.

    Parameters
    ----------
    project_name : str
        Name of the Comet ML project.
    run_name : str, optional
        Name of this specific run. Defaults to a timestamp if not provided.
    workspace : str, optional
        Comet ML workspace name.
    api_key : str, optional
        Comet ML API key. Falls back to COMET_API_KEY environment variable.
    """

    def __init__(
        self,
        project_name: str,
        run_name: str | None = None,
        workspace: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(project_name, run_name, **kwargs)
        import comet_ml  # type: ignore[import-not-found]  # optional dependency

        self._comet_ml = comet_ml
        self.experiment = comet_ml.Experiment(
            project_name=self.project_name,
            workspace=workspace,
            api_key=api_key,
        )
        self.experiment.set_name(self.run_name)
        self._current_step: int = 0

    def log_config(self, config: dict[str, Any]) -> None:
        """Log experiment configuration as Comet ML hyperparameters.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary of configuration keys and values to log.
        """
        self.experiment.log_parameters(config)

    def log_metric(self, key: str, value: Any) -> None:
        """Log a metric-like value to Comet ML at the current step.

        Numeric values are sent to Comet as metrics. Non-numeric values are
        recorded as step-tagged text entries instead, which avoids Comet's
        warning about string metrics while preserving the information.

        Parameters
        ----------
        key : str
            Name of the metric.
        value : Any
            Value of the metric.
        """
        if isinstance(value, Real) and not isinstance(value, bool):
            self.experiment.log_metric(key, value, step=self._current_step)
            return

        self.experiment.log_text(
            str(value),
            step=self._current_step,
            metadata={"key": key},
        )

    def log_figure(self, key: str, figure: Any) -> None:
        """Log a figure to Comet ML.

        Parameters
        ----------
        key : str
            Name or identifier for the figure.
        figure : Any
            The figure object (e.g., matplotlib Figure).
        """
        self.experiment.log_figure(figure_name=key, figure=figure)

    def log_step(self, step: int) -> None:
        """Update the current step context for subsequent metric logging.

        Parameters
        ----------
        step : int
            The current step or iteration number.
        """
        self._current_step = step

    def end(self) -> None:
        """End the Comet ML experiment."""
        self.experiment.end()


class AimLogger(Logger):
    """Logger that sends experiment data to Aim.

    Lazily imports aim at construction time. Metrics and figures are
    buffered and flushed together on each ``log_step`` call.

    Parameters
    ----------
    project_name : str
        Name of the Aim experiment (maps to ``experiment`` in ``aim.Run``).
    run_name : str, optional
        Name of this specific run. Defaults to a timestamp if not provided.
    repo : str, optional
        Path or URL of the Aim repository. Defaults to ``None``, which uses
        the ``.aim`` directory in the current working directory.
    """

    def __init__(
        self,
        project_name: str,
        run_name: str | None = None,
        repo: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(project_name, run_name, **kwargs)
        import aim  # type: ignore[import-not-found]  # optional dependency

        self._aim = aim
        self.run = aim.Run(repo=repo, experiment=self.project_name)
        self.run.name = self.run_name
        self._buffer: dict[str, Any] = {}

    def log_config(self, config: dict[str, Any]) -> None:
        """Log experiment configuration as Aim hyperparameters.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary of configuration keys and values to log.
        """
        self.run["config"] = config

    def log_metric(self, key: str, value: Any) -> None:
        """Buffer a metric-like value to be tracked at the next log_step call.

        Aim accepts numeric scalars or Aim objects in ``track()``. Non-numeric
        values are therefore wrapped as ``aim.Text`` so string-valued metadata
        such as ``best_candidate`` can still be logged safely.

        Parameters
        ----------
        key : str
            Name of the metric.
        value : Any
            Value of the metric.
        """
        if isinstance(value, Real) and not isinstance(value, bool):
            self._buffer[key] = value
            return

        self._buffer[key] = self._aim.Text(str(value))

    def log_figure(self, key: str, figure: Any) -> None:
        """Buffer a figure to be tracked at the next log_step call.

        Parameters
        ----------
        key : str
            Name or identifier for the figure.
        figure : Any
            The figure object. Matplotlib figures are buffered as ``aim.Image``
            to avoid Aim's Plotly conversion path; other figure-like objects
            continue to use ``aim.Figure``.
        """
        if isinstance(figure, MatplotlibFigure):
            self._buffer[key] = self._aim.Image(figure)
            return

        self._buffer[key] = self._aim.Figure(figure)

    def log_step(self, step: int) -> None:
        """Flush all buffered metrics and figures to Aim for this step.

        Parameters
        ----------
        step : int
            The current step or iteration number.
        """
        for key, value in self._buffer.items():
            self.run.track(value, name=key, step=step)
        self._buffer = {}

    def end(self) -> None:
        """Close the Aim run."""
        self.run.close()


class MultiLogger(Logger):
    """Logger that forwards all calls to a list of child loggers.

    Enables simultaneous logging to multiple backends (e.g., console and wandb).
    The MultiLogger itself does not hold project/run state; each child
    manages its own.

    Parameters
    ----------
    loggers : list[Logger]
        List of logger instances to delegate to.
    """

    def __init__(self, loggers: list[Logger]) -> None:
        # Do not call super().__init__() — MultiLogger has no project/run of its own.
        self._loggers = loggers

    def log_config(self, config: dict[str, Any]) -> None:
        """Forward log_config to all child loggers.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary of configuration keys and values to log.
        """
        for logger in self._loggers:
            logger.log_config(config)

    def log_metric(self, key: str, value: Any) -> None:
        """Forward log_metric to all child loggers.

        Parameters
        ----------
        key : str
            Name of the metric.
        value : Any
            Value of the metric.
        """
        for logger in self._loggers:
            logger.log_metric(key, value)

    def log_figure(self, key: str, figure: Any) -> None:
        """Forward log_figure to all child loggers.

        Parameters
        ----------
        key : str
            Name or identifier for the figure.
        figure : Any
            The figure object to log.
        """
        for logger in self._loggers:
            logger.log_figure(key, figure)

    def log_step(self, step: int) -> None:
        """Forward log_step to all child loggers.

        Parameters
        ----------
        step : int
            The current step or iteration number.
        """
        for logger in self._loggers:
            logger.log_step(step)

    def end(self) -> None:
        """Forward end to all child loggers."""
        for logger in self._loggers:
            logger.end()

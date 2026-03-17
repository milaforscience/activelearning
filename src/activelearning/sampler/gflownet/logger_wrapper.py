from __future__ import annotations

from typing import Any

from activelearning.logger.logger import Logger as ActiveLearningLogger
from gflownet.utils.logger import Logger as GFlowNetLogger


class RuntimeGFlowNetLoggerWrapper(GFlowNetLogger):
    """Mirror GFlowNet logs into an activelearning runtime logger.

    The external ``gflownet`` package expects its own logger contract, including
    checkpoint directories and progress-bar settings. This wrapper preserves that
    contract by subclassing the upstream logger while forwarding compatible
    metrics and figures to the runtime-bound activelearning logger.
    """

    def __init__(
        self,
        runtime_logger: ActiveLearningLogger,
        config: Any,
        logger_conf: Any,
    ) -> None:
        super().__init__(
            config=config,
            **self._build_logger_kwargs(logger_conf),
        )
        self._runtime_logger = runtime_logger
        self._last_step: int | None = None

    @staticmethod
    def _build_logger_kwargs(logger_conf: Any) -> dict[str, Any]:
        """Translate the Hydra logger config into upstream logger kwargs."""
        tags = logger_conf.get("tags")
        return {
            "do": logger_conf.do,
            "project_name": logger_conf.project_name,
            "logdir": logger_conf.logdir,
            "lightweight": logger_conf.lightweight,
            "debug": logger_conf.debug,
            "run_name": logger_conf.get("run_name"),
            "run_name_date": logger_conf.get("run_name_date", True),
            "run_name_job": logger_conf.get("run_name_job", True),
            "run_id": logger_conf.get("run_id"),
            "tags": list(tags) if tags is not None else None,
            "context": logger_conf.get("context", "0"),
            "notes": logger_conf.get("notes"),
            "entity": logger_conf.get("entity"),
            "progressbar": logger_conf.progressbar,
            "is_resumed": logger_conf.get("is_resumed", False),
        }

    def _format_key(self, key: str, use_context: bool) -> str:
        """Return the metric/figure name, optionally prefixed by logger context."""
        if use_context:
            return f"{self.context}/{key}"
        return key

    def _flush_runtime_logger(self, step: int) -> None:
        """Flush mirrored runtime logs for a specific GFlowNet step."""
        self._last_step = step
        self._runtime_logger.log_step(step)

    @staticmethod
    def _coerce_log_value(value: Any) -> Any:
        """Convert tensor-like scalar values into plain Python scalars when possible."""
        if value is None:
            return None

        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except (TypeError, ValueError):
                return value
        return value

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        use_context: bool = True,
    ) -> None:
        """Log GFlowNet metrics and mirror them into the runtime logger."""
        super().log_metrics(metrics, step=step, use_context=use_context)

        emitted_metrics = False
        for key, value in metrics.items():
            coerced_value = self._coerce_log_value(value)
            if coerced_value is None:
                continue
            self._runtime_logger.log_metric(
                self._format_key(key, use_context=use_context),
                coerced_value,
            )
            emitted_metrics = True

        if emitted_metrics:
            self._flush_runtime_logger(step)

    def log_rewards_and_scores(
        self,
        rewards: Any,
        logrewards: Any,
        scores: Any,
        step: int,
        prefix: str,
        use_context: bool = True,
    ) -> None:
        """Aggregate reward statistics and route them through ``log_metrics``."""
        metrics: dict[str, Any] = {
            f"{prefix} rewards mean": rewards.mean(),
            f"{prefix} rewards max": rewards.max(),
            f"{prefix} logrewards mean": logrewards.mean(),
            f"{prefix} logrewards max": logrewards.max(),
        }
        if scores is not None:
            metrics.update(
                {
                    f"{prefix} scores mean": scores.mean(),
                    f"{prefix} scores min": scores.min(),
                    f"{prefix} scores max": scores.max(),
                }
            )

        self.log_metrics(metrics, step=step, use_context=use_context)

    def log_plots(
        self,
        figs: dict[str, Any] | list[Any],
        step: int,
        use_context: bool = True,
    ) -> None:
        """Log evaluator figures and mirror them to the runtime logger."""
        named_figures: list[tuple[str, Any]] = []
        if isinstance(figs, dict):
            named_figures = list(figs.items())
        else:
            named_figures = [
                (f"Figure {index} at step {step}", figure)
                for index, figure in enumerate(figs)
            ]

        emitted_figures = False
        for key, figure in named_figures:
            if figure is None:
                continue
            self._runtime_logger.log_figure(
                self._format_key(key, use_context=use_context),
                figure,
            )
            emitted_figures = True

        if emitted_figures:
            self._flush_runtime_logger(step)

        super().log_plots(figs, step=step, use_context=use_context)

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Mirror evaluation summaries into the runtime logger when a step is known."""
        super().log_summary(summary)
        if self._last_step is None:
            return

        emitted_metrics = False
        for key, value in summary.items():
            coerced_value = self._coerce_log_value(value)
            if coerced_value is None:
                continue
            self._runtime_logger.log_metric(f"summary/{key}", coerced_value)
            emitted_metrics = True

        if emitted_metrics:
            self._flush_runtime_logger(self._last_step)

    def log_time(self, times: dict[str, Any], use_context: bool) -> None:
        """Log timing metrics without relying on the upstream logger's buggy helper."""
        if not self.do.times:
            return

        step = 0 if self._last_step is None else self._last_step
        prefixed_times = {f"time_{key}": value for key, value in times.items()}
        self.log_metrics(prefixed_times, step=step, use_context=use_context)

    def end(self) -> None:
        """Close only the GFlowNet logger backend.

        The runtime logger is owned by the outer active-learning loop and should
        be ended there exactly once.
        """
        super().end()

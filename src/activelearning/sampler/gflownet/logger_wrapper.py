from __future__ import annotations

from numbers import Real
import re
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from gflownet.utils.logger import Logger as GFlowNetLogger


class RuntimeGFlowNetLoggerWrapper(GFlowNetLogger):
    """Collect GFlowNet diagnostics under the activelearning namespace.

    The external ``gflownet`` package expects its own logger contract, including
    checkpoint directories and progress-bar settings. This wrapper preserves that
    contract by subclassing the upstream logger while retaining compatible
    metrics and figures for the enclosing sampler to drain after one AL round.
    """

    def __init__(
        self,
        config: Any,
        **logger_kwargs: Any,
    ) -> None:
        super().__init__(config=config, **logger_kwargs)
        self._last_step: int | None = None
        self._pending_metrics: dict[str, int | float] = {}
        self._pending_figures: dict[str, Figure] = {}

    def _format_key(self, key: str, use_context: bool) -> str:
        """Return a normalized runtime key under the GFlowNet namespace."""
        normalized_parts = []
        for part in key.strip().split("/"):
            normalized_part = re.sub(r"[^a-zA-Z0-9]+", "_", part).strip("_").lower()
            if normalized_part:
                normalized_parts.append(normalized_part)
        if normalized_parts and normalized_parts[-1].startswith("loss"):
            normalized_parts[-1] = f"gflownet_{normalized_parts[-1]}"
        normalized_key = "/".join(normalized_parts)
        prefix = "sampler/gflownet"
        if use_context:
            context = "/".join(
                part
                for part in (
                    re.sub(r"[^a-zA-Z0-9]+", "_", raw_part).strip("_").lower()
                    for raw_part in str(self.context).split("/")
                )
                if part
            )
            if context:
                prefix = f"{prefix}/{context}"
        if not normalized_key:
            return prefix
        return f"{prefix}/{normalized_key}"

    def _record_step(self, step: int) -> None:
        """Track the latest GFlowNet step without advancing the AL round."""
        self._last_step = step

    @staticmethod
    def _convert_log_value(value: Any) -> Any:
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
        """Log upstream metrics and retain compatible scalar round diagnostics."""
        formatted_metrics = {
            self._format_key(key, use_context=use_context): value
            for key, value in metrics.items()
        }
        super().log_metrics(formatted_metrics, step=step, use_context=False)

        for key, value in formatted_metrics.items():
            converted_value = self._convert_log_value(value)
            if not isinstance(converted_value, Real) or isinstance(
                converted_value, bool
            ):
                continue
            self._pending_metrics[key] = float(converted_value)
        self._record_step(step)

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
        """Log upstream figures and retain Matplotlib figures for round draining."""
        named_figures: list[tuple[str, Any]] = []
        if isinstance(figs, dict):
            named_figures = list(figs.items())
        else:
            named_figures = [
                (f"figure/{index}", figure) for index, figure in enumerate(figs)
            ]

        formatted_figures = {
            self._format_key(key, use_context=use_context): figure
            for key, figure in named_figures
        }
        for key, figure in formatted_figures.items():
            if not isinstance(figure, Figure):
                continue
            self._pending_figures[key] = figure
        self._record_step(step)
        super().log_plots(formatted_figures, step=step, use_context=False)

    def log_histogram(
        self,
        key: str,
        value: Any,
        step: int,
        use_context: bool = True,
    ) -> None:
        """Forward histograms with the sampler namespace applied."""
        super().log_histogram(
            self._format_key(key, use_context=use_context),
            value,
            step=step,
            use_context=False,
        )

    def close_figs(self, figs: list[Any] | dict[str, Any]) -> None:
        """Defer GFlowNet figure cleanup to the active-learning round boundary."""
        _ = figs

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Retain numeric evaluation summaries when a GFlowNet step is known."""
        formatted_summary = {
            self._format_key(f"summary/{key}", use_context=False): value
            for key, value in summary.items()
        }
        super().log_summary(formatted_summary)
        if self._last_step is None:
            return

        for key, value in formatted_summary.items():
            converted_value = self._convert_log_value(value)
            if not isinstance(converted_value, Real) or isinstance(
                converted_value, bool
            ):
                continue
            self._pending_metrics[key] = float(converted_value)

    def log_time(self, times: dict[str, Any], use_context: bool) -> None:
        """Log timing metrics without relying on the upstream logger's buggy helper."""
        if not self.do.times:
            return

        step = 0 if self._last_step is None else self._last_step
        prefixed_times = {f"timing/{key}": value for key, value in times.items()}
        self.log_metrics(prefixed_times, step=step, use_context=use_context)

    def drain_round_diagnostics(
        self,
        *,
        include_figures: bool,
        max_points: int,
    ) -> tuple[dict[str, int | float], dict[str, Figure]]:
        """Return and clear metrics and figures collected during one AL round."""
        _ = max_points
        metrics = self._pending_metrics
        figures = self._pending_figures if include_figures else {}
        if not include_figures:
            for figure in self._pending_figures.values():
                plt.close(figure)
        self._pending_metrics = {}
        self._pending_figures = {}
        return metrics, figures

    def end(self) -> None:
        """Close only the upstream GFlowNet logger backend."""
        super().end()

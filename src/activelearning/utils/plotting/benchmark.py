"""Reusable loading, aggregation, and plotting helpers for run benchmarks."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np

MetricValue: TypeAlias = float | None
MetricCallback: TypeAlias = Callable[[Sequence[Any]], MetricValue]
WarningCallback: TypeAlias = Callable[[str], None]


@dataclass(frozen=True)
class MethodStyle:
    """Visual style for one method in a comparison plot."""

    label: str
    color: str
    linestyle: str = "-"


@dataclass(frozen=True)
class RunCheckpoint:
    """Metrics and cumulative observations for one completed run round."""

    method: str
    seed: int | str | None
    round_index: int
    cumulative_cost: float
    observations: tuple[Any, ...]
    metrics: Mapping[str, MetricValue]

    @property
    def observation_count(self) -> int:
        """Return the number of cumulative observations at this checkpoint."""

        return len(self.observations)


Checkpoint = RunCheckpoint


@dataclass(frozen=True)
class MetricStatistics:
    """Mean and sample standard deviation for one aggregated metric."""

    mean: float | None
    std: float


@dataclass(frozen=True)
class AggregatedCheckpoint:
    """Cross-seed statistics for one method and round."""

    round_index: int
    seed_count: int
    cumulative_cost: MetricStatistics
    metrics: Mapping[str, MetricStatistics]

    @property
    def cumulative_cost_mean(self) -> float:
        """Return the mean cumulative cost for this checkpoint."""

        if self.cumulative_cost.mean is None:
            return 0.0
        return float(self.cumulative_cost.mean)

    @property
    def cumulative_cost_std(self) -> float:
        """Return the sample standard deviation of cumulative cost."""

        return self.cumulative_cost.std


def warn(message: str) -> None:
    """Print a warning message to standard error."""

    print(f"Warning: {message}", file=sys.stderr)


def compute_mean_top_k(values: Sequence[float], *, k: int) -> float | None:
    """Return the mean of the top-k values, or all values when fewer exist."""

    if not values:
        return None
    top_values = sorted((float(value) for value in values), reverse=True)[:k]
    return fmean(top_values)


mean_top_k = compute_mean_top_k


def load_initial_observations(
    manifest_path: Path,
    *,
    warn_fn: WarningCallback = warn,
) -> list[Any]:
    """Load initial serialized observations from one run manifest.

    Missing, empty, or malformed manifests are tolerated. In those cases the
    run is still processed, but the initial observation pool is empty.
    """

    if not manifest_path.is_file():
        warn_fn(f"Missing run manifest: {manifest_path}")
        return []

    text = manifest_path.read_text(encoding="utf-8")
    if not text.strip():
        warn_fn(f"Empty run manifest: {manifest_path}")
        return []

    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as error:
        warn_fn(f"Malformed run manifest {manifest_path}: {error}")
        return []

    if not isinstance(manifest, Mapping):
        warn_fn(f"Malformed run manifest {manifest_path}: expected an object")
        return []

    initial_data = manifest.get("initial_data", {})
    if not isinstance(initial_data, Mapping):
        warn_fn(f"Malformed initial_data section in {manifest_path}")
        return []

    observations = initial_data.get("initial_observations", [])
    if not isinstance(observations, list):
        warn_fn(f"Malformed initial_observations section in {manifest_path}")
        return []

    return list(observations)


def load_run_checkpoints(
    run_dir: Path,
    *,
    method: str,
    seed: int | str | None,
    metric_callbacks: Mapping[str, MetricCallback],
    warn_fn: WarningCallback = warn,
) -> list[RunCheckpoint]:
    """Load one JSON-lines run and evaluate metrics at every round.

    The loader follows the artifact contract of ``JSONLinesRunWriter``:
    ``run_manifest.json`` contributes ``initial_data.initial_observations`` and
    each ``round_history.jsonl`` record contributes its
    ``new_observations``, ``round_index``, and ``cumulative_cost`` fields.
    Observation payloads are kept opaque to this generic layer.
    """

    history_path = run_dir / "round_history.jsonl"
    if not history_path.is_file():
        warn_fn(f"Missing round history for {method} seed directory: {run_dir}")
        return []

    cumulative_observations = load_initial_observations(
        run_dir / "run_manifest.json",
        warn_fn=warn_fn,
    )
    checkpoints: list[RunCheckpoint] = []

    with history_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                warn_fn(f"Skipping empty JSONL line {line_number} in {history_path}")
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                warn_fn(
                    f"Skipping malformed JSONL line {line_number} in {history_path}: "
                    f"{error}"
                )
                continue

            if not isinstance(record, Mapping):
                warn_fn(
                    f"Skipping round record on line {line_number} in {history_path}: "
                    "expected an object"
                )
                continue

            try:
                round_index = int(record["round_index"])
                cumulative_cost = float(record["cumulative_cost"])
            except (KeyError, TypeError, ValueError) as error:
                warn_fn(
                    f"Skipping round record on line {line_number} in {history_path}: "
                    f"{error}"
                )
                continue

            new_observations = record.get("new_observations", [])
            if not isinstance(new_observations, list):
                warn_fn(
                    f"Skipping malformed new_observations on line {line_number} "
                    f"in {history_path}"
                )
                continue

            cumulative_observations.extend(new_observations)
            observations = tuple(cumulative_observations)
            metrics = {
                name: _coerce_metric_value(callback(observations))
                for name, callback in metric_callbacks.items()
            }
            checkpoints.append(
                RunCheckpoint(
                    method=method,
                    seed=seed,
                    round_index=round_index,
                    cumulative_cost=cumulative_cost,
                    observations=observations,
                    metrics=metrics,
                )
            )

    return checkpoints


def aggregate_checkpoints(
    rows: Sequence[RunCheckpoint],
) -> list[AggregatedCheckpoint]:
    """Aggregate one method's per-seed checkpoints by round index."""

    grouped: dict[int, list[RunCheckpoint]] = {}
    for row in rows:
        grouped.setdefault(row.round_index, []).append(row)

    aggregated: list[AggregatedCheckpoint] = []
    for round_index in sorted(grouped):
        group = grouped[round_index]
        metric_names = _metric_names(group)
        aggregated.append(
            AggregatedCheckpoint(
                round_index=round_index,
                seed_count=len(group),
                cumulative_cost=_mean_and_std([row.cumulative_cost for row in group]),
                metrics={
                    name: _mean_and_std(
                        [
                            row.metrics[name]
                            for row in group
                            if row.metrics.get(name) is not None
                        ]
                    )
                    for name in metric_names
                },
            )
        )
    return aggregated


def aggregate_runs(
    data: Mapping[str, Sequence[RunCheckpoint]],
) -> dict[str, list[AggregatedCheckpoint]]:
    """Aggregate all methods by round index across their seeds."""

    return {method: aggregate_checkpoints(rows) for method, rows in data.items()}


def plot_metric(
    aggregated_data: Mapping[str, Sequence[AggregatedCheckpoint]],
    *,
    metric: str,
    output_path: Path,
    method_order: Sequence[str] | None = None,
    method_styles: Mapping[str, MethodStyle] | None = None,
    x_label: str = "Cumulative Cost",
    y_label: str | None = None,
    title: str = "Benchmark",
    log_scale: bool = False,
) -> None:
    """Render one aggregated metric against mean cumulative cost.

    Parameters
    ----------
    aggregated_data : Mapping[str, Sequence[AggregatedCheckpoint]]
        Per-method, cross-seed checkpoint statistics.
    metric : str
        Name of the metric in each checkpoint's ``metrics`` mapping.
    output_path : Path
        Destination PNG path.
    method_order : Sequence[str] | None, optional
        Methods to render and their legend order. If omitted, input order is
        used.
    method_styles : Mapping[str, MethodStyle] | None, optional
        Optional presentation styles. Missing styles receive a default tab10
        color and the method name as their label.
    x_label : str, optional
        Label for the cumulative-cost x-axis.
    y_label : str | None, optional
        Label for the metric axis. Defaults to the metric name.
    title : str, optional
        Figure title.
    log_scale : bool, optional
        Whether to use a logarithmic x-axis.
    """

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    figure, axis = plt.subplots(figsize=(9, 5.5))
    methods = (
        tuple(method_order) if method_order is not None else tuple(aggregated_data)
    )
    styles = method_styles or {}
    default_colors = plt.get_cmap("tab10")
    plotted = False

    for index, method in enumerate(methods):
        rows = aggregated_data.get(method)
        if not rows:
            continue
        filtered_rows = []
        for row in rows:
            statistics = row.metrics.get(metric)
            if statistics is not None and statistics.mean is not None:
                filtered_rows.append(row)
        if not filtered_rows:
            continue

        style = styles.get(
            method,
            MethodStyle(
                label=method,
                color=default_colors(index % 10),
            ),
        )
        x_values = np.asarray(
            [row.cumulative_cost_mean for row in filtered_rows],
            dtype=float,
        )
        y_values = np.asarray(
            [float(row.metrics[metric].mean) for row in filtered_rows],
            dtype=float,
        )
        y_std = np.asarray(
            [row.metrics[metric].std for row in filtered_rows],
            dtype=float,
        )
        axis.plot(
            x_values,
            y_values,
            color=style.color,
            linestyle=style.linestyle,
            linewidth=2.0,
            label=style.label,
        )
        axis.fill_between(
            x_values,
            y_values - y_std,
            y_values + y_std,
            color=style.color,
            alpha=0.15,
        )
        plotted = True

    axis.set_xlabel(f"{x_label} (log)" if log_scale else x_label)
    axis.set_ylabel(y_label or metric)
    axis.set_title(title)
    if log_scale:
        axis.set_xscale("log")
        axis.minorticks_on()
        axis.grid(which="both", alpha=0.25)
    else:
        axis.grid(alpha=0.25)
    if plotted:
        axis.legend(loc="best", frameon=False)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)
    print(f"Figure saved to {output_path}")


def _coerce_metric_value(value: MetricValue) -> MetricValue:
    """Normalize metric callback output to a float or ``None``."""

    return None if value is None else float(value)


def _metric_names(rows: Sequence[RunCheckpoint]) -> tuple[str, ...]:
    """Return metric names in their first-seen order across rows."""

    names: dict[str, None] = {}
    for row in rows:
        for name in row.metrics:
            names.setdefault(name, None)
    return tuple(names)


def _mean_and_std(values: Sequence[float | None]) -> MetricStatistics:
    """Return an arithmetic mean and sample standard deviation."""

    present_values = [float(value) for value in values if value is not None]
    if not present_values:
        return MetricStatistics(mean=None, std=0.0)
    if len(present_values) == 1:
        return MetricStatistics(mean=present_values[0], std=0.0)
    return MetricStatistics(mean=fmean(present_values), std=stdev(present_values))

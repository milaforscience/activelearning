"""Plot cached xTB IP/EA benchmark metrics without invoking an oracle."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_METHODS = (
    "mf_gfn",
    "random_fidelity_gfn",
    "sf_gfn",
    "random",
)
TASKS = ("ip", "ea")
METRICS = ("mean_top_100_score", "mean_top_100_diversity")
DEFAULT_BUDGET = 1260.0
BOOTSTRAP_RESAMPLES = 5000
METHOD_STYLES = {
    "mf_gfn": ("MF-GFN", "#1f77b4", "-"),
    "random_fidelity_gfn": ("Random-fidelity GFN", "#ff7f0e", "--"),
    "sf_gfn": ("SF-GFN", "#2ca02c", "-."),
    "random": ("Random", "#d62728", ":"),
}


def load_metrics(path: Path) -> list[dict[str, Any]]:
    """Load rows from the versioned evaluator JSON artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported molecule metrics artifact: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"Metrics artifact {path} must contain a rows list.")
    return [dict(row) for row in rows]


def bootstrap_interval(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for a mean."""
    if not values:
        raise ValueError("At least one value is required for bootstrap intervals.")
    if resamples < 1:
        raise ValueError("resamples must be positive.")
    if len(values) == 1:
        value = float(values[0])
        return value, value
    sample_indices = rng.integers(0, len(values), size=(resamples, len(values)))
    samples = np.asarray(values, dtype=float)[sample_indices].mean(axis=1)
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def build_plot_data(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
    expected_methods: Sequence[str] = EXPECTED_METHODS,
    budget: float = DEFAULT_BUDGET,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    """Aggregate per-run rows on each task's common cost axis."""
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        try:
            task = str(row["task"])
            method = str(row["method"])
            seed = int(row["seed"])
            cost = float(row["cumulative_acquisition_cost"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed metrics row: {row!r}") from error
        if task not in TASKS or method not in expected_methods:
            continue
        if cost <= budget:
            grouped.setdefault((task, method, seed), []).append(row)

    expected_seed_set = set(expected_seeds)
    if not allow_incomplete:
        for task in TASKS:
            for method in expected_methods:
                found = {
                    seed
                    for (row_task, row_method, seed) in grouped
                    if row_task == task and row_method == method
                }
                missing = sorted(expected_seed_set - found)
                if missing:
                    raise ValueError(
                        f"Missing seeds for {task}/{method}: {missing}. "
                        "Pass --allow-incomplete to plot partial results."
                    )

    rng = np.random.default_rng(0)
    output: list[dict[str, Any]] = []
    for task in TASKS:
        costs = sorted(
            {
                float(row["cumulative_acquisition_cost"])
                for (row_task, _, _), method_rows in grouped.items()
                if row_task == task
                for row in method_rows
            }
        )
        for method in expected_methods:
            for cost in costs:
                values_by_seed: list[float] = []
                for seed in expected_seeds:
                    checkpoints = grouped.get((task, method, seed), [])
                    eligible = [
                        row
                        for row in checkpoints
                        if float(row["cumulative_acquisition_cost"]) <= cost
                    ]
                    if not eligible:
                        continue
                    latest = max(
                        eligible,
                        key=lambda row: (
                            float(row["cumulative_acquisition_cost"]),
                            int(row["round"]),
                        ),
                    )
                    value = latest.get("mean_top_100_score")
                    if value is not None:
                        values_by_seed.append(float(value))
                _append_metric_row(
                    output,
                    task=task,
                    method=method,
                    cost=cost,
                    metric="mean_top_100_score",
                    values=values_by_seed,
                    effective_seed_count=len(values_by_seed),
                    expected_seed_count=len(expected_seeds),
                    rng=rng,
                    allow_incomplete=allow_incomplete,
                )
                # Diversity is aggregated separately so the effective seed
                # count reflects only finite values for that metric.
                diversity_values: list[float] = []
                for seed in expected_seeds:
                    checkpoints = grouped.get((task, method, seed), [])
                    eligible = [
                        row
                        for row in checkpoints
                        if float(row["cumulative_acquisition_cost"]) <= cost
                    ]
                    if not eligible:
                        continue
                    latest = max(
                        eligible,
                        key=lambda row: (
                            float(row["cumulative_acquisition_cost"]),
                            int(row["round"]),
                        ),
                    )
                    value = latest.get("mean_top_100_diversity")
                    if value is not None:
                        diversity_values.append(float(value))
                _append_metric_row(
                    output,
                    task=task,
                    method=method,
                    cost=cost,
                    metric="mean_top_100_diversity",
                    values=diversity_values,
                    effective_seed_count=len(diversity_values),
                    expected_seed_count=len(expected_seeds),
                    rng=rng,
                    allow_incomplete=allow_incomplete,
                )
    return output


def plot_benchmark(
    metrics_path: Path,
    output_dir: Path,
    *,
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
    allow_incomplete: bool = False,
    budget: float = DEFAULT_BUDGET,
) -> list[dict[str, Any]]:
    """Write the 2x2 score/diversity figure and plot-data CSV."""
    rows = load_metrics(metrics_path)
    plot_data = build_plot_data(
        rows,
        expected_seeds=expected_seeds,
        allow_incomplete=allow_incomplete,
        budget=budget,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_plot_data(output_dir / "molecule_benchmark_plot_data.csv", plot_data)
    _write_figure(output_dir, plot_data, allow_incomplete=allow_incomplete)
    return plot_data


def _append_metric_row(
    output: list[dict[str, Any]],
    *,
    task: str,
    method: str,
    cost: float,
    metric: str,
    values: Sequence[float],
    effective_seed_count: int,
    expected_seed_count: int,
    rng: np.random.Generator,
    allow_incomplete: bool,
) -> None:
    """Append one finite cross-seed aggregate or a visible missing row."""
    if not values:
        if not allow_incomplete:
            return
        mean = lower = upper = None
    else:
        mean = float(np.mean(values))
        lower, upper = bootstrap_interval(values, rng=rng)
    output.append(
        {
            "task": task,
            "method": method,
            "cumulative_acquisition_cost": cost,
            "metric": metric,
            "mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
            "effective_seed_count": effective_seed_count,
            "expected_seed_count": expected_seed_count,
        }
    )


def _write_plot_data(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the aggregated plotting table."""
    fieldnames = [
        "task",
        "method",
        "cumulative_acquisition_cost",
        "metric",
        "mean",
        "ci_lower",
        "ci_upper",
        "effective_seed_count",
        "expected_seed_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_incomplete: bool,
) -> None:
    """Render the stable 2x2 benchmark figure."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    for column, task in enumerate(TASKS):
        for row_index, metric in enumerate(METRICS):
            axis = axes[row_index][column]
            for method in EXPECTED_METHODS:
                method_rows = [
                    row
                    for row in rows
                    if row["task"] == task
                    and row["method"] == method
                    and row["metric"] == metric
                    and row["mean"] is not None
                ]
                if not method_rows:
                    continue
                style = METHOD_STYLES[method]
                x_values = np.asarray(
                    [row["cumulative_acquisition_cost"] for row in method_rows],
                    dtype=float,
                )
                means = np.asarray([row["mean"] for row in method_rows], dtype=float)
                lower = np.asarray(
                    [row["ci_lower"] for row in method_rows], dtype=float
                )
                upper = np.asarray(
                    [row["ci_upper"] for row in method_rows], dtype=float
                )
                axis.plot(
                    x_values,
                    means,
                    color=style[1],
                    linestyle=style[2],
                    linewidth=2.0,
                    label=style[0],
                )
                axis.fill_between(x_values, lower, upper, color=style[1], alpha=0.14)
            axis.grid(alpha=0.25)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if row_index == 0:
                axis.set_title("Negative IP" if task == "ip" else "Electron affinity")
                axis.set_ylabel("Mean top-100 score (higher is better)")
            else:
                axis.set_ylabel("Mean pairwise Tanimoto distance")
            axis.set_xlabel("Cumulative acquisition cost")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    suffix = " (incomplete seeds)" if allow_incomplete else ""
    figure.suptitle(f"xTB IP/EA molecule benchmark{suffix}", y=1.02)
    figure.tight_layout()
    figure.savefig(output_dir / "molecule_benchmark.svg", bbox_inches="tight")
    figure.savefig(output_dir / "molecule_benchmark.png", dpi=250, bbox_inches="tight")
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    """Build the plotting CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Plot cached benchmark metrics from the command line."""
    args = _build_parser().parse_args(argv)
    plot_benchmark(
        args.metrics_path.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        allow_incomplete=args.allow_incomplete,
        budget=args.budget,
    )


if __name__ == "__main__":
    main()

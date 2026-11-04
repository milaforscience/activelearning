"""Plot Branin benchmark presentation metrics from structured run outputs.

The script reads ``run_manifest.json`` and ``round_history.jsonl`` artifacts
written by ``JSONLinesRunWriter`` for the Branin benchmark experiment. Generic
artifact loading, aggregation, and comparison-curve rendering live in
``activelearning.utils.plotting``; this script supplies the Branin metrics and
presentation defaults.

Usage
-----
    uv run python scripts/plot_branin_benchmark.py

    uv run python scripts/plot_branin_benchmark.py \
        --metric mean-top-k \
        --scale log

    uv run python scripts/plot_branin_benchmark.py \
        --base-dir outputs/branin_benchmark \
        --output-dir plots
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from botorch.test_functions.multi_fidelity import AugmentedBranin

from activelearning.utils.plotting import (
    AggregatedCheckpoint,
    MethodStyle,
    RunCheckpoint,
    aggregate_runs,
    compute_mean_top_k,
    load_run_checkpoints,
    plot_metric as plot_generic_metric,
    warn,
)

MetricName = Literal["mean_top_k", "mode_coverage"]
ScaleName = Literal["linear", "log"]
WarningCallback = Callable[[str], None]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = REPOSITORY_ROOT / "outputs" / "branin_benchmark"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "plots"
DEFAULT_TOP_K = 50
DEFAULT_COVERAGE_K = 5
BRANIN_MODES: tuple[tuple[float, float], ...] = (
    (-math.pi, 12.275),
    (math.pi, 2.275),
    (3.0 * math.pi, 2.475),
)


METHOD_ORDER: tuple[str, ...] = (
    "mf_gfn",
    "random",
    "sf_low_fid",
    "sf_mid_fid",
    "sf_high_fid",
)

METHOD_STYLES: dict[str, MethodStyle] = {
    "mf_gfn": MethodStyle("MF-GFN", "#1f77b4"),
    "random": MethodStyle("Random", "#d62728"),
    "sf_low_fid": MethodStyle("SF-GFN-LOW", "#17becf"),
    "sf_mid_fid": MethodStyle("SF-GFN-MID", "#2ca02c"),
    "sf_high_fid": MethodStyle("SF-GFN-HIGH", "#9467bd"),
}

_BRANIN_RESCORE_FN = AugmentedBranin(negate=True)


def positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI argument."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _as_point(value: Any) -> tuple[float, float] | None:
    """Convert an observation payload into a finite 2-D point."""

    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return None
    if len(value) != 2:
        return None
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(component) for component in point) else None


def extract_points(
    observations: Sequence[Any],
    *,
    source: str,
    warn_fn: WarningCallback = warn,
) -> list[tuple[float, float]]:
    """Extract valid Branin x-locations from observation payloads."""

    points: list[tuple[float, float]] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            warn_fn(
                f"Skipping malformed observation #{index} from {source}: "
                "expected object."
            )
            continue
        point = _as_point(observation.get("x"))
        if point is None:
            warn_fn(
                f"Skipping observation #{index} from {source}: "
                "expected finite 2D 'x' list."
            )
            continue
        points.append(point)
    return points


def rescore_at_highest_fidelity(
    points: Sequence[tuple[float, float]],
) -> list[float]:
    """Evaluate Branin x-locations at the highest fidelity confidence."""

    if not points:
        return []
    x_with_fidelity = torch.tensor(
        [[point[0], point[1], 1.0] for point in points],
        dtype=torch.double,
    )
    with torch.no_grad():
        values = _BRANIN_RESCORE_FN(x_with_fidelity)
    return [float(value) for value in values.tolist()]


def mode_coverage_metric(
    points: Sequence[Sequence[float]],
    *,
    k: int,
    modes: Sequence[Sequence[float]] = BRANIN_MODES,
) -> float | None:
    """Return the max-KNN distance to the known Branin modes."""

    if not points:
        return None

    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        return None

    neighbor_count = min(k, len(point_array))
    per_mode_averages: list[float] = []
    for mode in modes:
        distances = np.linalg.norm(point_array - np.asarray(mode, dtype=float), axis=1)
        nearest_distances = np.partition(distances, neighbor_count - 1)[:neighbor_count]
        per_mode_averages.append(float(np.mean(nearest_distances)))
    return max(per_mode_averages)


def parse_seed(seed_dir_name: str, *, warn_fn: WarningCallback = warn) -> int:
    """Parse an integer seed from a ``seed_*`` directory name."""

    try:
        return int(seed_dir_name.split("_", 1)[1])
    except (IndexError, ValueError):
        warn_fn(f"Could not parse seed from directory name '{seed_dir_name}'; using 0.")
        return 0


def load_seed_checkpoints(
    seed_dir: Path,
    *,
    method: str,
    top_k: int = DEFAULT_TOP_K,
    coverage_k: int = DEFAULT_COVERAGE_K,
    warn_fn: WarningCallback = warn,
) -> list[RunCheckpoint]:
    """Load one seed directory and compute both Branin benchmark metrics."""

    seed = parse_seed(seed_dir.name, warn_fn=warn_fn)
    last_observations: Sequence[Any] | None = None
    last_points: list[tuple[float, float]] = []

    def points_for(observations: Sequence[Any]) -> list[tuple[float, float]]:
        """Extract points once for both Branin metrics at one checkpoint."""

        nonlocal last_observations, last_points
        if observations is not last_observations:
            last_points = extract_points(
                observations,
                source=str(seed_dir),
                warn_fn=warn_fn,
            )
            last_observations = observations
        return last_points

    metric_callbacks = {
        "mean_top_k": lambda observations: compute_mean_top_k(
            rescore_at_highest_fidelity(points_for(observations)),
            k=top_k,
        ),
        "mode_coverage": lambda observations: mode_coverage_metric(
            points_for(observations),
            k=coverage_k,
        ),
    }
    generic_rows = load_run_checkpoints(
        seed_dir,
        method=method,
        seed=seed,
        metric_callbacks=metric_callbacks,
        warn_fn=warn_fn,
    )
    return generic_rows


def load_benchmark_data(
    base_dir: Path,
    *,
    methods: Sequence[str] = METHOD_ORDER,
    top_k: int = DEFAULT_TOP_K,
    coverage_k: int = DEFAULT_COVERAGE_K,
    warn_fn: WarningCallback = warn,
) -> dict[str, list[RunCheckpoint]]:
    """Load all requested Branin benchmark methods under one output root."""

    data: dict[str, list[RunCheckpoint]] = {}
    for method in methods:
        method_dir = base_dir / method
        if not method_dir.is_dir():
            warn_fn(f"Skipping missing method directory: {method_dir}")
            continue

        method_rows: list[RunCheckpoint] = []
        seed_dirs = sorted(
            directory
            for directory in method_dir.iterdir()
            if directory.is_dir() and directory.name.startswith("seed_")
        )
        if not seed_dirs:
            warn_fn(
                f"No seed_* directories found for method '{method}' in {method_dir}"
            )
            continue

        for seed_dir in seed_dirs:
            method_rows.extend(
                load_seed_checkpoints(
                    seed_dir,
                    method=method,
                    top_k=top_k,
                    coverage_k=coverage_k,
                    warn_fn=warn_fn,
                )
            )

        if not method_rows:
            warn_fn(f"No valid round checkpoints loaded for method '{method}'")
            continue
        data[method] = method_rows

    return data


def aggregate_benchmark_data(
    data: Mapping[str, Sequence[RunCheckpoint]],
) -> dict[str, list[AggregatedCheckpoint]]:
    """Aggregate all loaded Branin methods by round across seeds."""

    return aggregate_runs(data)


def metric_axis_label(metric: MetricName, *, top_k: int, coverage_k: int) -> str:
    """Return the y-axis label for the selected Branin metric."""

    if metric == "mean_top_k":
        return f"Mean top-{top_k} score @ highest fidelity \u2191"
    return f"Mode coverage (max-KNN, K={coverage_k}) @ highest fidelity \u2193"


def metric_filename(metric: MetricName) -> str:
    """Return the stable filename stem for one Branin metric."""

    return "mean_top_k" if metric == "mean_top_k" else "mode_coverage"


def output_path_for(
    output_dir: Path,
    *,
    metric: MetricName,
    scale: ScaleName,
) -> Path:
    """Return the canonical output path for one metric/scale pair."""

    suffix = "_log" if scale == "log" else ""
    return output_dir / f"branin_benchmark_{metric_filename(metric)}{suffix}.png"


def plot_metric(
    aggregated_data: Mapping[str, Sequence[AggregatedCheckpoint]],
    *,
    metric: MetricName,
    output_path: Path,
    top_k: int = DEFAULT_TOP_K,
    coverage_k: int = DEFAULT_COVERAGE_K,
    log_scale: bool = False,
) -> None:
    """Render and save one Branin benchmark presentation figure."""

    plot_generic_metric(
        aggregated_data,
        metric=metric,
        output_path=output_path,
        method_order=METHOD_ORDER,
        method_styles=METHOD_STYLES,
        x_label="Cumulative Cost",
        y_label=metric_axis_label(
            metric,
            top_k=top_k,
            coverage_k=coverage_k,
        ),
        title="Multi-Fidelity Branin Benchmark",
        log_scale=log_scale,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the Branin benchmark plotting CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot mean-top-k and mode-coverage curves for the Branin benchmark. "
            "By default, all metric/scale combinations are generated."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing method/seed run outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated PNG figures.",
    )
    parser.add_argument(
        "--metric",
        choices=("mean-top-k", "mode-coverage", "all"),
        default="all",
        help="Metric to plot. Use 'all' to generate both presentation metrics.",
    )
    parser.add_argument(
        "--scale",
        choices=("linear", "log", "all"),
        default="all",
        help="X-axis scale. Use 'all' to generate both linear and log plots.",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=DEFAULT_TOP_K,
        help="Top-k cutoff for the mean-top-k metric.",
    )
    parser.add_argument(
        "--coverage-k",
        type=positive_int,
        default=DEFAULT_COVERAGE_K,
        help="Nearest-neighbor count for the max-KNN mode-coverage metric.",
    )
    return parser


def resolve_metrics(selection: str) -> tuple[MetricName, ...]:
    """Expand a CLI metric selection into concrete metric names."""

    if selection == "all":
        return ("mean_top_k", "mode_coverage")
    return ("mean_top_k",) if selection == "mean-top-k" else ("mode_coverage",)


def resolve_scales(selection: str) -> tuple[ScaleName, ...]:
    """Expand a CLI scale selection into concrete x-axis scales."""

    if selection == "all":
        return ("linear", "log")
    return (selection,)  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> None:
    """Run the standalone Branin benchmark plotting CLI."""

    args = build_parser().parse_args(argv)
    base_dir = args.base_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    run_data = load_benchmark_data(
        base_dir,
        top_k=args.top_k,
        coverage_k=args.coverage_k,
    )
    if not run_data:
        raise SystemExit(f"No valid Branin benchmark runs found under {base_dir}")

    aggregated_data = aggregate_benchmark_data(run_data)
    for metric in resolve_metrics(args.metric):
        for scale in resolve_scales(args.scale):
            plot_metric(
                aggregated_data,
                metric=metric,
                output_path=output_path_for(output_dir, metric=metric, scale=scale),
                top_k=args.top_k,
                coverage_k=args.coverage_k,
                log_scale=scale == "log",
            )


if __name__ == "__main__":
    main()

"""Plot paper-reproduction results from recorded run outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Literal, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

if __package__ in {None, ""}:
    from reproduce_paper_metrics import RUN_MANIFEST_FILENAME, collect_run_metrics
else:
    from scripts.reproduce_paper_metrics import (
        RUN_MANIFEST_FILENAME,
        collect_run_metrics,
    )

PlotTaskGroup = Literal["synthetic", "molecules"]

TASK_ORDER_BY_GROUP: dict[PlotTaskGroup, tuple[str, ...]] = {
    "synthetic": ("branin", "hartmann"),
    "molecules": ("molecules_ip", "molecules_ea"),
}

TASK_LABELS: dict[str, str] = {
    "branin": "Branin",
    "hartmann": "Hartmann",
    "molecules_ip": "Molecules (IP)",
    "molecules_ea": "Molecules (EA)",
}

METHOD_ORDER: tuple[str, ...] = (
    "mf_gfn",
    "mf_gfn_stack",
    "random_fid_gfn",
    "sf_gfn",
    "random",
)


@dataclass(frozen=True)
class MethodStyle:
    """Visual style metadata for one paper baseline."""

    label: str
    color: str
    linestyle: str
    marker: str = "o"


METHOD_STYLES: dict[str, MethodStyle] = {
    "mf_gfn": MethodStyle(label="MF-GFN", color="#1f77b4", linestyle="-"),
    "mf_gfn_stack": MethodStyle(
        label="MF-GFN-STACK",
        color="#9467bd",
        linestyle="--",
    ),
    "random_fid_gfn": MethodStyle(
        label="Random fid. GFN",
        color="#ff7f0e",
        linestyle="--",
    ),
    "sf_gfn": MethodStyle(label="SF-GFN", color="#2ca02c", linestyle="-."),
    "random": MethodStyle(label="Random", color="#d62728", linestyle=":"),
}


def main(argv: Sequence[str] | None = None) -> None:
    """Run the unified paper-reproduction plotting CLI."""

    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rows = load_checkpoint_rows(args.inputs, task_group=args.task_group)
    if args.task_group == "synthetic":
        plotting_rows = aggregate_checkpoint_rows(
            checkpoint_rows,
            task_group="synthetic",
            y_keys=("mean_top_k_score", "best_so_far_y", "simple_regret"),
        )
        write_rows_to_csv(
            plotting_rows,
            output_dir / "synthetic_plotting_table.csv",
        )
        figure = build_synthetic_figure(plotting_rows)
        figure_path = output_dir / "synthetic_results.png"
    else:
        plotting_rows = aggregate_checkpoint_rows(
            checkpoint_rows,
            task_group="molecules",
            y_keys=(
                "mean_top_k_score",
                "mean_top_k_energy",
                "mean_pairwise_tanimoto_distance",
            ),
        )
        write_rows_to_csv(
            plotting_rows,
            output_dir / "molecule_plotting_table.csv",
        )
        figure = build_molecule_figure(plotting_rows)
        figure_path = output_dir / "molecule_results.png"

    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def load_checkpoint_rows(
    inputs: Sequence[Path],
    *,
    task_group: PlotTaskGroup,
) -> list[dict[str, Any]]:
    """Load flattened checkpoint rows from metrics JSONs and/or run directories."""

    if not inputs:
        raise ValueError("At least one plotting input path is required.")

    loaded_rows: list[dict[str, Any]] = []
    run_inputs: list[Path] = []
    for input_path in inputs:
        resolved = input_path.expanduser().resolve()
        if resolved.is_file() and _looks_like_metrics_catalog(resolved):
            loaded_rows.extend(
                _load_rows_from_metrics_catalog(resolved, task_group=task_group)
            )
            continue
        run_inputs.append(resolved)

    if run_inputs:
        catalog = collect_run_metrics(run_inputs)
        checkpoint_rows = (
            catalog.synthetic_checkpoint_rows()
            if task_group == "synthetic"
            else catalog.molecule_checkpoint_rows()
        )
        loaded_rows.extend(row.to_dict() for row in checkpoint_rows)

    _validate_plotting_rows(loaded_rows, task_group=task_group)
    return sorted(
        loaded_rows,
        key=lambda row: (
            *_task_sort_key(str(row["task"]), task_group=task_group),
            *_method_sort_key(str(row["method"])),
            int(row["seed"]),
            int(row["round_index"]),
        ),
    )


def aggregate_checkpoint_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_group: PlotTaskGroup,
    y_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate per-seed checkpoint rows into roundwise plotting tables."""

    grouped_rows: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        key = (
            str(row["task"]),
            str(row["method"]),
            int(row["round_index"]),
        )
        grouped_rows[key].append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for task in _ordered_tasks(rows, task_group=task_group):
        task_rows = [row for row in rows if str(row["task"]) == task]
        for method in _ordered_methods(task_rows):
            round_indices = sorted(
                round_index
                for grouped_task, grouped_method, round_index in grouped_rows
                if grouped_task == task and grouped_method == method
            )
            for round_index in round_indices:
                group = grouped_rows[(task, method, round_index)]
                x_values = [float(row["cumulative_budget"]) for row in group]
                aggregated_row: dict[str, Any] = {
                    "task": task,
                    "task_label": task_label(task),
                    "method": method,
                    "method_label": method_style(method).label,
                    "round_index": round_index,
                    "top_k": int(group[0]["top_k"]),
                    "seed_count": len({int(row["seed"]) for row in group}),
                    "cumulative_budget_mean": fmean(x_values),
                    "cumulative_budget_std": _standard_deviation(x_values),
                }
                for y_key in y_keys:
                    mean_value, std_value = _aggregate_optional_values(
                        row.get(y_key) for row in group
                    )
                    aggregated_row[f"{y_key}_mean"] = mean_value
                    aggregated_row[f"{y_key}_std"] = std_value
                aggregated_rows.append(aggregated_row)

    return aggregated_rows


def write_rows_to_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Persist a plotting table as CSV."""

    if not rows:
        raise ValueError("Cannot write an empty plotting table.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def build_synthetic_figure(
    plotting_rows: Sequence[dict[str, object]],
) -> Figure:
    """Render the synthetic paper-style figure from aggregated plotting rows."""

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    tasks = _ordered_tasks(plotting_rows, task_group="synthetic")
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(max(6.0, 6.0 * len(tasks)), 4.6),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    figure.subplots_adjust(top=0.78, wspace=0.28)

    legend_handles = []
    legend_labels = []
    seen_methods: set[str] = set()
    for axis, task in zip(flat_axes, tasks, strict=True):
        task_rows = [row for row in plotting_rows if str(row["task"]) == task]
        top_k = int(task_rows[0]["top_k"])
        axis.set_title(f"{task_rows[0]['task_label']} (top-{top_k})")
        axis.set_xlabel("Cumulative budget")
        axis.grid(alpha=0.25)
        if axis is flat_axes[0]:
            axis.set_ylabel("Mean top-k score")

        for method in _ordered_methods(task_rows):
            method_rows = [row for row in task_rows if str(row["method"]) == method]
            style = method_style(method)
            x_values = np.asarray(
                [float(row["cumulative_budget_mean"]) for row in method_rows],
                dtype=float,
            )
            y_values = _as_float_array(
                row["mean_top_k_score_mean"] for row in method_rows
            )
            y_std = _as_float_array(row["mean_top_k_score_std"] for row in method_rows)
            (line,) = axis.plot(
                x_values,
                y_values,
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
                linewidth=2.0,
                markersize=4.5,
                label=style.label,
            )
            axis.fill_between(
                x_values,
                y_values - y_std,
                y_values + y_std,
                color=style.color,
                alpha=0.14,
            )
            if method not in seen_methods:
                legend_handles.append(line)
                legend_labels.append(style.label)
                seen_methods.add(method)

    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=min(4, len(legend_labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    return figure


def build_molecule_figure(plotting_rows: Sequence[dict[str, object]]) -> Figure:
    """Render the molecule paper-style figure from aggregated plotting rows."""

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    tasks = _ordered_tasks(plotting_rows, task_group="molecules")
    figure, axes = plt.subplots(
        2,
        len(tasks),
        figsize=(max(6.0, 6.0 * len(tasks)), 7.2),
        sharex="col",
        squeeze=False,
    )
    figure.subplots_adjust(top=0.78, hspace=0.28, wspace=0.25)

    legend_handles = []
    legend_labels = []
    seen_methods: set[str] = set()
    for column_index, task in enumerate(tasks):
        task_rows = [row for row in plotting_rows if str(row["task"]) == task]
        top_k = int(task_rows[0]["top_k"])
        energy_axis = axes[0, column_index]
        diversity_axis = axes[1, column_index]
        energy_axis.set_title(f"{task_rows[0]['task_label']} (top-{top_k})")
        energy_axis.grid(alpha=0.25)
        diversity_axis.grid(alpha=0.25)
        diversity_axis.set_xlabel("Cumulative budget")
        diversity_axis.set_ylim(0.0, 1.0)

        if column_index == 0:
            energy_axis.set_ylabel("Mean top-k energy")
            diversity_axis.set_ylabel("Mean pairwise Tanimoto distance")

        for method in _ordered_methods(task_rows):
            method_rows = [row for row in task_rows if str(row["method"]) == method]
            style = method_style(method)
            x_values = np.asarray(
                [float(row["cumulative_budget_mean"]) for row in method_rows],
                dtype=float,
            )

            energy_values = _as_float_array(
                row["mean_top_k_energy_mean"] for row in method_rows
            )
            energy_std = _as_float_array(
                row["mean_top_k_energy_std"] for row in method_rows
            )
            (line,) = energy_axis.plot(
                x_values,
                energy_values,
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
                linewidth=2.0,
                markersize=4.5,
                label=style.label,
            )
            energy_axis.fill_between(
                x_values,
                energy_values - energy_std,
                energy_values + energy_std,
                color=style.color,
                alpha=0.14,
            )

            diversity_values = _as_float_array(
                row["mean_pairwise_tanimoto_distance_mean"] for row in method_rows
            )
            diversity_std = _as_float_array(
                row["mean_pairwise_tanimoto_distance_std"] for row in method_rows
            )
            diversity_axis.plot(
                x_values,
                diversity_values,
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
                linewidth=2.0,
                markersize=4.5,
            )
            diversity_axis.fill_between(
                x_values,
                diversity_values - diversity_std,
                diversity_values + diversity_std,
                color=style.color,
                alpha=0.14,
            )

            if method not in seen_methods:
                legend_handles.append(line)
                legend_labels.append(style.label)
                seen_methods.add(method)

    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=min(4, len(legend_labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    return figure


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate paper-reproduction runs into paper-style figures and save "
            "the exact plotting table used."
        )
    )
    parser.add_argument(
        "--task-group",
        choices=("synthetic", "molecules"),
        required=True,
        help="Which paper figure family to aggregate and render.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "One or more run roots, run directories, run manifests, or metrics "
            "catalog JSON files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the figure and plotting table will be written.",
    )
    return parser


def _looks_like_metrics_catalog(path: Path) -> bool:
    """Return whether a file path should be parsed as a metrics catalog JSON."""

    return path.suffix == ".json" and path.name != RUN_MANIFEST_FILENAME


def _load_rows_from_metrics_catalog(
    path: Path,
    *,
    task_group: PlotTaskGroup,
) -> list[dict[str, Any]]:
    """Load flattened checkpoint rows from one serialized metrics catalog."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    key = f"{task_group}_checkpoint_rows"
    rows = payload.get(key)
    if not isinstance(rows, list):
        raise ValueError(
            f"Metrics catalog {path!s} does not contain a list at {key!r}."
        )
    return [dict(row) for row in rows]


def _validate_plotting_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_group: PlotTaskGroup,
) -> None:
    """Reject empty plotting inputs for the requested task group."""

    if not rows:
        raise ValueError(f"No {task_group} checkpoint rows were found in the inputs.")


def _ordered_tasks(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_group: PlotTaskGroup,
) -> tuple[str, ...]:
    """Return observed tasks ordered by the paper's preferred task order."""

    observed_tasks = {str(row["task"]) for row in rows}
    ordered_tasks = [
        task for task in TASK_ORDER_BY_GROUP[task_group] if task in observed_tasks
    ]
    ordered_tasks.extend(sorted(observed_tasks.difference(ordered_tasks)))
    return tuple(ordered_tasks)


def _ordered_methods(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return observed methods ordered by the preferred plotting order."""

    observed_methods = {str(row["method"]) for row in rows}
    ordered_methods = [method for method in METHOD_ORDER if method in observed_methods]
    ordered_methods.extend(sorted(observed_methods.difference(ordered_methods)))
    return tuple(ordered_methods)


def _task_sort_key(task: str, *, task_group: PlotTaskGroup) -> tuple[int, int | str]:
    """Return a stable sort key for one task name."""

    if task in TASK_ORDER_BY_GROUP[task_group]:
        return (0, TASK_ORDER_BY_GROUP[task_group].index(task))
    return (1, task)


def _method_sort_key(method: str) -> tuple[int, int | str]:
    """Return a stable sort key for one plotting method name."""

    if method in METHOD_ORDER:
        return (0, METHOD_ORDER.index(method))
    return (1, method)


def _aggregate_optional_values(
    values: Iterable[Any],
) -> tuple[float | None, float | None]:
    """Return mean and sample standard deviation for optional numeric values."""

    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None, None
    return fmean(numeric_values), _standard_deviation(numeric_values)


def _as_float_array(values: Sequence[object] | Sequence[float | None]) -> np.ndarray:
    """Convert optional scalar values into a NumPy float array with NaNs."""

    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=float,
    )


def _standard_deviation(values: Sequence[float]) -> float:
    """Return sample standard deviation, defaulting to zero for one value."""

    if len(values) < 2:
        return 0.0
    return stdev(values)


def method_style(method: str) -> MethodStyle:
    """Return the configured style for one paper method identifier."""

    try:
        return METHOD_STYLES[method]
    except KeyError as error:
        raise ValueError(f"Unsupported plotting method: {method!r}.") from error


def task_label(task: str) -> str:
    """Return the paper-facing task label for one experiment key."""

    try:
        return TASK_LABELS[task]
    except KeyError as error:
        raise ValueError(f"Unsupported plotting task: {task!r}.") from error


if __name__ == "__main__":
    main()

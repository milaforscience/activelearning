import json
from pathlib import Path

import pytest

from scripts.plot_molecule_benchmark import (
    EXPECTED_METHODS,
    build_plot_data,
    plot_benchmark,
)


def metric_rows() -> list[dict[str, object]]:
    """Build a complete three-seed metrics fixture with staggered costs."""
    rows: list[dict[str, object]] = []
    for task in ("ip", "ea"):
        for method_index, method in enumerate(EXPECTED_METHODS):
            for seed_index, seed in enumerate((42, 43, 44)):
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "round": 1,
                        "cumulative_acquisition_cost": 10.0 * (seed_index + 1),
                        "mean_top_100_score": float(method_index + seed_index),
                        "mean_top_100_diversity": 0.1 * (method_index + 1),
                    }
                )
    return rows


def test_plot_data_uses_common_costs_and_carries_completed_checkpoints() -> None:
    """A seed is carried forward only after its first completed checkpoint."""
    rows = build_plot_data(metric_rows())

    score_rows = [
        row
        for row in rows
        if row["task"] == "ip"
        and row["method"] == "mf_s3gfn"
        and row["metric"] == "mean_top_100_score"
    ]
    assert [row["cumulative_acquisition_cost"] for row in score_rows] == [
        10.0,
        20.0,
        30.0,
    ]
    assert [row["effective_seed_count"] for row in score_rows] == [1, 2, 3]
    assert [row["mean"] for row in score_rows] == [0.0, 0.5, 1.0]


def test_plot_data_rejects_missing_seed_by_default() -> None:
    """The default figure contract requires all three declared seeds."""
    incomplete = [row for row in metric_rows() if row["seed"] != 44]

    with pytest.raises(ValueError, match="Missing seeds"):
        build_plot_data(incomplete)

    rows = build_plot_data(incomplete, allow_incomplete=True)
    assert rows


def test_plotting_writes_all_requested_artifacts_without_oracle_calls(
    tmp_path: Path,
) -> None:
    """Plotting consumes only metrics JSON and writes SVG, PNG, and CSV."""
    metrics_path = tmp_path / "molecule_metrics.json"
    metrics_path.write_text(
        json.dumps({"schema_version": 1, "rows": metric_rows()}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "plots"

    plot_benchmark(metrics_path, output_dir)

    assert (output_dir / "molecule_benchmark.svg").is_file()
    assert (output_dir / "molecule_benchmark.png").is_file()
    assert (output_dir / "molecule_benchmark_plot_data.csv").is_file()

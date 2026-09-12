"""Tests for the standalone Branin benchmark plotting CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from statistics import fmean, stdev

import numpy as np
import pytest

from activelearning.utils.plotting import (
    AggregatedCheckpoint,
    MetricStatistics,
    RunCheckpoint,
    aggregate_checkpoints,
)
from scripts import plot_branin_benchmark as benchmark_plot

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "plot_branin_benchmark.py"


def make_observation(
    point: tuple[float, float],
    *,
    y: float = 0.0,
    fidelity: int | None = None,
) -> dict[str, object]:
    """Build one serialized observation payload."""

    return {
        "x": [float(point[0]), float(point[1])],
        "y": float(y),
        "fidelity": fidelity,
        "metadata": None,
    }


def write_seed_run(
    base_dir: Path,
    *,
    method: str,
    seed: int,
    initial_points: list[tuple[float, float]],
    rounds: list[dict[str, object]],
    extra_lines: list[str] | None = None,
) -> Path:
    """Write one synthetic seed directory with manifest and round history."""

    seed_dir = base_dir / method / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "initial_data": {
            "initial_observations": [
                make_observation(point) for point in initial_points
            ]
        }
    }
    (seed_dir / "run_manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    lines = [json.dumps(round_record) for round_record in rounds]
    if extra_lines:
        lines.extend(extra_lines)
    (seed_dir / "round_history.jsonl").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return seed_dir


def make_round(
    round_index: int,
    cumulative_cost: float,
    new_points: list[tuple[float, float]],
    *,
    y_values: list[float] | None = None,
) -> dict[str, object]:
    """Build one serialized round-history record."""

    if y_values is None:
        y_values = [0.0] * len(new_points)
    return {
        "round_index": round_index,
        "cumulative_cost": cumulative_cost,
        "valid_observations": [
            make_observation(point, y=y_value)
            for point, y_value in zip(new_points, y_values, strict=True)
        ],
    }


def test_initial_observations_are_included_in_first_checkpoint(tmp_path: Path) -> None:
    """Round-one scoring must include manifest initial observations."""

    mode_a, mode_b, mode_c = benchmark_plot.BRANIN_MODES
    seed_dir = write_seed_run(
        tmp_path,
        method="mf_gfn",
        seed=1,
        initial_points=[mode_a, mode_b],
        rounds=[make_round(1, 1.0, [mode_c])],
    )

    checkpoints = benchmark_plot.load_seed_checkpoints(
        seed_dir, method="mf_gfn", top_k=10
    )

    assert len(checkpoints) == 1
    assert checkpoints[0].observation_count == 3
    expected = fmean(
        benchmark_plot.rescore_at_highest_fidelity([mode_a, mode_b, mode_c])
    )
    assert checkpoints[0].metrics["mean_top_k"] == pytest.approx(expected)


def test_metrics_rescore_points_at_highest_fidelity(tmp_path: Path) -> None:
    """Recorded y-values must not replace high-fidelity Branin rescoring."""

    mode = benchmark_plot.BRANIN_MODES[0]
    far_point = (-5.0, 0.0)
    seed_dir = write_seed_run(
        tmp_path,
        method="mf_gfn",
        seed=2,
        initial_points=[mode],
        rounds=[make_round(1, 1.0, [far_point], y_values=[-998.0])],
    )
    manifest = json.loads((seed_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["initial_data"]["initial_observations"][0]["y"] = -999.0
    (seed_dir / "run_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )

    checkpoint = benchmark_plot.load_seed_checkpoints(
        seed_dir, method="mf_gfn", top_k=1
    )[0]
    rescored = benchmark_plot.rescore_at_highest_fidelity([mode, far_point])

    assert checkpoint.metrics["mean_top_k"] == pytest.approx(max(rescored))
    assert checkpoint.metrics["mean_top_k"] != pytest.approx(-998.0)
    assert checkpoint.metrics["mode_coverage"] == pytest.approx(
        benchmark_plot.mode_coverage_metric([mode, far_point], k=5)
    )


def test_mean_top_k_uses_all_available_points_when_fewer_than_k_exist(
    tmp_path: Path,
) -> None:
    """Requesting a large top-k should average the available rescored points."""

    points = [
        benchmark_plot.BRANIN_MODES[0],
        benchmark_plot.BRANIN_MODES[1],
        benchmark_plot.BRANIN_MODES[2],
        (-5.0, 0.0),
        (0.0, 5.0),
    ]
    seed_dir = write_seed_run(
        tmp_path,
        method="mf_gfn",
        seed=3,
        initial_points=points[:2],
        rounds=[make_round(1, 2.0, points[2:])],
    )

    checkpoint = benchmark_plot.load_seed_checkpoints(
        seed_dir, method="mf_gfn", top_k=50
    )[0]
    expected = fmean(benchmark_plot.rescore_at_highest_fidelity(points))

    assert checkpoint.observation_count == 5
    assert checkpoint.metrics["mean_top_k"] == pytest.approx(expected)


def test_mode_coverage_is_dominated_by_the_worst_covered_mode() -> None:
    """The max-KNN metric should track the least-covered Branin mode."""

    offsets = [(0.0, 0.0), (0.05, 0.0), (-0.05, 0.0)]
    good_points = [
        (mode[0] + dx, mode[1] + dy)
        for mode in benchmark_plot.BRANIN_MODES
        for dx, dy in offsets
    ]
    bad_points = [
        (mode[0] + dx, mode[1] + dy)
        for mode in benchmark_plot.BRANIN_MODES[:2]
        for dx, dy in offsets
    ]

    good_metric = benchmark_plot.mode_coverage_metric(good_points, k=3)
    bad_metric = benchmark_plot.mode_coverage_metric(bad_points, k=3)
    assert good_metric is not None
    assert bad_metric is not None

    missing_mode = np.asarray(benchmark_plot.BRANIN_MODES[2], dtype=float)
    distances = np.linalg.norm(
        np.asarray(bad_points, dtype=float) - missing_mode, axis=1
    )
    expected_worst_mode_average = float(np.mean(np.sort(distances)[:3]))

    assert bad_metric == pytest.approx(expected_worst_mode_average)
    assert good_metric < 0.06
    assert bad_metric > good_metric * 20


def test_mode_coverage_with_k_one_reduces_to_nearest_neighbor_distance() -> None:
    """K=1 should return the maximum nearest-neighbor distance across modes."""

    points = [
        (benchmark_plot.BRANIN_MODES[0][0] + 1.0, benchmark_plot.BRANIN_MODES[0][1]),
        (benchmark_plot.BRANIN_MODES[1][0] + 0.5, benchmark_plot.BRANIN_MODES[1][1]),
        (benchmark_plot.BRANIN_MODES[2][0] + 0.25, benchmark_plot.BRANIN_MODES[2][1]),
    ]

    metric = benchmark_plot.mode_coverage_metric(points, k=1)

    assert metric == pytest.approx(1.0)


def test_cross_seed_aggregation_uses_sample_standard_deviation() -> None:
    """Aggregation should use mean plus ddof=1 sample standard deviation."""

    rows = [
        RunCheckpoint(
            method="mf_gfn",
            seed=1,
            round_index=1,
            cumulative_cost=10.0,
            observations=(),
            metrics={"mean_top_k": 1.0, "mode_coverage": 4.0},
        ),
        RunCheckpoint(
            method="mf_gfn",
            seed=2,
            round_index=1,
            cumulative_cost=12.0,
            observations=(),
            metrics={"mean_top_k": 3.0, "mode_coverage": 2.0},
        ),
        RunCheckpoint(
            method="mf_gfn",
            seed=1,
            round_index=2,
            cumulative_cost=20.0,
            observations=(),
            metrics={"mean_top_k": 5.0, "mode_coverage": 1.0},
        ),
    ]

    aggregated = aggregate_checkpoints(rows)

    assert aggregated[0].cumulative_cost_mean == pytest.approx(11.0)
    assert aggregated[0].cumulative_cost_std == pytest.approx(stdev([10.0, 12.0]))
    assert aggregated[0].metrics["mean_top_k"].mean == pytest.approx(2.0)
    assert aggregated[0].metrics["mean_top_k"].std == pytest.approx(stdev([1.0, 3.0]))
    assert aggregated[0].metrics["mode_coverage"].mean == pytest.approx(3.0)
    assert aggregated[0].metrics["mode_coverage"].std == pytest.approx(
        stdev([4.0, 2.0])
    )
    assert aggregated[1].seed_count == 1
    assert aggregated[1].cumulative_cost_std == 0.0
    assert aggregated[1].metrics["mean_top_k"].std == 0.0
    assert aggregated[1].metrics["mode_coverage"].std == 0.0


@pytest.mark.parametrize("log_scale", [False, True])
def test_plotting_writes_png_for_linear_and_log_scales(
    tmp_path: Path,
    log_scale: bool,
) -> None:
    """The plotting helper should save figures for both x-axis scales."""

    aggregated_data = {
        "mf_gfn": [
            AggregatedCheckpoint(
                round_index=1,
                seed_count=2,
                cumulative_cost=MetricStatistics(mean=1.0, std=0.1),
                metrics={
                    "mean_top_k": MetricStatistics(mean=0.5, std=0.05),
                    "mode_coverage": MetricStatistics(mean=2.0, std=0.2),
                },
            ),
            AggregatedCheckpoint(
                round_index=2,
                seed_count=2,
                cumulative_cost=MetricStatistics(mean=2.0, std=0.1),
                metrics={
                    "mean_top_k": MetricStatistics(mean=0.7, std=0.04),
                    "mode_coverage": MetricStatistics(mean=1.5, std=0.15),
                },
            ),
        ]
    }
    output_path = benchmark_plot.output_path_for(
        tmp_path,
        metric="mean_top_k",
        scale="log" if log_scale else "linear",
    )

    benchmark_plot.plot_metric(
        aggregated_data,
        metric="mean_top_k",
        output_path=output_path,
        log_scale=log_scale,
    )

    assert output_path.is_file()


def test_loading_skips_missing_methods_and_malformed_jsonl_lines(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing methods and bad JSONL lines should warn but not abort loading."""

    write_seed_run(
        tmp_path,
        method="mf_gfn",
        seed=4,
        initial_points=[benchmark_plot.BRANIN_MODES[0]],
        rounds=[make_round(1, 1.0, [benchmark_plot.BRANIN_MODES[1]])],
        extra_lines=["not-json", ""],
    )
    (tmp_path / "random" / "seed_4").mkdir(parents=True, exist_ok=True)

    data = benchmark_plot.load_benchmark_data(
        tmp_path,
        methods=("mf_gfn", "random", "sf_low_fid"),
    )
    captured = capsys.readouterr()

    assert "mf_gfn" in data
    assert len(data["mf_gfn"]) == 1
    assert "Skipping malformed JSONL line" in captured.err
    assert "Skipping empty JSONL line" in captured.err
    assert "Missing round history" in captured.err
    assert "Skipping missing method directory" in captured.err


def test_cli_help_lists_metric_and_scale_options() -> None:
    """The standalone CLI should advertise its metric and scale controls."""

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--metric" in result.stdout
    assert "mean-top-k" in result.stdout
    assert "--scale" in result.stdout
    assert "--base-dir" in result.stdout


def test_cli_default_paths_are_relative_to_repository_root() -> None:
    """The CLI defaults should work regardless of the current working directory."""

    args = benchmark_plot.build_parser().parse_args([])

    assert args.base_dir == REPO_ROOT / "outputs" / "branin_benchmark"
    assert args.output_dir == REPO_ROOT / "plots"


def test_cli_creates_expected_metric_and_scale_outputs(tmp_path: Path) -> None:
    """The CLI should generate the canonical Branin benchmark figure names."""

    for method, seed, points in [
        ("mf_gfn", 1, [benchmark_plot.BRANIN_MODES[0], benchmark_plot.BRANIN_MODES[1]]),
        ("random", 2, [benchmark_plot.BRANIN_MODES[1], benchmark_plot.BRANIN_MODES[2]]),
    ]:
        write_seed_run(
            tmp_path,
            method=method,
            seed=seed,
            initial_points=[points[0]],
            rounds=[make_round(1, 1.0, [points[1]])],
        )

    output_dir = tmp_path / "plots"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--metric",
            "all",
            "--scale",
            "all",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_files = {
        "branin_benchmark_mean_top_k.png",
        "branin_benchmark_mean_top_k_log.png",
        "branin_benchmark_mode_coverage.png",
        "branin_benchmark_mode_coverage_log.png",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}

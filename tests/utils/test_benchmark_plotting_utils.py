"""Tests for reusable offline benchmark plotting utilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from activelearning.monitoring.run_writer import JSONLinesRunWriter, RoundRecord
from activelearning.utils.plotting import (
    AggregatedCheckpoint,
    MethodStyle,
    MetricStatistics,
    RunCheckpoint,
    aggregate_checkpoints,
    compute_mean_top_k,
    load_run_checkpoints,
    plot_metric,
)
from activelearning.utils.types import Candidate, Observation


def _round_record(
    *,
    round_index: int,
    selected_candidate: Candidate,
    observation: Observation,
    selected_cost: float,
    cumulative_cost: float,
    remaining_budget: float,
) -> RoundRecord:
    """Build a minimal round record for benchmark-reader fixtures."""
    return RoundRecord(
        round_index=round_index,
        observations_before=[],
        observations_after=[observation],
        sampled_candidates=[],
        selected_candidates=[selected_candidate],
        selected_costs=[selected_cost],
        queried_observations=[observation],
        valid_observations=[observation],
        round_budget=selected_cost,
        initial_budget=cumulative_cost + remaining_budget,
        cumulative_cost=cumulative_cost,
        remaining_budget=remaining_budget,
        metrics={},
        profiling={},
        diagnostics={},
    )


def test_compute_mean_top_k_accepts_numpy_arrays() -> None:
    """Top-k aggregation should support NumPy sequence inputs."""

    values = np.asarray([1.0, 3.0, 2.0])

    assert compute_mean_top_k(values, k=2) == pytest.approx(2.5)


def test_loading_reads_jsonlines_run_writer_artifacts(tmp_path: Path) -> None:
    """The generic loader should follow the JSONLinesRunWriter artifact schema."""

    writer = JSONLinesRunWriter(
        output_dir=tmp_path,
        metadata={"run": {"method": "demo", "seed": 7}},
    )
    writer.start_run(
        {
            "initial_data": {
                "initial_observations": [
                    Observation(x=[0.0, 1.0], y=0.25),
                ]
            }
        }
    )
    writer.record_round(
        _round_record(
            round_index=1,
            selected_candidate=Candidate(x=[1.0, 2.0]),
            observation=Observation(x=[1.0, 2.0], y=0.5),
            selected_cost=2.0,
            cumulative_cost=2.0,
            remaining_budget=3.0,
        )
    )
    writer.record_round(
        _round_record(
            round_index=2,
            selected_candidate=Candidate(x=[2.0, 3.0]),
            observation=Observation(x=[2.0, 3.0], y=0.75),
            selected_cost=1.0,
            cumulative_cost=3.0,
            remaining_budget=2.0,
        )
    )

    rows = load_run_checkpoints(
        tmp_path,
        method="demo",
        seed=7,
        metric_callbacks={
            "observation_count": lambda observations: float(len(observations)),
            "best_y": lambda observations: max(
                float(observation["y"]) for observation in observations
            ),
        },
    )

    assert [row.round_index for row in rows] == [1, 2]
    assert [row.observation_count for row in rows] == [2, 3]
    assert [row.metrics["observation_count"] for row in rows] == [2.0, 3.0]
    assert [row.metrics["best_y"] for row in rows] == [0.5, 0.75]


def test_loading_preserves_cumulative_opaque_observations(tmp_path: Path) -> None:
    """Metric callbacks should receive the manifest observations plus new ones."""

    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "initial_data": {
                    "initial_observations": [{"kind": "initial"}],
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "round_history.jsonl").write_text(
        json.dumps(
            {
                "round_index": 1,
                "cumulative_cost": 1.0,
                "valid_observations": [{"kind": "round-1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    received: list[tuple[object, ...]] = []

    def capture(observations: tuple[object, ...]) -> float:
        received.append(observations)
        return float(len(observations))

    rows = load_run_checkpoints(
        tmp_path,
        method="demo",
        seed="seed-a",
        metric_callbacks={"count": capture},
    )

    assert rows[0].observations == (
        {"kind": "initial"},
        {"kind": "round-1"},
    )
    assert received == [rows[0].observations]


def test_loading_warns_and_skips_malformed_artifacts(tmp_path: Path) -> None:
    """Malformed records should not discard valid JSONL checkpoints."""

    history_path = tmp_path / "round_history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                "not-json",
                "",
                json.dumps(
                    {
                        "round_index": 1,
                        "cumulative_cost": 1.0,
                        "valid_observations": [],
                    }
                ),
                json.dumps(
                    {
                        "round_index": 2,
                        "cumulative_cost": 2.0,
                        "valid_observations": "not-a-list",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    warnings: list[str] = []

    rows = load_run_checkpoints(
        tmp_path,
        method="demo",
        seed=1,
        metric_callbacks={},
        warn_fn=warnings.append,
    )

    assert len(rows) == 1
    assert rows[0].round_index == 1
    assert any("Missing run manifest" in message for message in warnings)
    assert any("malformed JSONL line" in message for message in warnings)
    assert any("empty JSONL line" in message for message in warnings)
    assert any("malformed valid_observations" in message for message in warnings)


def test_aggregation_supports_arbitrary_metrics_and_sample_std() -> None:
    """Named metric mappings should aggregate independently by round."""

    rows = [
        RunCheckpoint(
            method="demo",
            seed=1,
            round_index=1,
            cumulative_cost=10.0,
            observations=(),
            metrics={"score": 1.0, "coverage": 4.0},
        ),
        RunCheckpoint(
            method="demo",
            seed=2,
            round_index=1,
            cumulative_cost=12.0,
            observations=(),
            metrics={"score": 3.0, "coverage": None},
        ),
        RunCheckpoint(
            method="demo",
            seed=1,
            round_index=2,
            cumulative_cost=20.0,
            observations=(),
            metrics={"score": 5.0},
        ),
    ]

    aggregated = aggregate_checkpoints(rows)

    assert aggregated[0].cumulative_cost_mean == pytest.approx(11.0)
    assert aggregated[0].cumulative_cost_std == pytest.approx(1.41421356237)
    assert aggregated[0].metrics["score"] == MetricStatistics(
        mean=2.0,
        std=pytest.approx(1.41421356237),
    )
    assert aggregated[0].metrics["coverage"] == MetricStatistics(mean=4.0, std=0.0)
    assert aggregated[1].metrics["score"] == MetricStatistics(mean=5.0, std=0.0)


@pytest.mark.parametrize("log_scale", [False, True])
def test_generic_plotting_writes_png_for_both_scales(
    tmp_path: Path,
    log_scale: bool,
) -> None:
    """The generic renderer should support linear and log cost axes."""

    aggregated_data = {
        "demo": [
            AggregatedCheckpoint(
                round_index=1,
                seed_count=2,
                cumulative_cost=MetricStatistics(mean=1.0, std=0.1),
                metrics={
                    "score": MetricStatistics(mean=0.5, std=0.05),
                },
            ),
            AggregatedCheckpoint(
                round_index=2,
                seed_count=2,
                cumulative_cost=MetricStatistics(mean=2.0, std=0.1),
                metrics={
                    "score": MetricStatistics(mean=0.7, std=0.04),
                },
            ),
        ]
    }
    output_path = tmp_path / ("plot_log.png" if log_scale else "plot.png")

    plot_metric(
        aggregated_data,
        metric="score",
        output_path=output_path,
        method_order=("demo",),
        method_styles={"demo": MethodStyle("Demo", "#1f77b4")},
        y_label="Score",
        title="Demo benchmark",
        log_scale=log_scale,
    )

    assert output_path.is_file()


def test_generic_plotting_does_not_mutate_global_spine_settings(
    tmp_path: Path,
) -> None:
    """The renderer should apply spine styling only to its local axis."""

    original_top = plt.rcParams["axes.spines.top"]
    original_right = plt.rcParams["axes.spines.right"]
    aggregated_data = {
        "demo": [
            AggregatedCheckpoint(
                round_index=1,
                seed_count=1,
                cumulative_cost=MetricStatistics(mean=1.0, std=0.0),
                metrics={"score": MetricStatistics(mean=0.5, std=0.0)},
            )
        ]
    }

    plot_metric(
        aggregated_data,
        metric="score",
        output_path=tmp_path / "plot.png",
        method_order=("demo",),
        method_styles={"demo": MethodStyle("Demo", "#1f77b4")},
    )

    assert plt.rcParams["axes.spines.top"] == original_top
    assert plt.rcParams["axes.spines.right"] == original_right

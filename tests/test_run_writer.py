import csv
import json
import math
from dataclasses import replace
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pytest

from activelearning.monitoring.run_writer import (
    JSONLinesRunWriter,
    RoundRecord,
    _resolve_method,
)
from activelearning.utils.types import Candidate, Observation


def _round_record(
    *,
    round_index: int,
    sampled_candidates: list[Candidate],
    selected_candidates: list[Candidate],
    selected_costs: list[float],
    observations: list[Observation],
    cumulative_cost: float,
    remaining_budget: float,
    metrics: dict[str, int | float] | None = None,
    profiling: dict[str, float] | None = None,
) -> RoundRecord:
    """Build a minimal completed round record for persistence tests."""
    return RoundRecord(
        round_index=round_index,
        observations_before=[],
        observations_after=observations,
        sampled_candidates=sampled_candidates,
        selected_candidates=selected_candidates,
        selected_costs=selected_costs,
        queried_observations=observations,
        valid_observations=observations,
        round_budget=0.0,
        initial_budget=0.0,
        cumulative_cost=cumulative_cost,
        remaining_budget=remaining_budget,
        metrics=metrics or {},
        profiling=profiling or {},
        diagnostics={},
    )


def test_resolve_method_uses_generic_metadata_before_output_path(tmp_path) -> None:
    """Method resolution should not depend on an experiment-specific config key."""
    output_dir = tmp_path / "runs" / "path_method" / "seed_0"
    manifest = {
        "config": {"branin_benchmark": {"method": "config_method"}},
        "run": {"method": "metadata_method"},
    }

    assert _resolve_method(manifest, output_dir) == "metadata_method"
    assert _resolve_method({"config": manifest["config"]}, output_dir) == "path_method"
    assert _resolve_method({}, Path(tmp_path.anchor)) is None


def test_run_writer_persists_manifest_round_history_and_summary(tmp_path) -> None:
    """The JSON-lines writer should persist all core run artifacts."""
    run_writer = JSONLinesRunWriter(
        output_dir=tmp_path,
        metadata={
            "config": {"runtime": {"seed": 7}},
            "run": {"method": "dummy_method"},
        },
    )

    initial_observations = [Observation(x=-1, y=0.5, fidelity=0)]
    run_writer.start_run(
        {
            "initial_budget": 4.0,
            "initial_data": {"initial_observations": initial_observations},
        }
    )
    run_writer.record_round(
        _round_record(
            round_index=1,
            sampled_candidates=[Candidate(1), Candidate(2)],
            selected_candidates=[Candidate(2)],
            selected_costs=[1.5],
            observations=[Observation(x=2, y=1.5)],
            cumulative_cost=1.5,
            remaining_budget=2.5,
            metrics={
                "active_learning/cost/round": 1.5,
                "active_learning/observations/new": 1,
            },
            profiling={"profiling/oracle/query_s": 0.25},
        )
    )
    run_writer.end_run(
        {
            "num_rounds": 1,
            "total_cost": 1.5,
            "budget_remaining": 2.5,
        }
    )

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    round_records = [
        json.loads(line)
        for line in (tmp_path / "round_history.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["initial_budget"] == 4.0
    assert manifest["initial_data"]["initial_observations"] == [
        {"x": -1, "y": 0.5, "fidelity": 0, "metadata": None}
    ]
    assert manifest["run"]["method"] == "dummy_method"
    assert summary == {
        "num_rounds": 1,
        "total_cost": 1.5,
        "budget_remaining": 2.5,
    }
    assert round_records == [
        {
            "artifacts": {},
            "diagnostics": {},
            "initial_budget": 0.0,
            "queried_observations": [
                {"x": 2, "y": 1.5, "fidelity": 0, "metadata": None}
            ],
            "round_budget": 0.0,
            "round_index": 1,
            "sampled_candidates": [
                {"x": 1, "fidelity": 0, "metadata": None},
                {"x": 2, "fidelity": 0, "metadata": None},
            ],
            "selected_candidates": [{"x": 2, "fidelity": 0, "metadata": None}],
            "selected_costs": [1.5],
            "round_cost": 1.5,
            "cumulative_cost": 1.5,
            "remaining_budget": 2.5,
            "valid_observations": [{"x": 2, "y": 1.5, "fidelity": 0, "metadata": None}],
            "metrics": {
                "active_learning/cost/round": 1.5,
                "active_learning/observations/new": 1,
            },
            "profiling": {"profiling/oracle/query_s": 0.25},
        }
    ]


def test_run_writer_records_best_objective_trajectory(tmp_path) -> None:
    """The experiment CSV should track the best objective value after each round."""
    run_writer = JSONLinesRunWriter(
        output_dir=tmp_path,
        metadata={
            "config": {"runtime": {"seed": 11}},
            "run": {"method": "sf_gfn"},
        },
        write_samples=False,
        write_config=False,
    )
    run_writer.start_run({})
    run_writer.record_round(
        _round_record(
            round_index=1,
            sampled_candidates=[],
            selected_candidates=[Candidate(1)],
            selected_costs=[1.0],
            observations=[Observation(x=1, y=1.5)],
            cumulative_cost=1.0,
            remaining_budget=2.0,
        )
    )
    run_writer.record_round(
        _round_record(
            round_index=2,
            sampled_candidates=[],
            selected_candidates=[Candidate(2)],
            selected_costs=[1.0],
            observations=[Observation(x=2, y=0.25)],
            cumulative_cost=2.0,
            remaining_budget=1.0,
        )
    )
    run_writer.record_round(
        _round_record(
            round_index=3,
            sampled_candidates=[],
            selected_candidates=[Candidate(3)],
            selected_costs=[1.0],
            observations=[Observation(x=3, y=2.0)],
            cumulative_cost=3.0,
            remaining_budget=0.0,
        )
    )
    run_writer.end_run({"num_rounds": 3, "total_cost": 3.0, "budget_remaining": 0.0})

    experiment_log_rows = list(
        csv.DictReader((tmp_path / "experiment_log.csv").open(encoding="utf-8"))
    )

    assert experiment_log_rows == [
        {
            "index": "0",
            "method": "sf_gfn",
            "seed": "11",
            "objective value": "1.5",
            "cost": "1.0",
            "round": "0",
        },
        {
            "index": "1",
            "method": "sf_gfn",
            "seed": "11",
            "objective value": "1.5",
            "cost": "2.0",
            "round": "1",
        },
        {
            "index": "2",
            "method": "sf_gfn",
            "seed": "11",
            "objective value": "2.0",
            "cost": "3.0",
            "round": "2",
        },
    ]


def test_write_config_false_preserves_reconstruction_manifest(tmp_path) -> None:
    """Initial observations remain available when config persistence is disabled."""
    run_writer = JSONLinesRunWriter(
        output_dir=tmp_path,
        metadata={"config": {"runtime": {"seed": 7}}},
        write_config=False,
    )

    run_writer.start_run(
        {
            "initial_budget": 2.0,
            "initial_data": {
                "initial_observations": [Observation(x=1, y=0.5)],
            },
        }
    )

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest == {
        "initial_budget": 2.0,
        "initial_data": {
            "initial_observations": [
                {"x": 1, "y": 0.5, "fidelity": 0, "metadata": None}
            ]
        },
    }


def test_run_writer_ignores_non_finite_objectives_in_best_trajectory(tmp_path) -> None:
    """The experiment CSV should ignore NaN objectives when tracking best-so-far."""
    run_writer = JSONLinesRunWriter(
        output_dir=tmp_path,
        metadata={
            "config": {"runtime": {"seed": 13}},
            "run": {"method": "sf_gfn"},
        },
        write_samples=False,
        write_config=False,
    )
    run_writer.start_run({})
    run_writer.record_round(
        _round_record(
            round_index=1,
            sampled_candidates=[],
            selected_candidates=[Candidate(1), Candidate(2)],
            selected_costs=[1.0, 1.0],
            observations=[Observation(x=1, y=math.nan), Observation(x=2, y=1.5)],
            cumulative_cost=2.0,
            remaining_budget=2.0,
        )
    )
    run_writer.record_round(
        _round_record(
            round_index=2,
            sampled_candidates=[],
            selected_candidates=[Candidate(3), Candidate(4)],
            selected_costs=[1.0, 1.0],
            observations=[Observation(x=3, y=math.inf), Observation(x=4, y=0.25)],
            cumulative_cost=4.0,
            remaining_budget=0.0,
        )
    )
    run_writer.end_run({"num_rounds": 2, "total_cost": 4.0, "budget_remaining": 0.0})

    experiment_log_rows = list(
        csv.DictReader((tmp_path / "experiment_log.csv").open(encoding="utf-8"))
    )

    assert experiment_log_rows == [
        {
            "index": "0",
            "method": "sf_gfn",
            "seed": "13",
            "objective value": "1.5",
            "cost": "2.0",
            "round": "0",
        },
        {
            "index": "1",
            "method": "sf_gfn",
            "seed": "13",
            "objective value": "1.5",
            "cost": "4.0",
            "round": "1",
        },
    ]


def test_run_writer_omits_sample_fields_and_handles_empty_observations(
    tmp_path,
) -> None:
    """Writer output should stay valid when a filtered round adds no observations."""
    run_writer = JSONLinesRunWriter(
        output_dir=tmp_path,
        metadata={"run": {"method": "filtered"}},
        write_samples=False,
        write_config=False,
    )

    run_writer.start_run({})
    run_writer.record_round(
        _round_record(
            round_index=1,
            sampled_candidates=[Candidate(1)],
            selected_candidates=[Candidate(1)],
            selected_costs=[1.0],
            observations=[],
            cumulative_cost=1.0,
            remaining_budget=0.0,
        )
    )
    run_writer.end_run({"num_rounds": 1, "total_cost": 1.0, "budget_remaining": 0.0})

    round_record = json.loads(
        (tmp_path / "round_history.jsonl").read_text(encoding="utf-8").strip()
    )
    experiment_log_rows = list(
        csv.DictReader((tmp_path / "experiment_log.csv").open(encoding="utf-8"))
    )

    assert "sampled_candidates" not in round_record
    assert round_record["valid_observations"] == []
    assert experiment_log_rows == [
        {
            "index": "0",
            "method": "filtered",
            "seed": "",
            "objective value": "",
            "cost": "1.0",
            "round": "0",
        }
    ]


def test_run_writer_requires_startup(tmp_path) -> None:
    """Rounds cannot be recorded before the writer is started."""
    run_writer = JSONLinesRunWriter(output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="Call start_run"):
        run_writer.record_round(
            _round_record(
                round_index=1,
                sampled_candidates=[],
                selected_candidates=[],
                selected_costs=[],
                observations=[],
                cumulative_cost=0.0,
                remaining_budget=1.0,
            )
        )

    run_writer.start_run({})
    run_writer.record_round(
        _round_record(
            round_index=1,
            sampled_candidates=[Candidate(1)],
            selected_candidates=[Candidate(1)],
            selected_costs=[1.0],
            observations=[],
            cumulative_cost=1.0,
            remaining_budget=0.0,
        )
    )


def test_run_writer_persists_diagnostic_figure_with_round_scoped_path(tmp_path) -> None:
    """Diagnostic figures should be saved under their component and round path."""
    run_writer = JSONLinesRunWriter(output_dir=tmp_path)
    figure = Figure()
    figure.add_subplot(1, 1, 1).plot([0.0, 1.0], [0.0, 1.0])
    run_writer.start_run({})

    run_writer.record_round(
        _round_record(
            round_index=3,
            sampled_candidates=[],
            selected_candidates=[],
            selected_costs=[],
            observations=[],
            cumulative_cost=0.0,
            remaining_budget=1.0,
        ),
        {"oracle/branin/query_landscape": figure},
    )

    payload = json.loads(
        (tmp_path / "round_history.jsonl").read_text(encoding="utf-8").strip()
    )
    relative_path = "artifacts/oracle/branin/round_0003/query_landscape.png"
    assert payload["artifacts"] == {"oracle/branin/query_landscape": relative_path}
    assert (tmp_path / relative_path).is_file()
    plt.close(figure)


def test_run_writer_rejects_unqualified_diagnostic_keys(tmp_path) -> None:
    """Persisted diagnostics must use the same namespaces as live metrics."""
    run_writer = JSONLinesRunWriter(output_dir=tmp_path)
    run_writer.start_run({})
    record = _round_record(
        round_index=1,
        sampled_candidates=[],
        selected_candidates=[],
        selected_costs=[],
        observations=[],
        cumulative_cost=0.0,
        remaining_budget=1.0,
    )

    with pytest.raises(ValueError, match="component-qualified namespace"):
        run_writer.record_round(replace(record, diagnostics={"loss": 0.5}))

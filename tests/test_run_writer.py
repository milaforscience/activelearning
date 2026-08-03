import csv
import json
import math

from activelearning.run_writer import JSONLinesRunWriter
from activelearning.utils.types import Candidate, Observation


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
        round_index=1,
        sampled_candidates=[Candidate(1), Candidate(2)],
        sampled_scores=[0.1, 0.2],
        selected_candidates=[Candidate(2)],
        selected_scores=[0.2],
        selected_costs=[1.5],
        observations=[Observation(x=2, y=1.5)],
        cumulative_cost=1.5,
        remaining_budget=2.5,
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
            "round": 1,
            "round_index": 1,
            "sampled_candidates": [
                {"x": 1, "fidelity": 0, "metadata": None},
                {"x": 2, "fidelity": 0, "metadata": None},
            ],
            "sampled_scores": [0.1, 0.2],
            "selected_candidates": [{"x": 2, "fidelity": 0, "metadata": None}],
            "selected_scores": [0.2],
            "selected_costs": [1.5],
            "observations": [{"x": 2, "y": 1.5, "fidelity": 0, "metadata": None}],
            "new_observations": [{"x": 2, "y": 1.5, "fidelity": 0, "metadata": None}],
            "round_cost": 1.5,
            "cumulative_cost": 1.5,
            "remaining_budget": 2.5,
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
        round_index=1,
        sampled_candidates=[],
        sampled_scores=[],
        selected_candidates=[Candidate(1)],
        selected_scores=[0.0],
        selected_costs=[1.0],
        observations=[Observation(x=1, y=1.5)],
        cumulative_cost=1.0,
        remaining_budget=2.0,
    )
    run_writer.record_round(
        round_index=2,
        sampled_candidates=[],
        sampled_scores=[],
        selected_candidates=[Candidate(2)],
        selected_scores=[0.0],
        selected_costs=[1.0],
        observations=[Observation(x=2, y=0.25)],
        cumulative_cost=2.0,
        remaining_budget=1.0,
    )
    run_writer.record_round(
        round_index=3,
        sampled_candidates=[],
        sampled_scores=[],
        selected_candidates=[Candidate(3)],
        selected_scores=[0.0],
        selected_costs=[1.0],
        observations=[Observation(x=3, y=2.0)],
        cumulative_cost=3.0,
        remaining_budget=0.0,
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
        round_index=1,
        sampled_candidates=[],
        sampled_scores=[],
        selected_candidates=[Candidate(1), Candidate(2)],
        selected_scores=[0.0, 0.0],
        selected_costs=[1.0, 1.0],
        observations=[Observation(x=1, y=math.nan), Observation(x=2, y=1.5)],
        cumulative_cost=2.0,
        remaining_budget=2.0,
    )
    run_writer.record_round(
        round_index=2,
        sampled_candidates=[],
        sampled_scores=[],
        selected_candidates=[Candidate(3), Candidate(4)],
        selected_scores=[0.0, 0.0],
        selected_costs=[1.0, 1.0],
        observations=[Observation(x=3, y=math.inf), Observation(x=4, y=0.25)],
        cumulative_cost=4.0,
        remaining_budget=0.0,
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
        round_index=1,
        sampled_candidates=[Candidate(1)],
        sampled_scores=[0.0],
        selected_candidates=[Candidate(1)],
        selected_scores=[0.0],
        selected_costs=[1.0],
        observations=[],
        cumulative_cost=1.0,
        remaining_budget=0.0,
    )
    run_writer.end_run({"num_rounds": 1, "total_cost": 1.0, "budget_remaining": 0.0})

    round_record = json.loads(
        (tmp_path / "round_history.jsonl").read_text(encoding="utf-8").strip()
    )
    experiment_log_rows = list(
        csv.DictReader((tmp_path / "experiment_log.csv").open(encoding="utf-8"))
    )

    assert "sampled_candidates" not in round_record
    assert "sampled_scores" not in round_record
    assert round_record["observations"] == []
    assert round_record["new_observations"] == []
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

from __future__ import annotations

import json
from pathlib import Path
import shutil
from statistics import fmean
from typing import Any, Iterator, Sequence

import pytest
import selfies as sf
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from scripts.reproduce_paper_metrics import (
    MoleculeRunMetrics,
    ROUND_HISTORY_FILENAME,
    RUN_MANIFEST_FILENAME,
    SyntheticRunMetrics,
    collect_run_metrics,
    compute_run_metrics,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reproduce_paper"
TOTAL_BUDGET_BY_TASK = {
    "branin": 300.0,
    "hartmann": 100.0,
    "molecules_ea": 1260.0,
    "molecules_ip": 1260.0,
}
TOP_K_BY_TASK = {
    "branin": 50,
    "hartmann": 10,
    "molecules_ea": 100,
    "molecules_ip": 100,
}
ORACLE_COSTS_BY_TASK = {
    "branin": {1: 0.01, 2: 0.1, 3: 1.0},
    "hartmann": {1: 0.125, 2: 0.25, 3: 1.0},
    "molecules_ea": {1: 1.0, 2: 3.5, 3: 7.0},
    "molecules_ip": {1: 1.0, 2: 3.5, 3: 7.0},
}
ORACLE_TYPE_BY_TASK = {
    "branin": "BraninOracle",
    "hartmann": "Hartmann6DOracle",
    "molecules_ea": "XTBIPEAOracle",
    "molecules_ip": "XTBIPEAOracle",
}


@pytest.fixture
def metrics_output_root() -> Iterator[Path]:
    """Provide a repository-local output directory for metrics tests."""

    root = FIXTURE_ROOT / "metrics_runtime_outputs"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    yield root
    if root.exists():
        shutil.rmtree(root)


def test_compute_run_metrics_builds_branin_high_fidelity_trace(
    metrics_output_root: Path,
) -> None:
    """Synthetic metrics should expose Branin top-k scores and HF trace rows."""

    run_directory = _write_recorded_run(
        output_root=metrics_output_root,
        task_group="synthetic",
        task="branin",
        method="mf_gfn",
        seed=1,
        initial_mode="multi_fidelity",
        initial_observations=[
            {
                "x": [-5.0, 0.0],
                "y": 10.0,
                "fidelity": 3,
                "metadata": {"identifier": "hf-a"},
            },
            {"x": [0.0, 0.0], "y": 20.0, "fidelity": 1},
            {
                "x": [10.0, 15.0],
                "y": 5.0,
                "fidelity": 3,
                "metadata": {"identifier": "hf-b"},
            },
        ],
        round_history=[
            {
                "round_index": 1,
                "cumulative_cost": 1.1,
                "new_observations": [
                    {"x": [2.0, 3.0], "y": 8.0, "fidelity": 3},
                    {"x": [4.0, 5.0], "y": 30.0, "fidelity": 2},
                ],
            },
            {
                "round_index": 2,
                "cumulative_cost": 2.1,
                "new_observations": [
                    {"x": [6.0, 7.0], "y": 2.0, "fidelity": 3},
                    {"x": [8.0, 9.0], "y": 12.0, "fidelity": 3},
                ],
            },
        ],
    )

    metrics = compute_run_metrics(run_directory)

    assert isinstance(metrics, SyntheticRunMetrics)
    assert [row.round_index for row in metrics.checkpoint_rows] == [0, 1, 2]
    assert [row.high_fidelity_observation_count for row in metrics.checkpoint_rows] == [
        2,
        3,
        5,
    ]
    assert [row.top_k for row in metrics.checkpoint_rows] == [50, 50, 50]
    assert [row.top_k_observation_count for row in metrics.checkpoint_rows] == [2, 3, 5]
    assert [row.mean_top_k_score for row in metrics.checkpoint_rows] == pytest.approx(
        [-7.5, (-5.0 - 8.0 - 10.0) / 3.0, (-2.0 - 5.0 - 8.0 - 10.0 - 12.0) / 5.0]
    )
    assert metrics.checkpoint_rows[
        -1
    ].budget_fraction_of_total_active_learning_budget == pytest.approx(2.1 / 300.0)

    assert [row.cumulative_budget for row in metrics.high_fidelity_trace_rows] == [
        0.0,
        0.0,
        1.1,
        2.1,
        2.1,
    ]
    assert [row.target_value for row in metrics.high_fidelity_trace_rows] == [
        10.0,
        5.0,
        8.0,
        2.0,
        12.0,
    ]
    assert [row.identifier for row in metrics.high_fidelity_trace_rows[:2]] == [
        "hf-a",
        "hf-b",
    ]


def test_compute_run_metrics_treats_single_fidelity_hartmann_rows_as_high_fidelity(
    metrics_output_root: Path,
) -> None:
    """Synthetic SF-GFN traces should map stripped fidelities back to the top oracle."""

    run_directory = _write_recorded_run(
        output_root=metrics_output_root,
        task_group="synthetic",
        task="hartmann",
        method="sf_gfn",
        seed=7,
        initial_mode="single_fidelity",
        initial_observations=[
            {"x": [0.0] * 6, "y": 1.0, "fidelity": None},
            {"x": [0.1] * 6, "y": 2.0, "fidelity": None},
        ],
        round_history=[
            {
                "round_index": 1,
                "cumulative_cost": 1.0,
                "new_observations": [
                    {"x": [0.2] * 6, "y": 3.0, "fidelity": None},
                ],
            }
        ],
    )

    metrics = compute_run_metrics(run_directory)

    assert isinstance(metrics, SyntheticRunMetrics)
    assert [row.fidelity for row in metrics.high_fidelity_trace_rows] == [3, 3, 3]
    assert [row.mean_top_k_score for row in metrics.checkpoint_rows] == pytest.approx(
        [1.5, 2.0]
    )


def test_compute_run_metrics_builds_molecule_ip_top100_and_tanimoto_metrics(
    metrics_output_root: Path,
) -> None:
    """Molecule metrics should expose score, energy, budget fraction, and diversity."""

    run_directory = _write_recorded_run(
        output_root=metrics_output_root,
        task_group="molecules",
        task="molecules_ip",
        method="mf_gfn",
        seed=9,
        initial_mode="multi_fidelity",
        initial_observations=[
            {"x": "[C]", "y": -5.0, "fidelity": 3, "metadata": {"identifier": "mol-c"}},
            {"x": "[O]", "y": -4.0, "fidelity": 3, "metadata": {"identifier": "mol-o"}},
            {"x": "[N]", "y": -6.0, "fidelity": 3, "metadata": {"identifier": "mol-n"}},
            {"x": "[F]", "y": -7.0, "fidelity": 1},
        ],
        round_history=[
            {
                "round_index": 1,
                "cumulative_cost": 21.0,
                "new_observations": [
                    {
                        "x": "[C][O]",
                        "y": -3.0,
                        "fidelity": 3,
                        "metadata": {"identifier": "mol-co"},
                    },
                    {
                        "x": "[C][N]",
                        "y": -2.0,
                        "fidelity": 3,
                        "metadata": {"identifier": "mol-cn"},
                    },
                    {"x": "[C][F]", "y": -8.0, "fidelity": 2},
                ],
            }
        ],
    )

    metrics = compute_run_metrics(run_directory)

    assert isinstance(metrics, MoleculeRunMetrics)
    assert [row.round_index for row in metrics.checkpoint_rows] == [0, 1]
    assert [row.high_fidelity_observation_count for row in metrics.checkpoint_rows] == [
        3,
        5,
    ]
    assert [row.mean_top_k_score for row in metrics.checkpoint_rows] == pytest.approx(
        [-5.0, -4.0]
    )
    assert [row.mean_top_k_energy for row in metrics.checkpoint_rows] == pytest.approx(
        [-5.0, -4.0]
    )
    assert metrics.checkpoint_rows[
        1
    ].budget_fraction_of_total_sf_gfn_budget == pytest.approx(21.0 / 1260.0)
    assert metrics.checkpoint_rows[0].mean_pairwise_tanimoto_distance == pytest.approx(
        _expected_tanimoto_distance(["[C]", "[O]", "[N]"])
    )
    assert metrics.checkpoint_rows[1].mean_pairwise_tanimoto_distance == pytest.approx(
        _expected_tanimoto_distance(["[C]", "[O]", "[N]", "[C][O]", "[C][N]"])
    )
    assert [
        row.paper_score for row in metrics.high_fidelity_trace_rows
    ] == pytest.approx([-5.0, -4.0, -6.0, -3.0, -2.0])


def test_collect_run_metrics_flattens_cross_run_rows(metrics_output_root: Path) -> None:
    """Metrics catalog should expose flattened rows for later cross-seed aggregation."""

    _write_recorded_run(
        output_root=metrics_output_root,
        task_group="synthetic",
        task="branin",
        method="mf_gfn",
        seed=1,
        initial_mode="multi_fidelity",
        initial_observations=[{"x": [0.0, 0.0], "y": 1.0, "fidelity": 3}],
        round_history=[],
    )
    _write_recorded_run(
        output_root=metrics_output_root,
        task_group="molecules",
        task="molecules_ea",
        method="sf_gfn",
        seed=2,
        initial_mode="single_fidelity",
        initial_observations=[{"x": "[C]", "y": 1.0, "fidelity": None}],
        round_history=[],
    )

    catalog = collect_run_metrics(metrics_output_root)
    payload = catalog.to_dict()

    assert len(catalog.synthetic_runs) == 1
    assert len(catalog.molecule_runs) == 1
    assert len(catalog.synthetic_checkpoint_rows()) == 1
    assert len(catalog.molecule_checkpoint_rows()) == 1
    assert len(payload["synthetic_high_fidelity_rows"]) == 1
    assert len(payload["molecule_high_fidelity_rows"]) == 1
    assert payload["schema_version"] == 1


def test_compute_run_metrics_can_infer_identity_from_layered_config_paths(
    metrics_output_root: Path,
) -> None:
    """Metrics should infer task and method from a checked-in runnable config path."""

    run_directory = _write_recorded_run(
        output_root=metrics_output_root,
        task_group="synthetic",
        task="branin",
        method="mf_gfn",
        seed=7,
        initial_mode="multi_fidelity",
        initial_observations=[{"x": [-5.0, 0.0], "y": 10.0, "fidelity": 3}],
        round_history=[
            {
                "round_index": 1,
                "cumulative_cost": 0.1,
                "new_observations": [{"x": [2.0, 3.0], "y": 8.0, "fidelity": 3}],
            }
        ],
        include_run_metadata=False,
    )

    metrics = compute_run_metrics(run_directory)

    assert metrics.run.task_group == "synthetic"
    assert metrics.run.task == "branin"
    assert metrics.run.method == "mf_gfn"
    assert metrics.run.seed == 7
    assert metrics.run.initial_data_mode == "multi_fidelity"
    assert metrics.run.top_k == 50
    assert metrics.run.negate_score is True


def _write_recorded_run(
    *,
    output_root: Path,
    task_group: str,
    task: str,
    method: str,
    seed: int,
    initial_mode: str,
    initial_observations: Sequence[dict[str, Any]],
    round_history: Sequence[dict[str, Any]],
    include_run_metadata: bool = True,
) -> Path:
    """Write a minimal recorded run directory for metrics regression tests."""

    run_directory = output_root / task / method / f"seed_{seed:04d}"
    run_directory.mkdir(parents=True, exist_ok=True)
    config_initial_data: dict[str, Any] = {"path": f"data/{task}/{initial_mode}.csv"}
    if initial_mode == "multi_fidelity":
        config_initial_data["fidelity_column"] = "fidelity"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "config": {
            "runtime": {"seed": seed},
            "dataset": {"initial_data": config_initial_data},
            "budget": {"available_budget": TOTAL_BUDGET_BY_TASK[task]},
            "oracle": {
                "type": ORACLE_TYPE_BY_TASK[task],
                "fidelity_costs": ORACLE_COSTS_BY_TASK[task],
            },
            "reproduce_paper": {
                "task_group": task_group,
                "top_k": TOP_K_BY_TASK[task],
                "negate_score": task == "branin",
                **(
                    {"molecule_representation": "selfies"}
                    if task_group == "molecules"
                    else {}
                ),
            },
        },
        "cli": {
            "config_files": _layered_config_files(
                task_group=task_group, task=task, method=method
            ),
            "config_overrides": [],
        },
        "initial_data": {
            "mode": initial_mode,
            "source": "fixture",
            "notes": [],
            "source_paths": {},
            "initial_observation_count": len(initial_observations),
            "initial_fidelity_counts": {},
            "initial_observations": list(initial_observations),
            "test_observation_count": 0,
            "test_fidelity_counts": {},
            "test_observations": [],
        },
        "artifacts": {"round_history": ROUND_HISTORY_FILENAME},
        "summary": {"status": "completed"},
    }
    if include_run_metadata:
        manifest["run"] = {
            "task_group": task_group,
            "task": task,
            "method": method,
            "seed": seed,
        }

    (run_directory / RUN_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_directory / ROUND_HISTORY_FILENAME).write_text(
        "".join(
            json.dumps(round_payload, sort_keys=True) + "\n"
            for round_payload in round_history
        ),
        encoding="utf-8",
    )
    return run_directory


def _layered_config_files(*, task_group: str, task: str, method: str) -> list[str]:
    """Return the runnable config path recorded for one reproduction run."""

    return [
        f"scripts/configs/reproduce_paper/{task_group}/{task}/{method}.yaml",
    ]


def _expected_tanimoto_distance(molecules: Sequence[str]) -> float:
    """Compute the expected mean Tanimoto distance for a small SELFIES set."""

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for molecule in molecules:
        smiles = sf.decoder(molecule)
        fingerprints.append(generator.GetFingerprint(Chem.MolFromSmiles(smiles)))

    distances = []
    for left_index, left in enumerate(fingerprints):
        for right in fingerprints[left_index + 1 :]:
            distances.append(1.0 - float(DataStructs.TanimotoSimilarity(left, right)))
    return fmean(distances)

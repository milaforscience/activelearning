from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from statistics import stdev
from typing import Any, Iterator, Sequence

import pytest
import selfies as sf
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from scripts.plot_reproduce_paper import main as plot_reproduce_paper_main
from scripts.reproduce_paper_metrics import (
    ROUND_HISTORY_FILENAME,
    RUN_MANIFEST_FILENAME,
    collect_run_metrics,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reproduce_paper"
PAPER_METHODS: tuple[str, ...] = (
    "mf_gfn",
    "random_fid_gfn",
    "sf_gfn",
    "random",
)
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
def plotting_output_root() -> Iterator[Path]:
    """Provide a repository-local output directory for plotting tests."""

    root = FIXTURE_ROOT / "plotting_runtime_outputs"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    yield root
    if root.exists():
        shutil.rmtree(root)


def test_plot_synthetic_results_writes_figure_and_plotting_table(
    plotting_output_root: Path,
) -> None:
    """Synthetic plotting should save a figure and the exact aggregated CSV."""

    run_root = plotting_output_root / "synthetic_runs"
    _write_synthetic_runs(run_root)

    metrics_json_path = plotting_output_root / "synthetic_metrics.json"
    metrics_json_path.write_text(
        json.dumps(collect_run_metrics(run_root).to_dict(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    output_dir = plotting_output_root / "synthetic_plots"
    plot_reproduce_paper_main(
        [
            "--task-group",
            "synthetic",
            str(metrics_json_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    figure_path = output_dir / "synthetic_results.png"
    table_path = output_dir / "synthetic_plotting_table.csv"
    assert figure_path.is_file()
    assert figure_path.stat().st_size > 0

    rows = _read_csv_rows(table_path)
    assert len(rows) == 16

    mf_gfn_row = _find_plotting_row(
        rows,
        task="branin",
        method="mf_gfn",
        round_index="1",
    )
    expected_scores = [
        -(11.0 + 13.0 + 10.0) / 3.0,
        -(12.0 + 14.0 + 11.0) / 3.0,
    ]
    expected_budgets = [1.2, 1.4]

    assert mf_gfn_row["task_label"] == "Branin"
    assert mf_gfn_row["method_label"] == "MF-GFN"
    assert float(mf_gfn_row["mean_top_k_score_mean"]) == pytest.approx(
        sum(expected_scores) / len(expected_scores)
    )
    assert float(mf_gfn_row["mean_top_k_score_std"]) == pytest.approx(
        stdev(expected_scores)
    )
    assert float(mf_gfn_row["cumulative_budget_mean"]) == pytest.approx(
        sum(expected_budgets) / len(expected_budgets)
    )
    assert float(mf_gfn_row["cumulative_budget_std"]) == pytest.approx(
        stdev(expected_budgets)
    )


def test_plot_molecule_results_writes_figure_and_plotting_table(
    plotting_output_root: Path,
) -> None:
    """Molecule plotting should save a figure and the exact aggregated CSV."""

    run_root = plotting_output_root / "molecule_runs"
    _write_molecule_runs(run_root)

    output_dir = plotting_output_root / "molecule_plots"
    plot_reproduce_paper_main(
        [
            "--task-group",
            "molecules",
            str(run_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    figure_path = output_dir / "molecule_results.png"
    table_path = output_dir / "molecule_plotting_table.csv"
    assert figure_path.is_file()
    assert figure_path.stat().st_size > 0

    rows = _read_csv_rows(table_path)
    assert len(rows) == 16

    mf_gfn_row = _find_plotting_row(
        rows,
        task="molecules_ip",
        method="mf_gfn",
        round_index="1",
    )
    expected_energies = [-5.0, -5.5]
    expected_scores = [-5.0, -5.5]
    expected_budget_fractions = [28.0 / 1260.0, 35.0 / 1260.0]
    expected_diversity = _expected_tanimoto_distance(["[C]", "[O]", "[N]", "[C][O]"])

    assert mf_gfn_row["task_label"] == "Molecules (IP)"
    assert mf_gfn_row["method_label"] == "MF-GFN"
    assert float(mf_gfn_row["mean_top_k_energy_mean"]) == pytest.approx(
        sum(expected_energies) / len(expected_energies)
    )
    assert float(mf_gfn_row["mean_top_k_energy_std"]) == pytest.approx(
        stdev(expected_energies)
    )
    assert float(mf_gfn_row["mean_top_k_score_mean"]) == pytest.approx(
        sum(expected_scores) / len(expected_scores)
    )
    assert float(mf_gfn_row["mean_top_k_score_std"]) == pytest.approx(
        stdev(expected_scores)
    )
    assert float(
        mf_gfn_row["budget_fraction_of_total_sf_gfn_budget_mean"]
    ) == pytest.approx(sum(expected_budget_fractions) / len(expected_budget_fractions))
    assert float(mf_gfn_row["mean_pairwise_tanimoto_distance_mean"]) == pytest.approx(
        expected_diversity
    )
    assert float(mf_gfn_row["mean_pairwise_tanimoto_distance_std"]) == pytest.approx(
        0.0
    )


def _write_synthetic_runs(output_root: Path) -> None:
    """Write a complete synthetic plotting fixture covering all paper methods."""

    for task in ("branin", "hartmann"):
        for method_index, method in enumerate(PAPER_METHODS):
            for seed in (1, 2):
                if task == "branin":
                    base_value = 10.0 + method_index + seed
                    initial_observations = [
                        {"x": [0.0, 0.0], "y": base_value, "fidelity": 3},
                        {"x": [1.0, 1.0], "y": base_value + 2.0, "fidelity": 3},
                    ]
                    round_history = [
                        {
                            "round_index": 1,
                            "cumulative_cost": 1.0 + 0.1 * method_index + 0.2 * seed,
                            "new_observations": [
                                {
                                    "x": [2.0, 2.0],
                                    "y": base_value - 1.0,
                                    "fidelity": 3,
                                }
                            ],
                        }
                    ]
                else:
                    base_value = 1.0 + method_index + seed
                    initial_observations = [
                        {"x": [0.0] * 6, "y": base_value, "fidelity": 3},
                        {"x": [0.1] * 6, "y": base_value + 2.0, "fidelity": 3},
                    ]
                    round_history = [
                        {
                            "round_index": 1,
                            "cumulative_cost": 2.0 + 0.1 * method_index + 0.25 * seed,
                            "new_observations": [
                                {
                                    "x": [0.2] * 6,
                                    "y": base_value + 4.0,
                                    "fidelity": 3,
                                }
                            ],
                        }
                    ]

                _write_recorded_run(
                    output_root=output_root,
                    task_group="synthetic",
                    task=task,
                    method=method,
                    seed=seed,
                    initial_mode="multi_fidelity",
                    initial_observations=initial_observations,
                    round_history=round_history,
                )


def _write_molecule_runs(output_root: Path) -> None:
    """Write a complete molecule plotting fixture covering all paper methods."""

    base_initial_values = [5.0, 4.0, 6.0]
    for task in ("molecules_ip", "molecules_ea"):
        for method_index, method in enumerate(PAPER_METHODS):
            for seed in (1, 2):
                offset = method_index + 0.5 * seed
                sign = -1.0 if task == "molecules_ip" else 1.0
                initial_observations = [
                    {
                        "x": "[C]",
                        "y": sign * (base_initial_values[0] + offset),
                        "fidelity": 3,
                        "metadata": {"identifier": f"{task}-{method}-c-{seed}"},
                    },
                    {
                        "x": "[O]",
                        "y": sign * (base_initial_values[1] + offset),
                        "fidelity": 3,
                        "metadata": {"identifier": f"{task}-{method}-o-{seed}"},
                    },
                    {
                        "x": "[N]",
                        "y": sign * (base_initial_values[2] + offset),
                        "fidelity": 3,
                        "metadata": {"identifier": f"{task}-{method}-n-{seed}"},
                    },
                    {"x": "[F]", "y": sign * (9.0 + offset), "fidelity": 1},
                ]
                round_history = [
                    {
                        "round_index": 1,
                        "cumulative_cost": 21.0 + method_index + 7.0 * seed,
                        "new_observations": [
                            {
                                "x": "[C][O]",
                                "y": sign * (3.0 + offset),
                                "fidelity": 3,
                                "metadata": {
                                    "identifier": f"{task}-{method}-co-{seed}"
                                },
                            },
                            {"x": "[C][F]", "y": sign * (8.0 + offset), "fidelity": 2},
                        ],
                    }
                ]

                _write_recorded_run(
                    output_root=output_root,
                    task_group="molecules",
                    task=task,
                    method=method,
                    seed=seed,
                    initial_mode="multi_fidelity",
                    initial_observations=initial_observations,
                    round_history=round_history,
                )


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
) -> Path:
    """Write a minimal recorded run directory for plotting regression tests."""

    run_directory = output_root / task / method / f"seed_{seed:04d}"
    run_directory.mkdir(parents=True, exist_ok=True)
    config_initial_data: dict[str, Any] = {"path": f"data/{task}/{initial_mode}.csv"}
    if initial_mode == "multi_fidelity":
        config_initial_data["fidelity_column"] = "fidelity"

    manifest = {
        "schema_version": 1,
        "run": {
            "task_group": task_group,
            "task": task,
            "method": method,
            "seed": seed,
        },
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
            "config_files": [
                f"scripts/configs/reproduce_paper/{task_group}/{task}/{method}.yaml",
            ],
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV plotting table into a list of string dictionaries."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_plotting_row(
    rows: Sequence[dict[str, str]],
    *,
    task: str,
    method: str,
    round_index: str,
) -> dict[str, str]:
    """Return one plotting-table row matching the requested identifiers."""

    for row in rows:
        if (
            row["task"] == task
            and row["method"] == method
            and row["round_index"] == round_index
        ):
            return row
    raise AssertionError(
        f"Plotting table row not found for task={task!r}, method={method!r}, "
        f"round_index={round_index!r}."
    )


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
    return sum(distances) / len(distances)

"""Dependency-light smoke test for the molecule benchmark artifact pipeline."""

import json
from pathlib import Path

from activelearning.config import ActiveLearningConfig
from activelearning.utils.config_loader import load_and_parse
from activelearning.utils.types import Observation
from activelearning_molecules.config_catalogs import CONFIG_CATALOGS
from scripts.evaluate_molecule_benchmark import evaluate_benchmark
from scripts.plot_molecule_benchmark import plot_benchmark


def test_composed_config_to_metrics_and_plot_artifacts(tmp_path: Path) -> None:
    """A composed benchmark run can be evaluated and plotted without xTB."""
    repository_root = Path(__file__).resolve().parents[2]
    config_root = repository_root / "applications/molecules/config/xtb_ipea_benchmark"
    config = load_and_parse(
        [
            config_root / "base.yaml",
            config_root / "encoders/gp_molformer.yaml",
            config_root / "tasks/ea_mf.yaml",
            config_root / "methods/random.yaml",
        ],
        ActiveLearningConfig,
        catalogs={"activelearning-molecules": CONFIG_CATALOGS},
    )
    assert config.sampler.output_representation == "smiles"

    run_dir = tmp_path / "runs/ea/random/seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "benchmark": {"task": "ea", "method": "random"},
                    "runtime": {"seed": 42},
                    "oracle": {
                        "type": "XTBIPEAOracle",
                        "task": "ea",
                        "fidelity_costs": {"3": 7.0},
                        "gfn_version": 2,
                        "ff": "mmff",
                        "correction_factor": 4.8455,
                        "mol_repr": "smiles",
                    },
                },
                "initial_data": {
                    "initial_observations": [{"x": "C", "y": 1.0, "fidelity": 3}]
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "round_history.jsonl").write_text(
        json.dumps(
            {
                "round_index": 1,
                "cumulative_cost": 7.0,
                "valid_observations": [{"x": "CC", "y": 0.0, "fidelity": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeOracle:
        """Return a deterministic fidelity-three value for the new molecule."""

        def query(self, candidates: list[object]) -> list[Observation]:
            return [
                Observation(x=getattr(candidate, "x"), y=2.0, fidelity=3)
                for candidate in candidates
            ]

    metrics_dir = tmp_path / "metrics"
    evaluate_benchmark(
        tmp_path / "runs",
        metrics_dir,
        oracle_factory=lambda _: FakeOracle(),
    )
    plot_dir = tmp_path / "plots"
    plot_benchmark(
        metrics_dir / "molecule_metrics.json",
        plot_dir,
        allow_incomplete=True,
    )

    assert (metrics_dir / "fidelity3_cache.jsonl").is_file()
    assert (metrics_dir / "molecule_metrics.json").is_file()
    assert (plot_dir / "molecule_benchmark.svg").is_file()
    assert (plot_dir / "molecule_benchmark.png").is_file()
    assert (plot_dir / "molecule_benchmark_plot_data.csv").is_file()

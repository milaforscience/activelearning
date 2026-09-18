import json
from pathlib import Path

import pytest

from activelearning.utils.types import Observation
from scripts.evaluate_molecule_benchmark import evaluate_benchmark


def write_run(
    root: Path,
    *,
    cost: float = 7.0,
    task: str = "ea",
    method: str = "random",
    seed: int = 42,
) -> Path:
    """Write a compact run artifact with duplicate and recorded-f3 molecules."""
    run_dir = root / task / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": {
            "benchmark": {"task": task, "method": method},
            "runtime": {"seed": seed},
            "oracle": {
                "type": "XTBIPEAOracle",
                "task": task,
                "fidelity_costs": {"3": cost},
                "gfn_version": 2,
                "ff": "mmff",
                "correction_factor": 4.8455,
                "mol_repr": "smiles",
            },
        },
        "initial_data": {
            "initial_observations": [
                {"x": "C", "y": 1.0, "fidelity": 3, "metadata": None}
            ]
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rounds = [
        {
            "round_index": 1,
            "cumulative_cost": 2.0,
            "valid_observations": [
                {"x": "C(C)", "y": 0.0, "fidelity": 1, "metadata": None}
            ],
        },
        {
            "round_index": 2,
            "cumulative_cost": 4.0,
            "valid_observations": [
                {"x": "CC", "y": 5.0, "fidelity": 3, "metadata": None},
                {"x": "O", "y": 0.0, "fidelity": 1, "metadata": None},
            ],
        },
    ]
    (run_dir / "round_history.jsonl").write_text(
        "\n".join(json.dumps(record) for record in rounds) + "\n",
        encoding="utf-8",
    )
    return run_dir


class FakeOracle:
    """Small deterministic oracle used to test cache behavior."""

    def __init__(self, calls: list[list[str]], *, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail

    def query(self, candidates):
        values = [candidate.x for candidate in candidates]
        self.calls.append(values)
        return [
            Observation(x=candidate.x, y=float("nan") if self.fail else 3.0, fidelity=3)
            for candidate in candidates
        ]


def test_evaluation_includes_initial_data_and_reuses_recorded_fidelity_three(
    tmp_path: Path,
) -> None:
    """Canonical duplicates collapse and later recorded f3 values are reused."""
    base_dir = tmp_path / "runs"
    write_run(base_dir)
    calls: list[list[str]] = []

    rows = evaluate_benchmark(
        base_dir,
        tmp_path / "metrics",
        oracle_factory=lambda _: FakeOracle(calls),
    )

    assert len(rows) == 2
    assert rows[0]["unique_molecule_count"] == 2
    assert rows[0]["fidelity_3_count"] == 2
    assert rows[1]["unique_molecule_count"] == 3
    assert rows[1]["fidelity_3_count"] == 3
    assert rows[1]["additional_fidelity_3_rescoring_count"] == 1
    assert calls == [["O"]]
    assert rows[0]["mean_top_100_score"] == pytest.approx(3.0)


def test_evaluation_cache_resumes_and_parameter_changes_invalidate(
    tmp_path: Path,
) -> None:
    """Successful cache entries are reused only for the same oracle fingerprint."""
    base_dir = tmp_path / "runs"
    write_run(base_dir)
    calls: list[list[str]] = []
    output_dir = tmp_path / "metrics"

    evaluate_benchmark(
        base_dir,
        output_dir,
        oracle_factory=lambda _: FakeOracle(calls),
    )
    evaluate_benchmark(
        base_dir,
        output_dir,
        oracle_factory=lambda _: FakeOracle(calls),
    )
    assert calls == [["O"]]

    write_run(base_dir, cost=8.0)
    evaluate_benchmark(
        base_dir,
        output_dir,
        oracle_factory=lambda _: FakeOracle(calls),
    )
    assert calls == [["O"], ["O"]]


def test_evaluation_records_failures_and_retries_only_when_requested(
    tmp_path: Path,
) -> None:
    """Failed cache entries are durable and require an explicit retry flag."""
    base_dir = tmp_path / "runs"
    write_run(base_dir)
    output_dir = tmp_path / "metrics"
    calls: list[list[str]] = []

    evaluate_benchmark(
        base_dir,
        output_dir,
        oracle_factory=lambda _: FakeOracle(calls, fail=True),
    )
    evaluate_benchmark(
        base_dir,
        output_dir,
        oracle_factory=lambda _: FakeOracle(calls, fail=False),
    )
    assert calls == [["O"]]

    evaluate_benchmark(
        base_dir,
        output_dir,
        retry_failures=True,
        oracle_factory=lambda _: FakeOracle(calls, fail=False),
    )
    assert calls == [["O"], ["O"]]

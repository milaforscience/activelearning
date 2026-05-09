from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _run_launcher(script_name: str, *seeds: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REPOSITORY_ROOT / "scripts" / script_name), *seeds],
        check=True,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "1"},
    )


def test_synthetic_launcher_prints_expected_commands_for_one_seed() -> None:
    result = _run_launcher("run_reproduce_paper_synthetic.sh", "0")

    lines = [line for line in result.stdout.splitlines() if line]

    assert len(lines) == 8
    assert any(
        "scripts/configs/reproduce_paper/synthetic/branin/mf_gfn.yaml" in line
        and "runtime.seed=0" in line
        for line in lines
    )
    assert any(
        "scripts/configs/reproduce_paper/synthetic/hartmann/random_fid_gfn.yaml" in line
        and "runtime.seed=0" in line
        for line in lines
    )


def test_molecule_launcher_prints_expected_commands_for_two_seeds() -> None:
    result = _run_launcher("run_reproduce_paper_molecules.sh", "0", "4")

    lines = [line for line in result.stdout.splitlines() if line]

    assert len(lines) == 16
    assert any(
        "scripts/configs/reproduce_paper/molecules/molecules_ip/mf_gfn.yaml" in line
        and "runtime.seed=0" in line
        for line in lines
    )
    assert any(
        "scripts/configs/reproduce_paper/molecules/molecules_ea/random.yaml" in line
        and "runtime.seed=4" in line
        for line in lines
    )

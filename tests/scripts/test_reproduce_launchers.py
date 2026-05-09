from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPO_PYTHON = REPOSITORY_ROOT / ".venv" / "bin" / "python"


def _run_launcher(
    script_name: str,
    *seeds: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REPOSITORY_ROOT / "scripts" / script_name), *seeds],
        check=True,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "1", **(extra_env or {})},
    )


def test_launchers_include_sbatch_directives() -> None:
    synthetic_script = (
        REPOSITORY_ROOT / "scripts" / "run_reproduce_paper_synthetic.sh"
    ).read_text()
    molecules_script = (
        REPOSITORY_ROOT / "scripts" / "run_reproduce_paper_molecules.sh"
    ).read_text()

    assert "#SBATCH --job-name=repro-synthetic" in synthetic_script
    assert "#SBATCH --nodes=1" in synthetic_script
    assert "#SBATCH --ntasks=1" in synthetic_script
    assert "#SBATCH --cpus-per-task=4" in synthetic_script
    assert "#SBATCH --mem=16G" in synthetic_script
    assert "#SBATCH --time=24:00:00" in synthetic_script
    assert "#SBATCH --output=slurm-logs/%x-%j.out" in synthetic_script
    assert "#SBATCH --error=slurm-logs/%x-%j.err" in synthetic_script
    assert 'VENV_DIR="${REPO_ROOT}/.venv"' in synthetic_script
    assert "sync_cmd=(uv sync --frozen)" in synthetic_script
    assert "#SBATCH --job-name=repro-molecules" in molecules_script
    assert "#SBATCH --nodes=1" in molecules_script
    assert "#SBATCH --ntasks=1" in molecules_script
    assert "#SBATCH --cpus-per-task=8" in molecules_script
    assert "#SBATCH --mem=32G" in molecules_script
    assert "#SBATCH --time=72:00:00" in molecules_script
    assert "#SBATCH --output=slurm-logs/%x-%j.out" in molecules_script
    assert "#SBATCH --error=slurm-logs/%x-%j.err" in molecules_script
    assert 'VENV_DIR="${REPO_ROOT}/.venv"' in molecules_script
    assert "sync_cmd=(uv sync --frozen --extra molecules)" in molecules_script


def test_synthetic_launcher_prints_expected_commands_for_one_seed() -> None:
    result = _run_launcher("run_reproduce_paper_synthetic.sh", "0")

    lines = [line for line in result.stdout.splitlines() if line]

    assert len(lines) == 8
    assert all(str(REPO_PYTHON) in line for line in lines)
    assert all("-m activelearning.main" in line for line in lines)
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


def test_synthetic_launcher_supports_slurm_array_dispatch() -> None:
    result = _run_launcher(
        "run_reproduce_paper_synthetic.sh",
        "0",
        extra_env={"SLURM_ARRAY_TASK_ID": "7"},
    )

    lines = [line for line in result.stdout.splitlines() if line]

    assert len(lines) == 1
    assert (
        "scripts/configs/reproduce_paper/synthetic/hartmann/random_fid_gfn.yaml"
        in lines[0]
    )
    assert "runtime.seed=0" in lines[0]


def test_molecule_launcher_prints_expected_commands_for_two_seeds() -> None:
    result = _run_launcher("run_reproduce_paper_molecules.sh", "0", "4")

    lines = [line for line in result.stdout.splitlines() if line]

    assert len(lines) == 16
    assert all(str(REPO_PYTHON) in line for line in lines)
    assert all("-m activelearning.main" in line for line in lines)
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

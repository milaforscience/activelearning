#!/usr/bin/env bash
# Submit a single xTB IP/EA benchmark job as a smoke test before the full matrix.
# Run with bash from a login node; extra arguments are forwarded to sbatch:
#   bash scripts/run_molecule_benchmark_smoke_slurm.sh [--partition=long ...]
# Override the combination with TASK, METHOD, and SEED (defaults: ea, mf_gfn, 42).
# The job writes to the same output directory as that combination in the full run.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
task="${TASK:-ea}"
method="${METHOD:-mf_gfn}"
seed="${SEED:-42}"

sbatch \
  --job-name="xtb-ipea-smoke-${task}-${method}-s${seed}" \
  --chdir="${REPO_ROOT}" \
  --export="ALL,MOLECULE_BENCHMARK_REPO_ROOT=${REPO_ROOT},MOLECULE_BENCHMARK_TASK=${task},MOLECULE_BENCHMARK_METHOD=${method},MOLECULE_BENCHMARK_SEED=${seed}" \
  "$@" \
  "${SCRIPT_DIR}/run_molecule_benchmark_slurm.sh"

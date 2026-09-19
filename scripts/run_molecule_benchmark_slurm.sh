#!/usr/bin/env bash
#SBATCH --job-name=xtb-ipea
#SBATCH --partition=main
#SBATCH --array=0-23%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this script with sbatch so SLURM_ARRAY_TASK_ID is set." >&2
  exit 2
fi

if ! [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] \
  || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= 24 )); then
  echo "SLURM_ARRAY_TASK_ID must be an integer in [0, 23]." >&2
  exit 2
fi

command -v uv >/dev/null || {
  echo "uv is required but was not found on PATH." >&2
  exit 1
}
command -v xtb >/dev/null || {
  echo "xtb is required but was not found on PATH." >&2
  exit 1
}

tasks=(ea ip)
methods=(sf_s3gfn mf_s3gfn random_fidelity_s3gfn random)
seeds=(42 43 44)

task_index=$((SLURM_ARRAY_TASK_ID / 12))
method_index=$(((SLURM_ARRAY_TASK_ID % 12) / 3))
seed_index=$((SLURM_ARRAY_TASK_ID % 3))

task="${tasks[task_index]}"
method="${methods[method_index]}"
seed="${seeds[seed_index]}"

echo "Starting molecule benchmark: task=${task} method=${method} seed=${seed}"
echo "SLURM job=${SLURM_JOB_ID:-unknown} array_task=${SLURM_ARRAY_TASK_ID}"

# Prepare the environment once before submission with:
#   uv sync --frozen --package activelearning-molecules
export ACTIVELEARNING_MOLECULES_COMMAND="${ACTIVELEARNING_MOLECULES_COMMAND:-uv run --frozen --package activelearning-molecules activelearning-molecules}"

runner_args=(
  --task "${task}"
  --method "${method}"
  --seed "${seed}"
)

if [[ "${MOLECULE_BENCHMARK_DRY_RUN:-0}" == "1" ]]; then
  runner_args+=(--dry-run)
fi

uv run --frozen --package activelearning-molecules \
  python scripts/run_molecule_benchmark.py "${runner_args[@]}"

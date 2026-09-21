#!/usr/bin/env bash
#SBATCH --job-name=xtb-ipea
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Run with bash (not sbatch) from a login node to submit one independent job
# per task/method/seed combination. Extra arguments are forwarded to sbatch:
#   bash scripts/run_molecule_benchmark_slurm.sh [--partition=long ...]
# Inside a submitted job, the same script runs that single combination.

set -euo pipefail

tasks=(ea ip)
methods=(sf_gfn mf_gfn random_fidelity_gfn random)
seeds=(42 43 44)

# Comet credentials: paste your API key here, or export COMET_API_KEY before
# running this script. Do not commit the key.
export COMET_API_KEY="${COMET_API_KEY:-PASTE_YOUR_COMET_API_KEY_HERE}"
if [[ "${COMET_API_KEY}" == "PASTE_YOUR_COMET_API_KEY_HERE" ]]; then
  echo "Set COMET_API_KEY in this script or in the environment." >&2
  exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  command -v sbatch >/dev/null || {
    echo "sbatch is required to submit the benchmark jobs." >&2
    exit 1
  }
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

  for task in "${tasks[@]}"; do
    for method in "${methods[@]}"; do
      for seed in "${seeds[@]}"; do
        sbatch \
          --job-name="xtb-ipea-${task}-${method}-s${seed}" \
          --chdir="${REPO_ROOT}" \
          --export="ALL,MOLECULE_BENCHMARK_REPO_ROOT=${REPO_ROOT},MOLECULE_BENCHMARK_TASK=${task},MOLECULE_BENCHMARK_METHOD=${method},MOLECULE_BENCHMARK_SEED=${seed}" \
          "$@" \
          "${SCRIPT_DIR}/run_molecule_benchmark_slurm.sh"
      done
    done
  done
  exit 0
fi

task="${MOLECULE_BENCHMARK_TASK:?Submit jobs by running this script with bash.}"
method="${MOLECULE_BENCHMARK_METHOD:?Submit jobs by running this script with bash.}"
seed="${MOLECULE_BENCHMARK_SEED:?Submit jobs by running this script with bash.}"
cd "${MOLECULE_BENCHMARK_REPO_ROOT:?Submit jobs by running this script with bash.}"

command -v uv >/dev/null || {
  echo "uv is required but was not found on PATH." >&2
  exit 1
}
command -v xtb >/dev/null || {
  echo "xtb is required but was not found on PATH." >&2
  exit 1
}

echo "Starting molecule benchmark: task=${task} method=${method} seed=${seed}"
echo "SLURM job=${SLURM_JOB_ID}"

# Prepare the environment once before submission with:
#   uv sync --frozen --all-packages --extra comet
export ACTIVELEARNING_MOLECULES_COMMAND="${ACTIVELEARNING_MOLECULES_COMMAND:-uv run --frozen --all-packages --extra comet activelearning-molecules}"

runner_args=(
  --task "${task}"
  --method "${method}"
  --seed "${seed}"
)

if [[ "${MOLECULE_BENCHMARK_DRY_RUN:-0}" == "1" ]]; then
  runner_args+=(--dry-run)
fi

uv run --frozen --all-packages --extra comet \
  python scripts/run_molecule_benchmark.py "${runner_args[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=repro-molecules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

# Submit the full sweep with:
#   sbatch scripts/run_reproduce_paper_molecules.sh
# Or fan out one config/seed pair per array task with:
#   sbatch --array=0-39 scripts/run_reproduce_paper_molecules.sh
# Override the defaults at submission time when needed, e.g.:
#   sbatch --time=7-00:00:00 --mem=48G scripts/run_reproduce_paper_molecules.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

methods=(mf_gfn sf_gfn random random_fid_gfn)
tasks=(molecules_ip molecules_ea)
seeds=(0 1 2 3 4)

if (($# > 0)); then
  seeds=("$@")
fi

work_items=()
for task in "${tasks[@]}"; do
  for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
      work_items+=("${task}|${method}|${seed}")
    done
  done
done

run_one() {
  local task="$1"
  local method="$2"
  local seed="$3"
  local cmd=(
    uv run activelearning
    "scripts/configs/reproduce_paper/molecules/${task}/${method}.yaml"
    "runtime.seed=${seed}"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#work_items[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range for ${#work_items[@]} work items." >&2
    exit 1
  fi

  IFS="|" read -r task method seed <<< "${work_items[SLURM_ARRAY_TASK_ID]}"
  run_one "${task}" "${method}" "${seed}"
  exit 0
fi

for work_item in "${work_items[@]}"; do
  IFS="|" read -r task method seed <<< "${work_item}"
  run_one "${task}" "${method}" "${seed}"
done

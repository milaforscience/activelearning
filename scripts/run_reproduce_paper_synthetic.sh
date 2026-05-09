#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

methods=(mf_gfn sf_gfn random random_fid_gfn)
tasks=(branin hartmann)
seeds=(0 1 2 3 4)

if (($# > 0)); then
  seeds=("$@")
fi

for task in "${tasks[@]}"; do
  for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
      cmd=(
        uv run activelearning
        "scripts/configs/reproduce_paper/synthetic/${task}/${method}.yaml"
        "runtime.seed=${seed}"
      )
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
      else
        "${cmd[@]}"
      fi
    done
  done
done

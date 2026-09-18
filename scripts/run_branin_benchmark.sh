#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_ROOT="config/branin_benchmark"
seeds=(0 1 2 3 4)
configs=(mf_gfn random sf_low_fid sf_mid_fid sf_high_fid)

for config in "${configs[@]}"; do
  for seed in "${seeds[@]}"; do
    echo "--- Running ${CONFIG_ROOT}/${config} seed=${seed} ---"
    uv run activelearning \
      "${CONFIG_ROOT}/base.yaml" \
      "${CONFIG_ROOT}/${config}.yaml" \
      "runtime.seed=${seed}"
  done
done

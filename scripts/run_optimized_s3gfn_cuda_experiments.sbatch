#!/usr/bin/env bash
#SBATCH --job-name=s3gfn_cuda
#SBATCH --output=s3gfn_cuda_%j.out
#SBATCH --error=s3gfn_cuda_%j.err
#SBATCH --gpus-per-task=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd -- "${REPO_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"
BENCHMARK_RUNNER="${REPO_ROOT}/benchmarks/run_optimized_s3gfn_cuda_experiments.py"
if [[ ! -f "${BENCHMARK_RUNNER}" ]]; then
  echo "Benchmark runner not found: ${BENCHMARK_RUNNER}" >&2
  echo "Submit from the repository root or set REPO_ROOT explicitly." >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${OUTPUT_DIR:-benchmark-results/s3gfn-cuda}"
PROFILE="${PROFILE:-full}"
SEEDS="${SEEDS:-0}"
BATCH_SIZES="${BATCH_SIZES:-64}"
RESUME="${RESUME:-0}"
GPU_UTILIZATION_TRACE="${GPU_UTILIZATION_TRACE:-1}"
FAKE_MODEL="${FAKE_MODEL:-0}"
STAGES="${STAGES:-}"

read -r -a seed_args <<< "${SEEDS}"
read -r -a batch_size_args <<< "${BATCH_SIZES}"

command_args=(
  uv run python "${BENCHMARK_RUNNER}"
  --device cuda
  --output-dir "${OUTPUT_DIR}"
  --profile "${PROFILE}"
  --seeds "${seed_args[@]}"
  --batch-sizes "${batch_size_args[@]}"
)

if [[ -n "${STAGES}" ]]; then
  read -r -a stage_args <<< "${STAGES}"
  command_args+=(--stages "${stage_args[@]}")
fi
if [[ "${RESUME}" == "1" ]]; then
  command_args+=(--resume)
fi
if [[ "${GPU_UTILIZATION_TRACE}" == "1" ]]; then
  command_args+=(--gpu-utilization-trace)
fi
if [[ "${FAKE_MODEL}" == "1" ]]; then
  command_args+=(--fake-model)
fi

echo "Running S3-GFN CUDA experiments"
echo "  host: ${HOSTNAME:-unknown}"
echo "  job: ${SLURM_JOB_ID:-unknown}"
echo "  output: ${OUTPUT_DIR}"
echo "  profile: ${PROFILE}"
echo "  seeds: ${SEEDS}"
echo "  batch sizes: ${BATCH_SIZES}"
echo "  stages: ${STAGES:-all}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader
fi

"${command_args[@]}"

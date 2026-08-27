#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=08:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/eval_run}"
SPLIT_FILE="${SPLIT_FILE:-}"
DATASET_PATH="${DATASET_PATH:-}"
TRAINER="${TRAINER:-bundle}"
SOURCE="${SOURCE:-real}"
REPRESENTATION="${REPRESENTATION:-scaled}"
MODEL_FAMILY="${MODEL_FAMILY:-minirocket}"
TARGET_LENGTH="${TARGET_LENGTH:-5000}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-5000}"
STRIDE="${STRIDE:-${SEGMENT_LENGTH}}"
MAX_SEGMENTS="${MAX_SEGMENTS:-}"
N_SAMPLES="${N_SAMPLES:-20000}"
SEED="${SEED:-42}"
N_RUNS="${N_RUNS:-1}"
TEST_RATIO="${TEST_RATIO:-0.2}"
INTERNAL_TEST_RATIO="${INTERNAL_TEST_RATIO:-}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
mkdir -p logs "$(dirname "${RUN_DIR}")"
echo "train job: RUN_DIR=${RUN_DIR} MODEL_FAMILY=${MODEL_FAMILY}"

ARGS=(
  scripts/train.py
  --trainer "${TRAINER}"
  --source "${SOURCE}"
  --representation "${REPRESENTATION}"
)

if [ -n "${SPLIT_FILE}" ]; then
  ARGS+=(--split-file "${SPLIT_FILE}")
fi
if [ -n "${DATASET_PATH}" ]; then
  ARGS+=(--dataset "${DATASET_PATH}")
fi
if [ "${TRAINER}" = "single-model" ]; then
  ARGS+=(--model-type "${MODEL_FAMILY}")
  ARGS+=(--target-length "${TARGET_LENGTH}")
  ARGS+=(--segment-length "${SEGMENT_LENGTH}")
  ARGS+=(--stride "${STRIDE}")
  [ -n "${MAX_SEGMENTS}" ] && ARGS+=(--max-segments "${MAX_SEGMENTS}")
fi
if [ "${TRAINER}" = "bundle" ] && [ "${SOURCE}" = "synthetic" ]; then
  ARGS+=(--n-samples "${N_SAMPLES}")
fi
if [ "${TRAINER}" = "bundle" ] && [ "${SOURCE}" = "real" ]; then
  ARGS+=(--target-length "${TARGET_LENGTH}")
fi
if [ "${TRAINER}" = "trivial" ] && [ "${SOURCE}" = "dataset" ]; then
  ARGS+=(--test-ratio "${TEST_RATIO}")
  ARGS+=(--segment-length "${SEGMENT_LENGTH}")
  ARGS+=(--stride "${STRIDE}")
fi
if [ "${TRAINER}" = "trivial" ] && [ "${SOURCE}" = "real" ]; then
  ARGS+=(--target-length "${TARGET_LENGTH}")
  [ -n "${INTERNAL_TEST_RATIO}" ] && ARGS+=(--internal-test-ratio "${INTERNAL_TEST_RATIO}")
fi

train() {
  if [ "${MODEL_FAMILY}" = "tsf" ] || [ "${MODEL_FAMILY}" = "minirocket" ]; then
    if [ -f "${REPO_ROOT}/pixi.toml" ] || [ -f "${REPO_ROOT}/pyproject.toml" ]; then
      module load 2025 2>/dev/null || true
      module load pixi 2>/dev/null || true
      pixi install 2>/dev/null || true
      pixi run python "$@"
    else
      module load 2025 2>/dev/null || true
      python "$@"
    fi
  else
    python "$@"
  fi
}

if [ "${N_RUNS}" -eq 1 ]; then
  train "${ARGS[@]}" --output-dir "${RUN_DIR}" --seed "${SEED}"
else
  for ((i=0; i<N_RUNS; i++)); do
    train "${ARGS[@]}" --output-dir "${RUN_DIR}/runs/${i}" --seed $((SEED + i))
  done
fi

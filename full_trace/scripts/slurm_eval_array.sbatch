#!/bin/bash
#SBATCH --job-name=eval_array
#SBATCH --time=04:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-0
#SBATCH --output=logs/eval_array-%A_%a.out
#SBATCH --error=logs/eval_array-%A_%a.err

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPLIT_FILE="${SPLIT_FILE:-${REPO_ROOT}/eval_run/split.json}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/eval_run/cache}"
ARRAY_RESULTS_DIR="${ARRAY_RESULTS_DIR:-${REPO_ROOT}/eval_run/array_results}"
EVALUATOR="${EVALUATOR:-bundle}"
METHOD_NAME="${METHOD_NAME:-}"
TARGET_LENGTH="${TARGET_LENGTH:-5000}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

if [ ! -f "${SPLIT_FILE}" ]; then
  echo "error: split file not found: ${SPLIT_FILE}" 1>&2
  exit 1
fi

mkdir -p "${ARRAY_RESULTS_DIR}" logs

ARGS=(
  scripts/eval.py
  --mode array-task
  --evaluator "${EVALUATOR}"
  --split-file "${SPLIT_FILE}"
  --output-dir "${ARRAY_RESULTS_DIR}"
  "${SLURM_ARRAY_TASK_ID}"
)

if [ "${EVALUATOR}" = "bundle" ]; then
  [ -z "${MODEL_DIR}" ] && { echo "error: MODEL_DIR required for evaluator=bundle" 1>&2; exit 1; }
  ARGS+=(--cache-dir "${MODEL_DIR}")
fi
if [ "${EVALUATOR}" = "saved-model" ]; then
  [ -z "${MODEL_DIR}" ] && { echo "error: MODEL_DIR required for evaluator=saved-model" 1>&2; exit 1; }
  if [ -f "${MODEL_DIR}/LATEST" ]; then
    SUB=$(cat "${MODEL_DIR}/LATEST")
    if [ "${SUB#/}" = "${SUB}" ]; then
      MODEL_DIR="${MODEL_DIR}/${SUB}"
    else
      MODEL_DIR="${SUB}"
    fi
  fi
  ARGS+=(--model-dir "${MODEL_DIR}")
  [ -n "${METHOD_NAME}" ] && ARGS+=(--method-name "${METHOD_NAME}")
fi
if [ "${EVALUATOR}" = "rules" ]; then
  ARGS+=(--target-length "${TARGET_LENGTH}")
  [ -n "${METHOD_NAME}" ] && ARGS+=(--method-name "${METHOD_NAME}")
fi

if [ "${EVALUATOR}" = "saved-model" ] && [ -n "${METHOD_NAME}" ]; then
  if [ -f "${REPO_ROOT}/pixi.toml" ] || [ -f "${REPO_ROOT}/pyproject.toml" ]; then
    module load 2025 2>/dev/null || true
    module load pixi 2>/dev/null || true
    pixi install 2>/dev/null || true
    pixi run python "${ARGS[@]}"
  else
    module load 2025 2>/dev/null || true
    python "${ARGS[@]}"
  fi
else
  python "${ARGS[@]}"
fi

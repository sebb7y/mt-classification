#!/bin/bash
#SBATCH --job-name=eval_summary
#SBATCH --time=01:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/eval_summary-%j.out
#SBATCH --error=logs/eval_summary-%j.err

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPLIT_FILE="${SPLIT_FILE:-${REPO_ROOT}/eval_run/split.json}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/eval_run/model}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/eval_run/bead_eval.json}"
EVALUATOR="${EVALUATOR:-saved-model}"
SUM_THRESHOLD_FRAC="${SUM_THRESHOLD_FRAC:-0.5}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
mkdir -p logs

if [ ! -f "${SPLIT_FILE}" ]; then
  echo "error: split file not found: ${SPLIT_FILE}" 1>&2
  exit 1
fi
if [ ! -d "${MODEL_DIR}" ]; then
  echo "error: model dir not found: ${MODEL_DIR}" 1>&2
  exit 1
fi

python scripts/eval.py \
  --mode bead-accuracy \
  --evaluator "${EVALUATOR}" \
  --split-file "${SPLIT_FILE}" \
  --model-dir "${MODEL_DIR}" \
  -o "${OUTPUT}" \
  --sum-threshold-frac "${SUM_THRESHOLD_FRAC}"

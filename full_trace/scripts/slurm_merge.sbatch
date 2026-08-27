#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --time=00:30:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/merge-%j.out
#SBATCH --error=logs/merge-%j.err

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARRAY_RESULTS_DIR="${ARRAY_RESULTS_DIR:-${REPO_ROOT}/eval_run/array_results}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/eval_run}"
N_RUNS="${N_RUNS:-1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
mkdir -p logs

python scripts/merge_results.py \
  --array-dir "${ARRAY_RESULTS_DIR}" \
  -o "${RUN_DIR}" \
  --n-runs "${N_RUNS}"

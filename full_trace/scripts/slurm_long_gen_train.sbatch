#!/bin/bash
#SBATCH --job-name=long_traces
#SBATCH --time=16:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=long_traces-%j.out
#SBATCH --error=long_traces-%j.err

set -e

ORIG_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="${ORIG_SUBMIT_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

N_TRACES=2000
TIME_MIN=1500.0
TIME_MAX=15000.0
DT_MIN=0.03
DT_MAX=0.03
SEED=42

DATASET_NAME="dataset_n${N_TRACES}_t${TIME_MIN}-${TIME_MAX}s_seed${SEED}.npz"
if [ -n "${TMPDIR}" ] && [ -d "${TMPDIR}" ]; then
  DATASET_PATH="${TMPDIR}/dataset_cache/${DATASET_NAME}"
  OUTPUT_DIR="${TMPDIR}/minirocket_long_traces_output"
else
  DATASET_PATH="${REPO_ROOT}/dataset_cache/${DATASET_NAME}"
  OUTPUT_DIR="${REPO_ROOT}/minirocket_long_traces_output"
fi

module load 2025 2>/dev/null || true
if [ -f "${REPO_ROOT}/pixi.toml" ] || [ -f "${REPO_ROOT}/pyproject.toml" ]; then
  module load pixi 2>/dev/null || true
  pixi install 2>/dev/null || true
  RUN_PYTHON="pixi run python"
else
  RUN_PYTHON="python"
fi

${RUN_PYTHON} "${REPO_ROOT}/slurm_gen_dataset.py" \
  --n-traces ${N_TRACES} \
  --seed ${SEED} \
  --total-time-min ${TIME_MIN} \
  --total-time-max ${TIME_MAX} \
  --dt-min ${DT_MIN} \
  --dt-max ${DT_MAX} \
  --generation-method direct

if [ ! -f "${DATASET_PATH}" ]; then
  echo "error: dataset not found at ${DATASET_PATH}" 1>&2
  if [ -n "${TMPDIR}" ] && [ -d "${TMPDIR}/dataset_cache" ]; then
    DATASET_PATH=$(ls "${TMPDIR}"/dataset_cache/dataset_n${N_TRACES}_t*.npz 2>/dev/null | head -1)
    [ -n "${DATASET_PATH}" ] || exit 1
  else
    exit 1
  fi
fi

${RUN_PYTHON} "${REPO_ROOT}/train_minirocket_synthetic.py" \
  --dataset "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --test-size 0.2 \
  --n-fractions 8 \
  --seed ${SEED}

if [ -n "${TMPDIR}" ] && [ -d "${TMPDIR}" ]; then
  mkdir -p "${ORIG_SUBMIT_DIR}/dataset_cache" "${ORIG_SUBMIT_DIR}/minirocket_long_traces_output"
  [ -f "${DATASET_PATH}" ] && cp -v "${DATASET_PATH}" "${ORIG_SUBMIT_DIR}/dataset_cache/"
  [ -d "${OUTPUT_DIR}" ] && cp -rv "${OUTPUT_DIR}"/* "${ORIG_SUBMIT_DIR}/minirocket_long_traces_output/"
fi

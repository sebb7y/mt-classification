#!/usr/bin/env bash
# eval pipeline: build manifest -> make_split -> train -> eval array -> merge
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
SOURCE="${SOURCE:-TX_OF}"
N_RUNS="${N_RUNS:-1}"
RUN_SETUPS="${RUN_SETUPS:-same_kind,same_to_multi,multi_to_multi,synthetic_train,no_data,minirocket_real,tsf_real,minirocket_cached}"
MERGE_JOBS=()

cd "${REPO_ROOT}"
mkdir -p logs

python scripts/eval_manifest.py "${DATA_ROOT}" -o eval_manifest.csv
if [ ! -f eval_manifest.csv ]; then
  echo "error: eval_manifest.csv not created" 1>&2
  exit 1
fi
N_MANIFEST=$(tail -n +2 eval_manifest.csv | wc -l)
echo "manifest: ${N_MANIFEST} experiments"

run_setup() {
  local name="$1"
  local out_dir="$2"
  shift 2
  local split_args=("$@")
  if [[ ",${RUN_SETUPS}," != *",${name},"* ]]; then
    echo "  skip ${name} (not in RUN_SETUPS)"
    return 0
  fi
  echo "  setup: ${name} -> ${out_dir}"
  mkdir -p "${out_dir}"
  python scripts/make_split.py eval_manifest.csv -o "${out_dir}" "${split_args[@]}"
  N_TEST=$(python -c "import json; d=json.load(open('${out_dir}/split.json')); print(len(d['test_rows']))")
  N_TRAIN=$(python -c "import json; d=json.load(open('${out_dir}/split.json')); print(len(d['train_rows']))")
  echo "  train: ${N_TRAIN}, test: ${N_TEST}"

  export SPLIT_FILE="${REPO_ROOT}/${out_dir}/split.json"
  export RUN_DIR="${REPO_ROOT}/${out_dir}"
  export ARRAY_RESULTS_DIR="${REPO_ROOT}/${out_dir}/array_results"
  export MODEL_DIR="${REPO_ROOT}/${out_dir}/cache"

  if [ "${name}" = "no_data" ]; then
    EVAL_JOB=$(sbatch --array=0-$(($N_TEST - 1)) --parsable --export=SPLIT_FILE,ARRAY_RESULTS_DIR,EVALUATOR=rules,METHOD_NAME=approach4_rules scripts/slurm_eval_array.sbatch)
    echo "  submitted rules eval: ${EVAL_JOB}"
  elif [ "${name}" = "minirocket_cached" ]; then
    EVAL_JOB=""
    if [ -z "${CACHED_MINIROCKET_DIR}" ] && [ -d "${REPO_ROOT}/outputs" ]; then
      LATEST_MR=$(ls -td "${REPO_ROOT}/outputs"/minirocket_* 2>/dev/null | head -1)
      [ -n "${LATEST_MR}" ] && [ -f "${LATEST_MR}/model.pkl" ] && CACHED_MINIROCKET_DIR="${LATEST_MR}"
    fi
    if [ -z "${CACHED_MINIROCKET_DIR}" ]; then
      echo "  skip minirocket_cached (set CACHED_MINIROCKET_DIR or place model under outputs/minirocket_*)"
    else
      CACHED="${CACHED_MINIROCKET_DIR}"
      [ -f "${CACHED}/LATEST" ] && CACHED=$(cat "${CACHED}/LATEST") || true
      [ ! -d "${CACHED}" ] && CACHED="${CACHED_MINIROCKET_DIR}"
      if [ -f "${CACHED}/model.pkl" ]; then
        export MODEL_DIR="${CACHED}"
        EVAL_JOB=$(sbatch --array=0-$(($N_TEST - 1)) --parsable --export=SPLIT_FILE,MODEL_DIR,ARRAY_RESULTS_DIR,EVALUATOR=saved-model,METHOD_NAME=minirocket scripts/slurm_eval_array.sbatch)
        echo "  submitted eval (cached minirocket): ${EVAL_JOB}"
      else
        echo "  skip minirocket_cached (no model.pkl at ${CACHED})"
      fi
    fi
  else
    if [ "${name}" = "synthetic_train" ]; then
      export RUN_DIR="${REPO_ROOT}/${out_dir}/cache"
      TRAIN_JOB=$(sbatch --parsable --export=RUN_DIR,N_RUNS,TRAINER=bundle,SOURCE=synthetic,REPRESENTATION=scaled scripts/slurm_train.sbatch)
      MODEL_BASE="${REPO_ROOT}/${out_dir}/cache"
      EVAL_STUFF="EVALUATOR=bundle"
    elif [ "${name}" = "minirocket_real" ]; then
      export RUN_DIR="${REPO_ROOT}/${out_dir}/cache_minirocket"
      TRAIN_JOB=$(sbatch --parsable --export=SPLIT_FILE,RUN_DIR,N_RUNS,TRAINER=single-model,SOURCE=real,REPRESENTATION=scaled,MODEL_FAMILY=minirocket scripts/slurm_train.sbatch)
      MODEL_BASE="${REPO_ROOT}/${out_dir}/cache_minirocket"
      EVAL_STUFF="EVALUATOR=saved-model,METHOD_NAME=minirocket"
    elif [ "${name}" = "tsf_real" ]; then
      export RUN_DIR="${REPO_ROOT}/${out_dir}/cache_tsf"
      TRAIN_JOB=$(sbatch --parsable --export=SPLIT_FILE,RUN_DIR,N_RUNS,TRAINER=single-model,SOURCE=real,REPRESENTATION=scaled,MODEL_FAMILY=tsf scripts/slurm_train.sbatch)
      MODEL_BASE="${REPO_ROOT}/${out_dir}/cache_tsf"
      EVAL_STUFF="EVALUATOR=saved-model,METHOD_NAME=tsf"
    else
      export RUN_DIR="${REPO_ROOT}/${out_dir}/cache"
      TRAIN_JOB=$(sbatch --parsable --export=SPLIT_FILE,RUN_DIR,N_RUNS,TRAINER=bundle,SOURCE=real,REPRESENTATION=scaled scripts/slurm_train.sbatch)
      MODEL_BASE="${REPO_ROOT}/${out_dir}/cache"
      EVAL_STUFF="EVALUATOR=bundle"
    fi
    if [ "${N_RUNS}" -eq 1 ]; then
      EVAL_JOB=$(sbatch --array=0-$(($N_TEST - 1)) --dependency=afterok:${TRAIN_JOB} --parsable --export=SPLIT_FILE,MODEL_DIR=${MODEL_BASE},ARRAY_RESULTS_DIR,${EVAL_STUFF} scripts/slurm_eval_array.sbatch)
    else
      EVAL_JOBS=()
      for ((i=0;i<N_RUNS;i++)); do
        j=$(sbatch --array=0-$(($N_TEST - 1)) --dependency=afterok:${TRAIN_JOB} --parsable --export=SPLIT_FILE,MODEL_DIR=${MODEL_BASE}/runs/${i},ARRAY_RESULTS_DIR=${REPO_ROOT}/${out_dir}/array_results/runs/${i},${EVAL_STUFF} scripts/slurm_eval_array.sbatch)
        EVAL_JOBS+=("${j}")
      done
      EVAL_JOB=$(IFS=:; echo "${EVAL_JOBS[*]}")
    fi
    echo "  submitted train: ${TRAIN_JOB} -> eval: ${EVAL_JOB}"
  fi

  if [ -n "${EVAL_JOB}" ]; then
    export RUN_DIR="${REPO_ROOT}/${out_dir}"
    MERGE_N_RUNS="${N_RUNS}"
    [ "${name}" = "no_data" ] || [ "${name}" = "minirocket_cached" ] && MERGE_N_RUNS=1
    MERGE_JOB=$(sbatch --dependency=afterok:${EVAL_JOB} --parsable --export=ARRAY_RESULTS_DIR,RUN_DIR,N_RUNS=${MERGE_N_RUNS} scripts/slurm_merge.sbatch)
    MERGE_JOBS+=("${MERGE_JOB}")
    echo "  merge: ${MERGE_JOB}"
  fi
  echo "  results: ${out_dir}"
}

run_setup "same_kind" "eval_same_kind_${SOURCE}" --sources "${SOURCE}" --seed 42 --train-frac 0.8
run_setup "same_to_multi" "eval_by_source" --by-source
run_setup "multi_to_multi" "eval_multi_to_multi" --seed 42 --train-frac 0.8
run_setup "synthetic_train" "eval_synthetic_train" --seed 42 --train-frac 0.8
run_setup "no_data" "eval_no_data" --test-all
run_setup "minirocket_real" "eval_minirocket_real" --seed 42 --train-frac 0.8
run_setup "tsf_real" "eval_tsf_real" --seed 42 --train-frac 0.8
run_setup "minirocket_cached" "eval_minirocket_cached" --seed 42 --train-frac 0.8

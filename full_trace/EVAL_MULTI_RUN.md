# Collecting full_trace results with multiple runs

This matches the setups from `scripts/run_eval_pipeline.sh` (same idea as `seb_thesis/full_trace/scripts/run_eval_pipeline_all_setups.sh`, but this repo uses `eval_manifest.py` / `make_split.py` and writes **per-method CSVs** instead of `report.md`).

## 1. Run on the cluster with replicates

From the **full_trace repo root** on the cluster (where `data/` and `scripts/` live):

```bash
export N_RUNS=5                    # e.g. 5 seeds: 42, 43, 44, 45, 46
export SEED=42                     # base seed (train uses SEED+i for run i)
export DATA_ROOT=~/path/to/data    # if not default
export SOURCE=TX_OF                # same_kind output dir: eval_same_kind_TX_OF

# Optional: subset of setups (comma-separated)
# export RUN_SETUPS=same_kind,same_to_multi,multi_to_multi,synthetic_train,no_data,minirocket_real,tsf_real,minirocket_cached

bash scripts/run_eval_pipeline.sh
```

**What gets written (per setup directory, e.g. `eval_same_kind_TX_OF/`):**

| Path | Contents |
|------|----------|
| `*.csv` (e.g. `bundle.csv`, `minirocket.csv`) | **Merged across runs**: mean accuracy, mean correct (rounded) per (method, path, experiment_id). |
| `array_results/runs/0/`, `.../1/`, … | Raw `result_*.csv` from each Slurm array task **per seed**. |
| `cache/.../runs/0/` (or `cache_minirocket/...`) | One trained artifact tree per run when `N_RUNS>1`. |

Setups **without** multi-run aggregation (single merge as before): `no_data`, `minirocket_cached` (`N_RUNS` forced to 1 for merge).

**Note:** This pipeline does **not** include `synthetic_from_real` by default (that existed in older tweezcat scripts). Add a `run_setup` line if you need it.

## 2. Pull results to your laptop (rsync)

Use `scripts/pull_eval_results_from_cluster.sh` after editing `JUMP`, `CLUSTER`, and `REMOTE_FULL_TRACE` to your paths.

Default pull **excludes** `cache` and `array_results` (small sync, you keep merged CSVs only).

To also pull **per-run** CSVs for variance/bootstrap analysis:

```bash
INCLUDE_ARRAY_RUNS=1 bash scripts/pull_eval_results_from_cluster.sh
```

## 3. Re-merge locally (optional)

If you pulled `array_results/runs/*` but want to recompute merged CSVs:

```bash
cd /path/to/full_trace
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python scripts/merge_results.py \
  --array-dir ./eval_same_kind_TX_OF/array_results \
  -o ./eval_same_kind_TX_OF \
  --n-runs 5
```

Use the same `--n-runs` as the number of `runs/0`…`runs/N-1` directories present.

## 4. Verify outputs (after jobs finish)

From the full_trace repo root (cluster or local copy):

```bash
bash scripts/check_eval_complete.sh
# Match your pipeline if not defaults:
N_RUNS=5 SOURCE=TX_OF bash scripts/check_eval_complete.sh
```

The script checks `split.json` test count vs `result_*.csv` counts, multi-run `array_results/runs/*`, merged `*.csv` at the setup root, and that `eval_minirocket_real` / `eval_tsf_real` contain `minirocket.csv` / `tsf.csv`. Exit code 1 if any check fails.

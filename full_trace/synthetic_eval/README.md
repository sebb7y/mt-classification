# Synthetic Evaluation Framework

This package evaluates whether synthetic or augmented full-trace datasets remain tethered to real magnetic-tweezers data.

The core idea is simple:

1. Split real experiments at the experiment level.
2. Build or load a synthetic dataset using only training-side information.
3. Convert real and synthetic traces into the same fixed-length representation.
4. Train simple models on synthetic data and test transfer to held-out real experiments.
5. Compare real and synthetic feature distributions.
6. Write a report directory with metrics, statistics, split metadata, and a human-readable `report.md`.

## Quick Smoke Test

From `seb-mt-framework/full_trace`:

```bash
python scripts/run_synthetic_eval.py \
  --data-root ../../seb_thesis/DATA \
  --output-dir synthetic_eval_smoke \
  --max-experiments 4 \
  --max-beads-per-experiment 25 \
  --target-length 500 \
  --synthetic-size 80 \
  --seed 42
```

This uses a small bootstrap generator that resamples real training traces and applies mild perturbations/corruptions. It is not intended as the final generator; it is a baseline target for the evaluator.

## Main Outputs

Each run writes:

- `manifest.csv`: labelled experiments discovered under `--data-root`.
- `split.json`: experiment-level train/test split.
- `bootstrap_synthetic_dataset.npz`: generated only when no `--synthetic-dataset` is supplied.
- `transfer_metrics.csv`: model performance on synthetic and held-out real data.
- `feature_distances_real_train_vs_synthetic.csv`: per-feature real/synthetic distribution gaps.
- `real_vs_synthetic_discriminator.json`: how easily a model can distinguish real from synthetic.
- `report.md`: readable summary.

## Evaluating an External Synthetic Dataset

Pass a fixed-length `.npz` dataset with `traces` or `traces_sktime(_2d)` and `labels`:

```bash
python scripts/run_synthetic_eval.py \
  --data-root ../../seb_thesis/DATA \
  --synthetic-dataset path/to/synthetic_dataset.npz \
  --output-dir synthetic_eval_external \
  --target-length 5000
```

The synthetic trace length must match `--target-length`.

## Current Baseline Models

The first version uses feature-level classifiers:

- random forest
- logistic regression

These are deliberately lightweight. They make the evaluator fast and stable before adding heavier time-series models such as MiniROCKET.

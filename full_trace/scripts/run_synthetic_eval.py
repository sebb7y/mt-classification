#!/usr/bin/env python3
import argparse
import os
import sys

os.environ.setdefault("XDG_CACHE_HOME", os.path.join("/tmp", "synthetic-eval-cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from synthetic_eval.augment import bootstrap_real_synthetic
from synthetic_eval.data import (
    check_no_path_overlap,
    class_counts,
    discover_labelled_experiments,
    load_npz_dataset,
    load_real_dataset,
    save_npz_dataset,
    save_split,
    split_experiments,
    write_manifest_csv,
)
from synthetic_eval.modeling import fit_eval_models, real_vs_synthetic_discriminator, synthetic_train_test_split
from synthetic_eval.report import write_csv, write_json, write_markdown_report
from synthetic_eval.stats import FEATURE_NAMES, distribution_distances, summarize_features, trace_feature_matrix
from synthetic_eval.figures import (
    plot_confusion_matrices,
    plot_feature_histograms,
    plot_feature_pca,
    plot_trace_grid,
    plot_worst_gap_examples,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate how well a synthetic full-trace dataset transfers to held-out real experiments.",
    )
    parser.add_argument(
        "--data-root",
        default=os.path.abspath(os.path.join(REPO_ROOT, "..", "..", "seb_thesis", "DATA")),
        help="root containing real experiment folders (default: ../../seb_thesis/DATA)",
    )
    parser.add_argument("--output-dir", "-o", default="synthetic_eval_run", help="directory for report outputs")
    parser.add_argument("--synthetic-dataset", default=None, help="optional .npz synthetic dataset to evaluate")
    parser.add_argument("--target-length", type=int, default=5000, help="fixed trace length after resampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--sources", default=None, help="comma-separated top-level data sources to include")
    parser.add_argument("--max-experiments", type=int, default=None, help="cap labelled experiments for fast smoke tests")
    parser.add_argument("--max-beads-per-experiment", type=int, default=None, help="cap beads loaded per experiment")
    parser.add_argument("--synthetic-size", type=int, default=None, help="number of bootstrap synthetic samples if no synthetic dataset is supplied")
    parser.add_argument("--corrupt-frac", type=float, default=0.35, help="fraction of bootstrap samples forced bad by corruption")
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    config = vars(args).copy()
    config["data_root"] = data_root
    config["output_dir"] = output_dir
    write_json(os.path.join(output_dir, "config.json"), config)

    rows = discover_labelled_experiments(
        data_root,
        max_depth=args.max_depth,
        sources=args.sources,
        require_both_classes=True,
    )
    if args.max_experiments is not None:
        rows = rows[: args.max_experiments]
    if len(rows) < 2:
        print(f"Need at least 2 labelled experiments with both classes; found {len(rows)}", file=sys.stderr)
        return 1
    write_manifest_csv(os.path.join(output_dir, "manifest.csv"), rows)

    train_rows, test_rows = split_experiments(rows, train_frac=args.train_frac, seed=args.seed)
    check_no_path_overlap(train_rows, test_rows)
    split = save_split(os.path.join(output_dir, "split.json"), train_rows, test_rows, seed=args.seed, train_frac=args.train_frac)

    print(f"Loading real train ({len(train_rows)} experiments) and real test ({len(test_rows)} experiments)...")
    real_train, train_errors = load_real_dataset(
        train_rows,
        target_length=args.target_length,
        max_beads_per_experiment=args.max_beads_per_experiment,
        seed=args.seed,
    )
    real_test, test_errors = load_real_dataset(
        test_rows,
        target_length=args.target_length,
        max_beads_per_experiment=args.max_beads_per_experiment,
        seed=args.seed + 1,
    )
    write_json(os.path.join(output_dir, "load_errors.json"), {"train": train_errors, "test": test_errors})
    if len(real_train["labels"]) == 0 or len(real_test["labels"]) == 0:
        print("Real train or test dataset is empty after loading.", file=sys.stderr)
        return 1

    if args.synthetic_dataset:
        synthetic = load_npz_dataset(os.path.abspath(os.path.expanduser(args.synthetic_dataset)))
        if synthetic["traces"].shape[1] != args.target_length:
            print(
                f"Synthetic trace length {synthetic['traces'].shape[1]} does not match target length {args.target_length}.",
                file=sys.stderr,
            )
            return 1
    else:
        synthetic = bootstrap_real_synthetic(
            real_train,
            n_samples=args.synthetic_size,
            corrupt_frac=args.corrupt_frac,
            seed=args.seed,
        )
        save_npz_dataset(os.path.join(output_dir, "bootstrap_synthetic_dataset.npz"), synthetic)

    print("Computing trace statistics...")
    real_train_features = trace_feature_matrix(real_train["traces"])
    real_test_features = trace_feature_matrix(real_test["traces"])
    synth_features = trace_feature_matrix(synthetic["traces"])
    write_csv(os.path.join(output_dir, "real_train_feature_summary.csv"), summarize_features(real_train_features))
    write_csv(os.path.join(output_dir, "real_test_feature_summary.csv"), summarize_features(real_test_features))
    write_csv(os.path.join(output_dir, "synthetic_feature_summary.csv"), summarize_features(synth_features))
    distance_rows = distribution_distances(real_train_features, synth_features)
    write_csv(os.path.join(output_dir, "feature_distances_real_train_vs_synthetic.csv"), distance_rows)

    print("Running transfer baselines...")
    X_synth_train, X_synth_test, y_synth_train, y_synth_test = synthetic_train_test_split(
        synth_features,
        synthetic["labels"],
        test_size=0.25,
        seed=args.seed,
    )
    eval_sets_for_synth = {
        "synthetic_test": (X_synth_test, y_synth_test),
        "real_test": (real_test_features, real_test["labels"]),
    }
    transfer_rows = []
    for row in fit_eval_models(X_synth_train, y_synth_train, eval_sets_for_synth, seed=args.seed):
        row["train_setup"] = "synthetic_only"
        transfer_rows.append(row)

    eval_sets_for_real = {
        "real_test": (real_test_features, real_test["labels"]),
    }
    for row in fit_eval_models(real_train_features, real_train["labels"], eval_sets_for_real, seed=args.seed):
        row["train_setup"] = "real_only"
        transfer_rows.append(row)

    X_mix = np.vstack([real_train_features, X_synth_train])
    y_mix = np.concatenate([real_train["labels"], y_synth_train])
    for row in fit_eval_models(X_mix, y_mix, eval_sets_for_real, seed=args.seed):
        row["train_setup"] = "real_plus_synthetic"
        transfer_rows.append(row)

    write_csv(os.path.join(output_dir, "transfer_metrics.csv"), transfer_rows)

    discriminator = real_vs_synthetic_discriminator(real_train_features, synth_features, seed=args.seed)
    write_json(os.path.join(output_dir, "real_vs_synthetic_discriminator.json"), discriminator)

    figures_dir = os.path.join(output_dir, "figures")
    figure_paths = []
    print("Writing figures...")
    figure_paths.append(("Feature histograms", plot_feature_histograms(real_train_features, synth_features, FEATURE_NAMES, figures_dir)))
    figure_paths.append(("Feature PCA", plot_feature_pca(real_train_features, real_test_features, synth_features, figures_dir)))
    figure_paths.extend((f"Confusion matrix {i + 1}", p) for i, p in enumerate(plot_confusion_matrices(transfer_rows, figures_dir)))
    figure_paths.append(("Random real vs synthetic traces", plot_trace_grid(real_train, synthetic, figures_dir, seed=args.seed)))
    figure_paths.append(("Worst-gap synthetic examples", plot_worst_gap_examples(synthetic, synth_features, distance_rows, FEATURE_NAMES, figures_dir, seed=args.seed)))
    figures = []
    for label, path in figure_paths:
        if path:
            figures.append({
                "label": label,
                "path": path,
                "relative_path": os.path.relpath(path, output_dir),
            })
    write_json(os.path.join(output_dir, "figures.json"), figures)

    data_summary = {
        "n_experiments_total": len(rows),
        "n_train_experiments": len(train_rows),
        "n_test_experiments": len(test_rows),
        "real_train_counts": class_counts(real_train["labels"]),
        "real_test_counts": class_counts(real_test["labels"]),
        "synthetic_counts": class_counts(synthetic["labels"]),
        "load_errors_train": len(train_errors),
        "load_errors_test": len(test_errors),
    }
    write_json(os.path.join(output_dir, "data_summary.json"), data_summary)
    write_markdown_report(
        os.path.join(output_dir, "report.md"),
        config=config,
        data_summary=data_summary,
        transfer_rows=transfer_rows,
        distance_rows=distance_rows,
        discriminator=discriminator,
        figures=figures,
    )

    print(f"Wrote synthetic evaluation report to {os.path.join(output_dir, 'report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

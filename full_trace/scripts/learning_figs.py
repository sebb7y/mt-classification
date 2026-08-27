import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from trace_gen import generate_training_dataset, save_dataset, load_dataset
from train_models import create_model, transform_dataset_sktime, transform_dataset, generate_random_kernels, SKTIME_AVAILABLE, SKTIME_CLASSIFIERS, SKTIME_TRANSFORMERS

def model_list(n_kernels, exclude=None):
    exclude = set(exclude or [])
    models = []
    if "minirocket" in SKTIME_TRANSFORMERS and "minirocket" not in exclude:
        models.append(("minirocket", "MiniROCKET"))
    if "tsf" in SKTIME_CLASSIFIERS and "tsf" not in exclude:
        models.append(("tsf", "TimeSeriesForest"))
    if "rocket" in SKTIME_TRANSFORMERS and "rocket" not in exclude:
        models.append(("rocket", "ROCKET"))
    if "weasel" in SKTIME_CLASSIFIERS and "weasel" not in exclude:
        models.append(("weasel", "WEASEL"))
    if "boss" in SKTIME_CLASSIFIERS and "boss" not in exclude:
        models.append(("boss", "BOSS"))
    if "bagging" in SKTIME_CLASSIFIERS and "bagging" not in exclude:
        models.append(("bagging", "Bagging"))
    if "weighted_ensemble" in SKTIME_CLASSIFIERS and "weighted_ensemble" not in exclude:
        models.append(("weighted_ensemble", "WeightedEnsemble"))
    if ("hivecotev2" in SKTIME_CLASSIFIERS or "hivecote" in SKTIME_CLASSIFIERS) and "hivecotev2" not in exclude:
        models.append(("hivecotev2", "HIVECOTEV2"))
    if "hivecotev1" in SKTIME_CLASSIFIERS and "hivecotev1" not in exclude:
        models.append(("hivecotev1", "HIVECOTEV1"))
    if "custom" not in exclude:
        models.append(("custom", "Custom ROCKET"))
    return models

def data(n_traces, test_ratio, seed, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"dataset_n{n_traces}_seed{seed}.npz")
    if os.path.exists(path):
        dataset = load_dataset(path, format="npz_padded")
    else:
        dataset = generate_training_dataset(
            n_traces=n_traces,
            usable_ratio=0.5,
            total_time_range=(100.0, 300.0),
            dt_range=(0.01, 0.05),
            noise_std_range=(5.0, 20.0),
            base_seed=seed,
        )
        save_dataset(dataset, path, format="npz_padded")
        dataset = load_dataset(path, format="npz_padded")
    if "traces_sktime" not in dataset:
        raise ValueError("dataset must have traces_sktime (npz_padded)")
    X = np.asarray(dataset["traces_sktime"])
    y = np.asarray(dataset["labels"], dtype=np.intp)
    n_test = int(len(X) * test_ratio)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X_train = X[idx[n_test:]]
    y_train = y[idx[n_test:]]
    X_test = X[idx[:n_test]]
    y_test = y[idx[:n_test]]
    return X_train, y_train, X_test, y_test

def fit_evaluate_subset(model_type, X_train, y_train, X_test, y_test, n_train_use, n_kernels, seed):
    X_sub = X_train[:n_train_use]
    y_sub = y_train[:n_train_use]
    t0 = time.time()
    model_type_lower = model_type.lower()
    if model_type_lower in ("tsf", "boss", "weasel", "bagging", "weighted_ensemble", "hivecotev2", "hivecotev1"):
        model = create_model(model_type, seed=seed)
        model.fit(X_sub, y_sub)
        train_acc = accuracy_score(y_sub, model.predict(X_sub))
        test_acc = accuracy_score(y_test, model.predict(X_test))
    elif model_type_lower == "custom":
        kernels = generate_random_kernels(n_kernels=n_kernels, seed=seed)
        X_sub_2d = X_sub[:, 0, :]
        X_test_2d = X_test[:, 0, :]
        X_sub_feat = transform_dataset(X_sub_2d, kernels, n_workers=None)
        X_test_feat = transform_dataset(X_test_2d, kernels, n_workers=None)
        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub_feat)
        X_test_scaled = scaler.transform(X_test_feat)
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        clf.fit(X_sub_scaled, y_sub)
        train_acc = accuracy_score(y_sub, clf.predict(X_sub_scaled))
        test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    elif model_type_lower in ("minirocket", "rocket"):
        transformer = create_model(
            model_type, model_params={"num_kernels": n_kernels}, seed=seed
        )
        X_sub_2d = X_sub[:, 0, :]
        X_test_2d = X_test[:, 0, :]
        X_sub_feat = transform_dataset_sktime(
            X_sub_2d, transformer, fit=True
        )
        X_test_feat = transform_dataset_sktime(
            X_test_2d, transformer, fit=False
        )
        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub_feat)
        X_test_scaled = scaler.transform(X_test_feat)
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        clf.fit(X_sub_scaled, y_sub)
        train_acc = accuracy_score(y_sub, clf.predict(X_sub_scaled))
        test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    else:
        raise ValueError(f"unknown model_type for subset fit: {model_type}")
    fit_time = time.time() - t0
    return train_acc, test_acc, fit_time

def fit_full_model(model_type, dataset_path, n_traces, n_kernels, test_ratio, seed):
    from train_models import train_model
    res = train_model(
        model_type=model_type,
        n_traces=n_traces,
        n_kernels=n_kernels,
        test_ratio=test_ratio,
        use_existing_dataset=dataset_path,
        seed=seed,
        force_regenerate=False,
    )
    test_acc = res["results"]["test_accuracy"]
    rt = res.get("runtime_timings") or {}
    train_time = rt.get("overall_runtime", 0.0)
    return test_acc, train_time

def main():
    parser = argparse.ArgumentParser(
        description="small-dataset model comparison and learning curves"
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=400,
        help="number of traces (small so all models finish in ~1 min)",
    )
    parser.add_argument(
        "--n-kernels",
        type=int,
        default=500,
        help="kernels for ROCKET/MiniROCKET",
    )
    parser.add_argument(
        "--curve-steps",
        type=int,
        default=5,
        help="number of learning-curve points (fractions of train set)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory (default: tweezcat repo root)",
    )
    parser.add_argument(
        "--dataset-cache",
        type=str,
        default=None,
        help="directory for dataset cache (default: dataset_cache under repo)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=(),
        metavar="MODEL",
        help="model types to skip (e.g. --exclude hivecotev2 hivecotev1 for faster runs)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or REPO_ROOT
    cache_dir = args.dataset_cache or os.path.join(REPO_ROOT, "dataset_cache")
    os.makedirs(out_dir, exist_ok=True)

    if not SKTIME_AVAILABLE:
        print("error: sktime required", file=sys.stderr)
        sys.exit(1)

    models = model_list(args.n_kernels, exclude=args.exclude)
    print(f"  models: {[m[1] for m in models]}")

    print("loading/generating small dataset...")
    X_train, y_train, X_test, y_test = data(
        args.n_traces, args.test_ratio, args.seed, cache_dir
    )
    dataset_path = os.path.join(cache_dir, f"dataset_n{args.n_traces}_seed{args.seed}.npz")
    n_train = len(X_train)
    print(f"  train: {n_train}, test: {len(X_test)}")

    results = []
    for model_type, display_name in models:
        print(f"  running {display_name} (full)...", end=" ", flush=True)
        try:
            test_acc, train_time = fit_full_model(
                model_type, dataset_path, args.n_traces, args.n_kernels,
                args.test_ratio, args.seed
            )
            results.append({
                "model_type": model_type,
                "display_name": display_name,
                "test_acc": test_acc,
                "train_time": train_time,
                "error": None
            })
            print(f"acc={test_acc:.3f} time={train_time:.1f}s")
        except Exception as e:
            print(f"failed: {e}")
            results.append({
                "model_type": model_type,
                "display_name": display_name,
                "test_acc": None,
                "train_time": None,
                "error": str(e)
            })

    fractions = np.linspace(0.2, 1.0, args.curve_steps)
    curve_data = {}
    for model_type, display_name in models:
        if any(r["model_type"] == model_type and r["error"] for r in results):
            continue
        print(f"  learning curve: {display_name}...", end=" ", flush=True)
        train_sizes = []
        train_accs = []
        test_accs = []
        for frac in fractions:
            n_use = max(1, int(n_train * frac))
            train_acc, test_acc, _ = fit_evaluate_subset(
                model_type, X_train, y_train, X_test, y_test,
                n_use, args.n_kernels, args.seed,
            )
            train_sizes.append(n_use)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        curve_data[display_name] = {
            "train_sizes": train_sizes,
            "train_accs": train_accs,
            "test_accs": test_accs
        }
        print("done")

    valid = [r for r in results if r["error"] is None]
    if not valid:
        print("no successful runs to plot")
        return
    names = [r["display_name"] for r in valid]
    accs = [r["test_acc"] for r in valid]
    times = [r["train_time"] for r in valid]
    y = np.arange(len(names))
    height = 0.38
    colour_acc = "#1a5276"
    colour_time = "#922b21"
    fig, ax1 = plt.subplots(figsize=(8, max(5, len(names) * 0.5)))
    bars1 = ax1.barh(y - height / 2, accs, height=height, color=colour_acc, label="Test accuracy")
    ax1.set_xlabel("Test accuracy", color=colour_acc)
    ax1.set_xlim(0, 1.05)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.tick_params(axis="x", labelcolor=colour_acc)
    ax1.invert_yaxis()
    ax2 = ax1.twiny()
    bars2 = ax2.barh(y + height / 2, times, height=height, color=colour_time, alpha=0.9, label="Train time (s)")
    ax2.set_xlabel("Train time (s)", color=colour_time)
    ax2.set_xlim(0, max(times) * 1.08 if times else 1)
    ax2.tick_params(axis="x", labelcolor=colour_time)
    fig.legend([bars1, bars2], ["Test accuracy", "Train time (s)"], loc="lower right")
    fig.suptitle(f"Time-series classification: performance and training time (synthetic, {args.n_traces:,} traces)", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "accuracy_and_time_small.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {fig_path}")

    colour_train = "#1a5276"
    colour_test = "#922b21"
    n_curves = len(curve_data)
    if n_curves > 0:
        learning_curves_dir = os.path.join(out_dir, "learning_curves")
        os.makedirs(learning_curves_dir, exist_ok=True)

        def save_one_curve(display_name, data, path):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(
                data["train_sizes"],
                data["train_accs"],
                "o-",
                label="Train accuracy",
                linewidth=2,
                markersize=6,
                color=colour_train,
            )
            ax.plot(
                data["train_sizes"],
                data["test_accs"],
                "s-",
                label="Test accuracy",
                linewidth=2,
                markersize=6,
                color=colour_test,
            )
            ax.set_xlabel("Training set size")
            ax.set_ylabel("Accuracy")
            ax.set_title(display_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.3, 1.05)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()

        for display_name, data in curve_data.items():
            safe_name = display_name.replace(" ", "_")
            one_path = os.path.join(learning_curves_dir, f"{safe_name}.png")
            save_one_curve(display_name, data, one_path)
            print(f"saved: {one_path}")

        n_cols = 2
        n_rows = (n_curves + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        for idx, (display_name, data) in enumerate(curve_data.items()):
            r, c = idx // n_cols, idx % n_cols
            ax = axes[r, c]
            ax.plot(
                data["train_sizes"],
                data["train_accs"],
                "o-",
                label="Train accuracy",
                linewidth=2,
                markersize=6,
                color=colour_train,
            )
            ax.plot(
                data["train_sizes"],
                data["test_accs"],
                "s-",
                label="Test accuracy",
                linewidth=2,
                markersize=6,
                color=colour_test,
            )
            ax.set_xlabel("Training set size")
            ax.set_ylabel("Accuracy")
            ax.set_title(display_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.3, 1.05)
        for idx in range(len(curve_data), n_rows * n_cols):
            r, c = idx // n_cols, idx % n_cols
            axes[r, c].set_visible(False)
        plt.suptitle(f"learning curves {args.n_traces} traces, {args.curve_steps} steps (seed {args.seed})")
        plt.tight_layout()
        curve_path = os.path.join(learning_curves_dir, "learning_curves_small.png")
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved: {curve_path}")

if __name__ == "__main__":
    main()

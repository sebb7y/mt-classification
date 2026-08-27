import os

import numpy as np


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _safe_name(name):
    return str(name).replace("/", "_").replace(" ", "_")


def plot_feature_histograms(real_features, synth_features, feature_names, output_dir, max_cols=3):
    plt = _setup_matplotlib()
    os.makedirs(output_dir, exist_ok=True)
    n_features = len(feature_names)
    n_cols = max_cols
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows))
    axes = np.asarray(axes).ravel()
    for j, name in enumerate(feature_names):
        ax = axes[j]
        r = np.asarray(real_features[:, j], dtype=np.float64)
        s = np.asarray(synth_features[:, j], dtype=np.float64)
        finite = np.isfinite(np.concatenate([r, s]))
        if not np.any(finite):
            ax.set_title(name)
            ax.text(0.5, 0.5, "no finite values", ha="center", va="center")
            continue
        vals = np.concatenate([r[np.isfinite(r)], s[np.isfinite(s)]])
        lo, hi = np.quantile(vals, [0.01, 0.99])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.min(vals), np.max(vals)
        bins = np.linspace(lo, hi, 35) if lo != hi else 20
        ax.hist(r[np.isfinite(r)], bins=bins, alpha=0.55, density=True, label="real train", color="#2f6f9f")
        ax.hist(s[np.isfinite(s)], bins=bins, alpha=0.55, density=True, label="synthetic", color="#d17a22")
        ax.set_title(name)
        ax.tick_params(axis="x", labelrotation=25)
    for ax in axes[n_features:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(output_dir, "feature_histograms.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_feature_pca(real_train_features, real_test_features, synth_features, output_dir):
    plt = _setup_matplotlib()
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    os.makedirs(output_dir, exist_ok=True)
    parts = []
    labels = []
    colors = []
    for name, X, color in [
        ("real_train", real_train_features, "#2f6f9f"),
        ("real_test", real_test_features, "#3a9d57"),
        ("synthetic", synth_features, "#d17a22"),
    ]:
        X = np.asarray(X, dtype=np.float64)
        if len(X):
            parts.append(X)
            labels.extend([name] * len(X))
            colors.extend([color] * len(X))
    if not parts or sum(len(p) for p in parts) < 3:
        return None
    X_all = np.vstack(parts)
    X_scaled = StandardScaler().fit_transform(np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0))
    emb = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    labels = np.asarray(labels)
    colors = np.asarray(colors)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for name in ["real_train", "real_test", "synthetic"]:
        mask = labels == name
        if np.any(mask):
            ax.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=0.72, label=name, c=colors[mask])
    ax.set_title("Feature PCA: real vs synthetic")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, "feature_pca_real_vs_synthetic.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_confusion_matrices(transfer_rows, output_dir):
    plt = _setup_matplotlib()
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    rows = [r for r in transfer_rows if r.get("eval_set") == "real_test"]
    if not rows:
        rows = transfer_rows
    for row in rows:
        matrix = np.array([[row.get("tn", 0), row.get("fp", 0)], [row.get("fn", 0), row.get("tp", 0)]], dtype=float)
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], labels=["pred bad", "pred good"])
        ax.set_yticks([0, 1], labels=["true bad", "true good"])
        ax.set_title(f"{row.get('train_setup', '')} / {row.get('model', '')} / {row.get('eval_set', '')}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        filename = f"confusion_{_safe_name(row.get('train_setup', ''))}_{_safe_name(row.get('model', ''))}_{_safe_name(row.get('eval_set', ''))}.png"
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_trace_grid(real_dataset, synthetic_dataset, output_dir, seed=42, n_each=6):
    plt = _setup_matplotlib()
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    real_traces = np.asarray(real_dataset["traces"], dtype=np.float64)
    real_labels = np.asarray(real_dataset["labels"], dtype=np.intp)
    synth_traces = np.asarray(synthetic_dataset["traces"], dtype=np.float64)
    synth_labels = np.asarray(synthetic_dataset["labels"], dtype=np.intp)
    n_real = min(n_each, len(real_traces))
    n_synth = min(n_each, len(synth_traces))
    real_idx = rng.choice(np.arange(len(real_traces)), size=n_real, replace=False) if n_real else []
    synth_idx = rng.choice(np.arange(len(synth_traces)), size=n_synth, replace=False) if n_synth else []
    n_rows = max(n_real, n_synth)
    if n_rows == 0:
        return None
    fig, axes = plt.subplots(n_rows, 2, figsize=(11.0, 2.1 * n_rows), sharex=False)
    if n_rows == 1:
        axes = np.asarray([axes])
    for r in range(n_rows):
        for c in range(2):
            axes[r, c].axis("off")
    for r, idx in enumerate(real_idx):
        ax = axes[r, 0]
        ax.axis("on")
        ax.plot(real_traces[idx], linewidth=0.8, color="#2f6f9f")
        ax.set_title(f"real label={int(real_labels[idx])}")
        ax.set_xticks([])
    for r, idx in enumerate(synth_idx):
        ax = axes[r, 1]
        ax.axis("on")
        ax.plot(synth_traces[idx], linewidth=0.8, color="#d17a22")
        ax.set_title(f"synthetic label={int(synth_labels[idx])}")
        ax.set_xticks([])
    axes[0, 0].set_ylabel("z")
    fig.suptitle("Random Real vs Synthetic Traces", y=0.995)
    fig.tight_layout()
    path = os.path.join(output_dir, "trace_grid_real_vs_synthetic.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_worst_gap_examples(synthetic_dataset, synth_features, distance_rows, feature_names, output_dir, seed=42, n_examples=8):
    plt = _setup_matplotlib()
    os.makedirs(output_dir, exist_ok=True)
    if not distance_rows or len(synth_features) == 0:
        return None
    worst = max(distance_rows, key=lambda r: r.get("ks_stat") if r.get("ks_stat") == r.get("ks_stat") else -1)
    feature = worst.get("feature")
    if feature not in feature_names:
        return None
    j = feature_names.index(feature)
    vals = np.asarray(synth_features[:, j], dtype=np.float64)
    center = np.nanmedian(vals)
    order = np.argsort(np.abs(vals - center))[::-1]
    idx = order[: min(n_examples, len(order))]
    traces = np.asarray(synthetic_dataset["traces"], dtype=np.float64)
    labels = np.asarray(synthetic_dataset["labels"], dtype=np.intp)
    n = len(idx)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.5, 2.4 * n_rows))
    axes = np.asarray(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for k, sample_idx in enumerate(idx):
        ax = axes[k]
        ax.plot(traces[sample_idx], linewidth=0.8, color="#d17a22")
        ax.set_title(f"idx={int(sample_idx)} label={int(labels[sample_idx])} {feature}={vals[sample_idx]:.3g}")
        ax.set_xticks([])
    fig.suptitle(f"Synthetic Examples Extreme on `{feature}`", y=0.995)
    fig.tight_layout()
    path = os.path.join(output_dir, "worst_gap_synthetic_examples.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


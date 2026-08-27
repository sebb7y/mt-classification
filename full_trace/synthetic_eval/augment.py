import numpy as np


def _fill_trace(trace):
    trace = np.asarray(trace, dtype=np.float64)
    finite = np.isfinite(trace)
    fill = float(np.mean(trace[finite])) if np.any(finite) else 0.0
    return np.nan_to_num(trace, nan=fill, posinf=fill, neginf=fill)


def mild_real_augmentation(trace, rng):
    trace = _fill_trace(trace)
    out = trace.copy()
    n = len(out)
    if n == 0:
        return out

    center = np.median(out)
    out = center + out * rng.uniform(0.9, 1.1) - center * rng.uniform(0.9, 1.1)

    diff = np.diff(out)
    noise_std = np.nanstd(diff) if len(diff) else np.nanstd(out)
    if not np.isfinite(noise_std) or noise_std <= 0:
        noise_std = max(np.nanstd(out) * 0.02, 1e-6)
    out = out + rng.normal(0.0, noise_std * rng.uniform(0.05, 0.25), size=n)

    drift_scale = np.nanstd(out) * rng.uniform(-0.05, 0.05)
    if np.isfinite(drift_scale) and drift_scale != 0:
        out = out + np.linspace(0.0, drift_scale, n)
    return out


def corrupt_trace(trace, rng):
    trace = mild_real_augmentation(trace, rng)
    n = len(trace)
    if n < 4:
        return trace
    out = trace.copy()
    mode = rng.choice(["high_noise", "dropout", "spike", "lost_tail", "flat_region", "curved_drift"])
    scale = np.nanstd(out)
    if not np.isfinite(scale) or scale <= 0:
        scale = max(np.nanmedian(np.abs(np.diff(out))) if n > 1 else 1.0, 1e-6)

    if mode == "high_noise":
        start = int(rng.integers(0, max(1, n // 2)))
        length = int(rng.integers(max(2, n // 20), max(3, n // 3)))
        end = min(n, start + length)
        out[start:end] += rng.normal(0.0, scale * rng.uniform(1.5, 5.0), size=end - start)
    elif mode == "dropout":
        start = int(rng.integers(0, max(1, n - 2)))
        length = int(rng.integers(max(2, n // 50), max(3, n // 8)))
        end = min(n, start + length)
        fill = out[start - 1] if start > 0 else np.nanmedian(out)
        out[start:end] = fill
    elif mode == "spike":
        k = int(rng.integers(1, max(2, n // 100)))
        idx = rng.choice(np.arange(n), size=min(k, n), replace=False)
        out[idx] += rng.choice([-1.0, 1.0], size=len(idx)) * scale * rng.uniform(5.0, 20.0)
    elif mode == "lost_tail":
        start = int(rng.integers(max(1, n // 4), max(2, n - 1)))
        lo = np.nanmin(out) - scale * rng.uniform(3.0, 8.0)
        hi = np.nanmax(out) + scale * rng.uniform(3.0, 8.0)
        out[start:] = rng.uniform(lo, hi, size=n - start)
    elif mode == "flat_region":
        start = int(rng.integers(0, max(1, n // 2)))
        length = int(rng.integers(max(2, n // 10), max(3, n // 2)))
        end = min(n, start + length)
        out[start:end] = np.nanmedian(out[start:end]) if end > start else np.nanmedian(out)
    elif mode == "curved_drift":
        x = np.linspace(-1.0, 1.0, n)
        out = out + (x * x - np.mean(x * x)) * scale * rng.uniform(0.5, 2.0)
    return out


def bootstrap_real_synthetic(real_dataset, n_samples=None, corrupt_frac=0.35, seed=42):
    traces = np.asarray(real_dataset["traces"], dtype=np.float64)
    labels = np.asarray(real_dataset["labels"], dtype=np.intp)
    metadata = real_dataset.get("metadata", [])
    if len(traces) == 0:
        raise ValueError("cannot bootstrap synthetic data from an empty real dataset")
    rng = np.random.default_rng(seed)
    if n_samples is None:
        n_samples = len(traces)
    idx = rng.choice(np.arange(len(traces)), size=n_samples, replace=True)
    out_traces = np.zeros((n_samples, traces.shape[1]), dtype=np.float64)
    out_labels = np.zeros(n_samples, dtype=np.intp)
    out_meta = []
    for out_i, src_i in enumerate(idx):
        src_label = int(labels[src_i])
        force_corrupt = rng.random() < corrupt_frac
        if force_corrupt:
            out_traces[out_i] = corrupt_trace(traces[src_i], rng)
            out_labels[out_i] = 0
            aug = "corrupt_real"
        else:
            out_traces[out_i] = mild_real_augmentation(traces[src_i], rng)
            out_labels[out_i] = src_label
            aug = "mild_real_resample"
        parent = metadata[src_i] if src_i < len(metadata) and isinstance(metadata[src_i], dict) else {}
        out_meta.append({
            "source": "synthetic_bootstrap",
            "augmentation": aug,
            "parent_index": int(src_i),
            "parent_label": src_label,
            "parent_experiment_id": parent.get("experiment_id", ""),
            "parent_source": parent.get("experiment_source", ""),
            "parent_bead_id": parent.get("bead_id", None),
        })
    return {
        "traces": out_traces,
        "labels": out_labels,
        "metadata": out_meta,
        "target_length": traces.shape[1],
    }


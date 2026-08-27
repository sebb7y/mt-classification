import numpy as np


FEATURE_NAMES = [
    "mean",
    "std",
    "range",
    "diff_std",
    "max_abs_diff",
    "median_abs_diff",
    "q01",
    "q99",
    "autocorr_lag1",
    "flat_fraction",
    "large_jump_fraction",
    "slope",
]


def _fill_trace(trace):
    trace = np.asarray(trace, dtype=np.float64)
    finite = np.isfinite(trace)
    fill = float(np.mean(trace[finite])) if np.any(finite) else 0.0
    return np.nan_to_num(trace, nan=fill, posinf=fill, neginf=fill)


def trace_feature_matrix(traces):
    traces = np.asarray(traces, dtype=np.float64)
    if traces.ndim == 3 and traces.shape[1] == 1:
        traces = traces[:, 0, :]
    out = np.zeros((len(traces), len(FEATURE_NAMES)), dtype=np.float64)
    for i, raw in enumerate(traces):
        trace = _fill_trace(raw)
        diff = np.diff(trace)
        abs_diff = np.abs(diff)
        std = float(np.std(trace))
        diff_std = float(np.std(diff)) if len(diff) else 0.0
        autocorr = 0.0
        if len(trace) > 2 and std > 1e-12:
            corr = np.corrcoef(trace[:-1], trace[1:])[0, 1]
            autocorr = float(corr) if np.isfinite(corr) else 0.0
        flat_fraction = float(np.mean(abs_diff < max(diff_std * 0.02, 1e-12))) if len(abs_diff) else 0.0
        large_jump_fraction = float(np.mean(abs_diff > max(diff_std * 6.0, 1e-12))) if len(abs_diff) else 0.0
        x = np.linspace(-0.5, 0.5, len(trace))
        slope = float(np.polyfit(x, trace, 1)[0]) if len(trace) > 1 else 0.0
        out[i] = [
            float(np.mean(trace)),
            std,
            float(np.ptp(trace)),
            diff_std,
            float(np.max(abs_diff)) if len(abs_diff) else 0.0,
            float(np.median(abs_diff)) if len(abs_diff) else 0.0,
            float(np.quantile(trace, 0.01)),
            float(np.quantile(trace, 0.99)),
            autocorr,
            flat_fraction,
            large_jump_fraction,
            slope,
        ]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def summarize_features(features):
    features = np.asarray(features, dtype=np.float64)
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        col = features[:, j] if len(features) else np.array([], dtype=np.float64)
        if len(col) == 0:
            rows.append({"feature": name, "mean": np.nan, "std": np.nan, "median": np.nan, "q05": np.nan, "q95": np.nan})
        else:
            rows.append({
                "feature": name,
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "median": float(np.median(col)),
                "q05": float(np.quantile(col, 0.05)),
                "q95": float(np.quantile(col, 0.95)),
            })
    return rows


def distribution_distances(real_features, synth_features):
    try:
        from scipy.stats import ks_2samp, wasserstein_distance
    except Exception:
        ks_2samp = None
        wasserstein_distance = None

    real_features = np.asarray(real_features, dtype=np.float64)
    synth_features = np.asarray(synth_features, dtype=np.float64)
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        r = real_features[:, j] if len(real_features) else np.array([], dtype=np.float64)
        s = synth_features[:, j] if len(synth_features) else np.array([], dtype=np.float64)
        row = {"feature": name}
        if len(r) and len(s):
            row["real_median"] = float(np.median(r))
            row["synthetic_median"] = float(np.median(s))
            row["median_delta"] = float(np.median(s) - np.median(r))
            if ks_2samp is not None:
                row["ks_stat"] = float(ks_2samp(r, s).statistic)
            else:
                row["ks_stat"] = np.nan
            if wasserstein_distance is not None:
                row["wasserstein"] = float(wasserstein_distance(r, s))
            else:
                row["wasserstein"] = np.nan
        else:
            row.update({"real_median": np.nan, "synthetic_median": np.nan, "median_delta": np.nan, "ks_stat": np.nan, "wasserstein": np.nan})
        rows.append(row)
    return rows


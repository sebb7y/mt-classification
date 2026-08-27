import numpy as np


def ensure_2d_traces(traces):
    traces = np.asarray(traces, dtype=np.float64)
    if traces.ndim == 3 and traces.shape[1] == 1:
        return traces[:, 0, :]
    return traces


def fill_trace(trace):
    finite = np.isfinite(trace)
    fill_value = float(np.mean(trace[finite])) if np.any(finite) else 0.0
    return np.nan_to_num(trace, nan=fill_value, posinf=0.0, neginf=0.0)


def sanitize_feature_matrix(features):
    return np.nan_to_num(np.asarray(features, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def extract_feats4(traces):
    traces_2d = ensure_2d_traces(traces)
    n_traces = traces_2d.shape[0]
    out = np.zeros((n_traces, 4), dtype=np.float64)
    for i in range(n_traces):
        trace = fill_trace(traces_2d[i])
        diff = np.diff(trace)
        abs_diff = np.abs(diff)
        out[i, 0] = np.var(trace)
        out[i, 1] = np.std(diff) if len(diff) else 0.0
        out[i, 2] = np.max(abs_diff) if len(abs_diff) else 0.0
        out[i, 3] = np.median(abs_diff) if len(abs_diff) else 0.0
    return out


def extract_feats_extra(traces):
    traces_2d = ensure_2d_traces(traces)
    n_traces = traces_2d.shape[0]
    out = np.zeros((n_traces, 4), dtype=np.float64)
    for i in range(n_traces):
        trace = fill_trace(traces_2d[i])
        autocorr = 0.0
        if len(trace) > 1 and np.std(trace) > 1e-12:
            corr = np.corrcoef(trace[:-1], trace[1:])[0, 1]
            if np.isfinite(corr):
                autocorr = float(corr)
        out[i] = [
            np.ptp(trace),
            np.percentile(trace, 25),
            np.percentile(trace, 75),
            autocorr,
        ]
    return out


def extract_feats_full(traces):
    return np.column_stack([extract_feats4(traces), extract_feats_extra(traces)])

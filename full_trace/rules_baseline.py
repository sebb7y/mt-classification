import numpy as np

def variance_above_threshold(trace, thresh):
    t = np.asarray(trace, dtype=np.float64)
    t = np.nan_to_num(t, nan=np.nanmean(t), posinf=0.0, neginf=0.0)
    return np.var(t) > thresh

def has_spike(trace, n_sigma=5.0, min_jump=None):
    t = np.asarray(trace, dtype=np.float64)
    t = np.nan_to_num(t, nan=np.nanmean(t), posinf=0.0, neginf=0.0)
    d = np.diff(t)
    if len(d) == 0:
        return False
    std_d = np.std(d)
    if std_d <= 0:
        std_d = 1e-10  # avoiding division by 0
    threshold = n_sigma * std_d
    if min_jump is not None:
        threshold = max(threshold, min_jump)
    return np.any(np.abs(d) > threshold)

def has_dropout(trace, gap_len=5, tol=1e-06):
    # aka flat trace detection
    t = np.asarray(trace, dtype=np.float64)
    t = np.nan_to_num(t, nan=np.nanmean(t), posinf=0.0, neginf=0.0)
    d = np.abs(np.diff(t))
    if len(d) < gap_len:
        return False
    run = 0
    for i in range(len(d)):
        if d[i] <= tol:
            run += 1
            if run >= gap_len:
                return True
        else:
            run = 0
    return False

def is_flat(trace, tol=1e-06):
    t = np.asarray(trace, dtype=np.float64)
    t = np.nan_to_num(t, nan=np.nanmean(t), posinf=0.0, neginf=0.0)
    return np.std(t) <= tol

def classify_trace(
        trace, variance_thresh=None, spike_n_sigma=5.0, spike_min_jump=None,
        dropout_gap_len=5, dropout_tol=1e-06, flat_tol=1e-06):
    t = np.asarray(trace, dtype=np.float64)
    t = np.nan_to_num(t, nan=np.nanmean(t), posinf=0.0, neginf=0.0)
    if is_flat(t, tol=flat_tol):
        return 0
    if has_dropout(t, gap_len=dropout_gap_len, tol=dropout_tol):
        return 0
    if has_spike(t, n_sigma=spike_n_sigma, min_jump=spike_min_jump):
        return 0
    if variance_thresh is not None and variance_above_threshold(t, variance_thresh):
        return 0
    return 1

def classify_traces(traces, **kwargs):
    n = traces.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        out[i] = classify_trace(traces[i], **kwargs)
    return out

def main():
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from experimental_loader import prep_exp_traces_for_model

    parser = argparse.ArgumentParser(description="rules baseline: classify experimental beads as usable/unusable")
    parser.add_argument("--exp-folder", required=True, help="experiment folder (config.yaml, traces.npy, optional pickle)")
    parser.add_argument("--target-length", type=int, default=5000, help="resample each trace to this length (default 5000)")
    parser.add_argument("--variance-thresh", type=float, default=None, help="variance above this -> unusable (default: no variance rule)")
    parser.add_argument("--spike-n-sigma", type=float, default=5.0, help="spike threshold in sigma of diff (default 5)")
    args = parser.parse_args()

    data = prep_exp_traces_for_model(
        args.exp_folder,
        good_only=False,
        target_length=args.target_length,
        use_resample=True,
    )
    traces = data["traces"]
    labels = data["labels"]
    if traces.size == 0:
        print("no traces loaded", file=sys.stderr)
        sys.exit(1)

    pred = classify_traces(
        traces,
        variance_thresh=args.variance_thresh,
        spike_n_sigma=args.spike_n_sigma,
    )
    acc = np.mean(pred == labels)
    print(f"rules baseline: accuracy vs good/bad labels = {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Labels: good=1: {(labels == 1).sum()}, bad=0: {(labels == 0).sum()}")
    print(f"  Predictions: Class 0: {(pred == 0).sum()}, Class 1: {(pred == 1).sum()}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

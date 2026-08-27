import argparse
import json
import os
import sys
import time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from experimental_loader import resample_trace, split_trace_into_segments
from features import extract_feats4, sanitize_feature_matrix
from trace_gen import load_dataset

def aggr_pred_trace(y_pred, trace_id_per_segment, test_trace_indices):
    trace_pred = np.zeros(len(test_trace_indices), dtype=np.int64)
    for i, t in enumerate(test_trace_indices):
        mask = trace_id_per_segment == t
        if not np.any(mask):
            continue
        votes = y_pred[mask]
        trace_pred[i] = int(np.bincount(votes).argmax())
    return trace_pred

def main():
    parser = argparse.ArgumentParser(
        description="run MiniROCKET and trivial on synthetic data, print scaled and split accuracies",
    )
    parser.add_argument("--dataset", required=True, help="path to synthetic .npz (e.g. dataset_cache/dataset_n10000_seed42.npz)")
    parser.add_argument("-o", "--output", default=None, help="write results json here (optional)")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segment-length", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=2500)
    parser.add_argument(
        "--split-eval-length",
        type=int,
        default=None,
        help="resample each trace to this length before splitting (for split eval only)",
    )
    parser.add_argument("--save-models-to", default=None, help="if set, save all four models here for eval on real (eval_synthetic_models_on_real.py)")
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    dataset = load_dataset(args.dataset, format="npz_padded")
    if "traces_sktime" not in dataset:
        print("dataset must contain traces_sktime (fixed-length)", file=sys.stderr)
        return 1
    traces_sktime = np.asarray(dataset["traces_sktime"])
    labels = np.asarray(dataset["labels"], dtype=np.intp)
    n_traces = len(labels)
    T = traces_sktime.shape[2]
    n_test = max(1, int(n_traces * args.test_ratio))
    idx = np.arange(n_traces)
    idx_0 = idx[labels == 0]
    idx_1 = idx[labels == 1]
    rng.shuffle(idx_0)
    rng.shuffle(idx_1)
    n0_test = max(0, int(len(idx_0) * args.test_ratio))
    n1_test = max(0, int(len(idx_1) * args.test_ratio))
    test_idx = np.concatenate([idx_0[:n0_test], idx_1[:n1_test]])
    rng.shuffle(test_idx)
    train_idx = np.setdiff1d(idx, test_idx)
    X_train_scaled = traces_sktime[train_idx]
    X_test_scaled = traces_sktime[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    results = {}
    save_base = os.path.expanduser(args.save_models_to) if args.save_models_to else None

    try:
        from train_models import transform_dataset_sktime, create_model
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeClassifierCV

        transformer = create_model("minirocket", model_params={"num_kernels": 10000}, seed=args.seed)
        X_tr = X_train_scaled[:, 0, :]
        X_te = X_test_scaled[:, 0, :]
        ft_train = transform_dataset_sktime(X_tr, transformer, fit=True)
        ft_test = transform_dataset_sktime(X_te, transformer, fit=False)
        scaler = StandardScaler()
        ft_train = scaler.fit_transform(ft_train)
        ft_test = scaler.transform(ft_test)
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        clf.fit(ft_train, y_train)
        y_pred = clf.predict(ft_test)
        acc = float(np.mean(y_test == y_pred))
        results["minirocket_scaled_accuracy"] = acc
        if save_base:
            import pickle
            d = os.path.join(save_base, "cache_minirocket_scaled", "run")
            os.makedirs(d, exist_ok=True)
            for name, obj in [("transformer", transformer), ("scaler", scaler), ("model", clf)]:
                with open(os.path.join(d, f"{name}.pkl"), "wb") as f:
                    pickle.dump(obj, f)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"target_length": T, "representation": "scaled", "data_source": "synthetic"}, f, indent=2)
            latest_dir = os.path.join(save_base, "cache_minirocket_scaled")
            os.makedirs(latest_dir, exist_ok=True)
            with open(os.path.join(latest_dir, "LATEST"), "w") as f:
                f.write("run")
    except Exception as e:
        print(f"  MiniROCKET scaled failed: {e}", file=sys.stderr)
        results["minirocket_scaled_accuracy"] = None

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeClassifierCV

        X_tr = sanitize_feature_matrix(extract_feats4(X_train_scaled))
        X_te = sanitize_feature_matrix(extract_feats4(X_test_scaled))
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        acc = float(np.mean(y_test == y_pred))
        results["trivial_scaled_accuracy"] = acc
        if save_base:
            import pickle
            d = os.path.join(save_base, "cache_trivial_scaled")
            os.makedirs(d, exist_ok=True)
            for name, obj in [("scaler", scaler), ("model", clf)]:
                with open(os.path.join(d, f"{name}.pkl"), "wb") as f:
                    pickle.dump(obj, f)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"target_length": T, "representation": "trivial_4feat", "data_source": "synthetic"}, f, indent=2)
    except Exception as e:
        print(f"  trivial scaled failed: {e}", file=sys.stderr)
        results["trivial_scaled_accuracy"] = None

    split_length = args.split_eval_length if args.split_eval_length is not None else T
    if split_length < args.segment_length:
        print(f"split length {split_length} < segment_length {args.segment_length}, skipping split", file=sys.stderr)
        results["minirocket_split_by_split_accuracy"] = None
        results["minirocket_split_by_trace_accuracy"] = None
        results["trivial_split_by_split_accuracy"] = None
        results["trivial_split_by_trace_accuracy"] = None
    else:
        all_segments = []
        all_labels = []
        trace_ids = []
        for i in range(n_traces):
            trace_1d = traces_sktime[i, 0, :].astype(np.float64)
            trace_1d = np.nan_to_num(trace_1d, nan=np.nanmean(trace_1d), posinf=0.0, neginf=0.0)
            if args.split_eval_length is not None and len(trace_1d) != args.split_eval_length:
                trace_1d = resample_trace(trace_1d, args.split_eval_length)
            segs = split_trace_into_segments(trace_1d, args.segment_length, stride=args.stride)
            if len(segs) == 0:
                continue
            all_segments.append(segs)
            all_labels.extend([labels[i]] * len(segs))
            trace_ids.extend([i] * len(segs))
        segments = np.vstack(all_segments)
        seg_labels = np.array(all_labels, dtype=np.intp)
        trace_id_per_segment = np.array(trace_ids, dtype=np.intp)

        train_mask = np.isin(trace_id_per_segment, train_idx)
        test_mask = np.isin(trace_id_per_segment, test_idx)
        seg_train = segments[train_mask]
        seg_test = segments[test_mask]
        y_seg_train = seg_labels[train_mask]
        y_seg_test = seg_labels[test_mask]
        test_trace_ids = trace_id_per_segment[test_mask]
        y_test_trace = labels[test_idx]

        try:
            from train_models import transform_dataset_sktime, create_model
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import RidgeClassifierCV

            seg_train_3d = seg_train[:, np.newaxis, :]
            seg_test_3d = seg_test[:, np.newaxis, :]
            transformer = create_model("minirocket", model_params={"num_kernels": 10000}, seed=args.seed)
            ft_train = transform_dataset_sktime(seg_train, transformer, fit=True)
            ft_test = transform_dataset_sktime(seg_test, transformer, fit=False)
            scaler = StandardScaler()
            ft_train = scaler.fit_transform(ft_train)
            ft_test = scaler.transform(ft_test)
            clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            clf.fit(ft_train, y_seg_train)
            y_seg_pred = clf.predict(ft_test)
            acc_split = float(np.mean(y_seg_test == y_seg_pred))
            results["minirocket_split_by_split_accuracy"] = acc_split
            trace_pred = aggr_pred_trace(y_seg_pred, test_trace_ids, test_idx)
            acc_trace = float(np.mean(trace_pred == y_test_trace))
            results["minirocket_split_by_trace_accuracy"] = acc_trace
            if save_base:
                import pickle
                d = os.path.join(save_base, "cache_minirocket_split", "run")
                os.makedirs(d, exist_ok=True)
                for name, obj in [("transformer", transformer), ("scaler", scaler), ("model", clf)]:
                    with open(os.path.join(d, f"{name}.pkl"), "wb") as f:
                        pickle.dump(obj, f)
                with open(os.path.join(d, "config.json"), "w") as f:
                    json.dump({
                        "segment_length": args.segment_length, "stride": args.stride,
                        "representation": "split", "data_source": "synthetic"
                    }, f, indent=2)
                latest_dir = os.path.join(save_base, "cache_minirocket_split")
                os.makedirs(latest_dir, exist_ok=True)
                with open(os.path.join(latest_dir, "LATEST"), "w") as f:
                    f.write("run")
        except Exception as e:
            print(f"  MiniROCKET split failed: {e}", file=sys.stderr)
            results["minirocket_split_by_split_accuracy"] = None
            results["minirocket_split_by_trace_accuracy"] = None

        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import RidgeClassifierCV

            X_tr = sanitize_feature_matrix(extract_feats4(seg_train))
            X_te = sanitize_feature_matrix(extract_feats4(seg_test))
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            clf.fit(X_tr, y_seg_train)
            y_seg_pred = clf.predict(X_te)
            acc_split = float(np.mean(y_seg_test == y_seg_pred))
            results["trivial_split_by_split_accuracy"] = acc_split
            trace_pred = aggr_pred_trace(y_seg_pred, test_trace_ids, test_idx)
            acc_trace = float(np.mean(trace_pred == y_test_trace))
            results["trivial_split_by_trace_accuracy"] = acc_trace


            if save_base:
                import joblib
                d = os.path.join(save_base, "cache_trivial_split")
                os.makedirs(d, exist_ok=True)
                joblib.dump(scaler, os.path.join(d, "scaler.pkl"))
                joblib.dump(clf, os.path.join(d, "model.pkl"))
                with open(os.path.join(d, "config.json"), "w") as f:
                    json.dump({
                        "segment_length": args.segment_length, "stride": args.stride,
                        "representation": "split", "data_source": "synthetic"
                    }, f, indent=2)
                    
        except Exception as e:
            print(f"  trivial split failed: {e}", file=sys.stderr)
            results["trivial_split_by_split_accuracy"] = None
            results["trivial_split_by_trace_accuracy"] = None

    def pct(x):
        return f"{x:.2%}" if x is not None else "--"

    print("\n--- summary (synthetic, 80/20 train/test) ---")
    print(f"MiniROCKET:  scaled={pct(results.get('minirocket_scaled_accuracy'))}  split(by split)={pct(results.get('minirocket_split_by_split_accuracy'))}  split(by trace)={pct(results.get('minirocket_split_by_trace_accuracy'))}")
    print(f"simple 4-feat: scaled={pct(results.get('trivial_scaled_accuracy'))}  split(by split)={pct(results.get('trivial_split_by_split_accuracy'))}  split(by trace)={pct(results.get('trivial_split_by_trace_accuracy'))}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.output}")

    return 0

if __name__ == "__main__":
    sys.exit(main())

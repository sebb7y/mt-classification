import argparse
import json
import joblib
import os
import sys
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def train_bundle_synth(args):

    rng = np.random.default_rng(args.seed)
    n = args.n_samples
    X = np.column_stack([
        rng.uniform(0.1, 50, n),
        rng.uniform(0.01, 5, n),
        rng.uniform(0.01, 20, n),
        rng.uniform(0.01, 10, n),
    ])
    y = ((X[:, 0] > np.median(X[:, 0])) & (X[:, 1] > np.median(X[:, 1]))).astype(np.intp)
    flip = rng.random(n) < 0.1
    y[flip] = 1 - y[flip]
    feature_medians = np.nanmedian(X, axis=0)
    feature_medians = np.where(np.isfinite(feature_medians), feature_medians, 0.0)
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        if np.any(bad):
            X[bad, j] = feature_medians[j]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    training_stats = {
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
        "feature_medians": feature_medians.tolist()
    }
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
    n_classes = len(np.unique(y))
    joblib.dump(
        DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xs, y),
        os.path.join(args.output_dir, "clf_baseline_dt.joblib"),
    )
    joblib.dump(
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xs, y),
        os.path.join(args.output_dir, "clf_baseline_rf.joblib"),
    )
    if n_classes >= 2:
        joblib.dump(
            LogisticRegression(max_iter=500, random_state=42).fit(Xs, y),
            os.path.join(args.output_dir, "clf_baseline_lr.joblib"),
        )
    with open(os.path.join(args.output_dir, "training_stats.json"), "w") as f:
        json.dump(training_stats, f, indent=2)
    return 0


def train_bundle_real(args):
    from experimental_loader import prep_exp_traces_for_model
    from features import extract_feats4 as extract_feats
    from features import extract_feats_extra, extract_feats_full

    def load_exp(path, target_length, training_stats, return_traces=False):
        try:
            data = prep_exp_traces_for_model(
                path, good_only=False, target_length=target_length,
                use_resample=True, training_stats=training_stats,
            )
            traces = data["traces"]
            labels = np.asarray(data["labels"], dtype=np.intp)
        except Exception:
            return None, None
        if traces.size == 0:
            return None, None
        if return_traces:
            return traces, labels
        X = extract_feats(traces)
        return X, labels

    with open(args.split_file) as f:
        split_data = json.load(f)
    train_rows = split_data["train_rows"]
    if not train_rows:
        return 1
    cache_dir = os.path.abspath(args.output_dir)
    os.makedirs(cache_dir, exist_ok=True)
    target_length = args.target_length

    X_list, y_list = [], []
    all_traces = []
    for r in train_rows:
        path = r.get("path", "").strip()
        if not path or not os.path.isdir(path):
            continue
        X, y = load_exp(path, target_length, None)
        if X is not None:
            X_list.append(X)
            y_list.append(y)
        traces, _ = load_exp(path, target_length, None, return_traces=True)
        if traces is not None and traces.size > 0:
            all_traces.append(traces)
    if not X_list:
        return 1

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    feature_medians = np.nanmedian(X_train, axis=0)
    feature_medians = np.where(np.isfinite(feature_medians), feature_medians, 0.0)
    for j in range(X_train.shape[1]):
        bad = ~np.isfinite(X_train[:, j])
        if np.any(bad):
            X_train[bad, j] = feature_medians[j]
    scaler = StandardScaler().fit(X_train)

    X_train_full_list = []
    for r in train_rows:
        path = r.get("path", "").strip()
        if not path or not os.path.isdir(path):
            continue
        traces, _ = load_exp(path, target_length, None, return_traces=True)
        if traces is not None and traces.size > 0:
            X_train_full_list.append(extract_feats_full(traces))
    X_train_full = np.vstack(X_train_full_list) if X_train_full_list else None
    feature_medians_extra = None
    if X_train_full is not None and X_train_full.size > 0:
        feature_medians_extra = np.nanmedian(X_train_full, axis=0)
        feature_medians_extra = np.where(np.isfinite(feature_medians_extra), feature_medians_extra, 0.0)
        for j in range(X_train_full.shape[1]):
            bad = ~np.isfinite(X_train_full[:, j])
            if np.any(bad):
                X_train_full[bad, j] = feature_medians_extra[j]
        scaler_extra = StandardScaler().fit(X_train_full)
        if len(X_train_full) == len(y_train) and len(np.unique(y_train)) >= 2:
            joblib.dump(scaler_extra, os.path.join(cache_dir, "scaler_extra.joblib"))
            joblib.dump(
                DecisionTreeClassifier(max_depth=5, random_state=42).fit(scaler_extra.transform(X_train_full), y_train),
                os.path.join(cache_dir, "clf_baseline_dt_extra.joblib"),
            )
            joblib.dump(
                RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42).fit(scaler_extra.transform(X_train_full), y_train),
                os.path.join(cache_dir, "clf_baseline_rf_extra.joblib"),
            )

    if all_traces:
        concat = np.concatenate(all_traces, axis=0)
        training_stats = {"mean": float(np.nanmean(concat)), "std": float(np.nanstd(concat))}
    else:
        training_stats = {"mean": 0.0, "std": 1.0}
    training_stats["feature_medians"] = feature_medians.tolist()
    training_stats["feature_medians_extra"] = feature_medians_extra.tolist() if feature_medians_extra is not None else []
    with open(os.path.join(cache_dir, "training_stats.json"), "w") as f:
        json.dump(training_stats, f, indent=2)
    n_classes = len(np.unique(y_train))
    joblib.dump(scaler, os.path.join(cache_dir, "scaler.joblib"))
    Xs = scaler.transform(X_train)
    joblib.dump(DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xs, y_train), os.path.join(cache_dir, "clf_baseline_dt.joblib"))
    joblib.dump(RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xs, y_train), os.path.join(cache_dir, "clf_baseline_rf.joblib"))
    if n_classes >= 2:
        joblib.dump(LogisticRegression(max_iter=500, random_state=42).fit(Xs, y_train), os.path.join(cache_dir, "clf_baseline_lr.joblib"))

    X_list_a1, y_list_a1 = [], []
    for r in train_rows:
        path = r.get("path", "").strip()
        if not path or not os.path.isdir(path):
            continue
        X, y = load_exp(path, target_length, training_stats)
        if X is not None:
            X_list_a1.append(X)
            y_list_a1.append(y)
    if X_list_a1:
        X_train_a1 = np.vstack(X_list_a1)
        y_train_a1 = np.concatenate(y_list_a1)
        for j in range(X_train_a1.shape[1]):
            bad = ~np.isfinite(X_train_a1[:, j])
            if np.any(bad):
                X_train_a1[bad, j] = feature_medians[j]
        if len(np.unique(y_train_a1)) >= 2:
            joblib.dump(
                DecisionTreeClassifier(max_depth=3, random_state=42).fit(scaler.transform(X_train_a1), y_train_a1),
                os.path.join(cache_dir, "clf_approach1_scaled_dt.joblib"),
            )
    return 0


def train_single_real(args):
    from real_data_jobs import train_real_model
    try:
        train_real_model(
            split_file=args.split_file,
            output_dir=args.output_dir,
            model_type=args.model_type,
            representation=args.representation,
            target_length=args.target_length,
            segment_length=args.segment_length,
            stride=args.stride,
            seed=args.seed,
            max_segments=args.max_segments,
            command_line=" ".join(sys.argv),
        )
    except ValueError:
        return 1
    return 0


def train_trivial_real_scaled(args):
    from features import extract_feats4, sanitize_feature_matrix
    from real_data_jobs import read_real_traces_resampled, read_split_rows
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    split_data, train_rows = read_split_rows(args.split_file, key="train_rows")
    test_rows = split_data.get("test_rows", []) if args.internal_test_ratio is None else []
    if not train_rows or (args.internal_test_ratio is None and not test_rows):
        return 1

    t0 = time.time()
    traces_train, labels_train = read_real_traces_resampled(train_rows, args.target_length)
    if args.internal_test_ratio is not None:
        traces_test = labels_test = None
    else:
        traces_test, labels_test = read_real_traces_resampled(test_rows, args.target_length)
    load_time = time.time() - t0
    if len(traces_train) == 0:
        return 1

    if args.internal_test_ratio is not None:
        X_all = sanitize_feature_matrix(extract_feats4(traces_train))
        X_train, X_test, labels_train, labels_test = train_test_split(
            X_all, labels_train, test_size=args.internal_test_ratio, random_state=args.seed, stratify=labels_train
        )
        n_train, n_test = len(X_train), len(X_test)
    else:
        if len(traces_test) == 0:
            return 1
        X_train = sanitize_feature_matrix(extract_feats4(traces_train))
        X_test = sanitize_feature_matrix(extract_feats4(traces_test))
        n_train, n_test = len(traces_train), len(traces_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    t0 = time.time()
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=3)
    clf.fit(X_train, labels_train)
    train_time = time.time() - t0
    y_pred = clf.predict(X_test)
    acc = accuracy_score(labels_test, y_pred)
    results = {
        "model": "RidgeClassifierCV_4feat_scaled",
        "representation": "scaled",
        "split_file": args.split_file,
        "target_length": args.target_length,
        "n_train": n_train,
        "n_test": n_test,
        "test_accuracy": float(acc),
        "train_time_seconds": train_time,
        "load_time_seconds": load_time,
        "seed": args.seed,
        "internal_test_ratio": args.internal_test_ratio
    }
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
        joblib.dump(clf, os.path.join(args.output_dir, "model.pkl"))
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump({"target_length": args.target_length, "representation": "scaled"}, f, indent=2)
    return 0


def train_trivial_dataset_split(args):
    from features import extract_feats4, sanitize_feature_matrix
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    if not os.path.isfile(args.dataset):
        return 1
    ds = np.load(args.dataset, allow_pickle=True)
    segments = np.asarray(ds["traces"])
    labels = np.asarray(ds["labels"], dtype=np.intp)
    if segments.ndim != 2 or len(labels) != len(segments):
        return 1
    X = sanitize_feature_matrix(extract_feats4(segments))
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=args.test_ratio, random_state=args.seed, stratify=labels
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump({
                "model": "RidgeClassifierCV_4feat",
                "dataset": args.dataset,
                "n_segments": len(segments),
                "test_accuracy": float(acc),
                "seed": args.seed,
                "segment_length": args.segment_length,
                "stride": args.stride
            }, f, indent=2)
        joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
        joblib.dump(clf, os.path.join(args.output_dir, "model.pkl"))
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump({"segment_length": args.segment_length, "stride": args.stride, "representation": "split"}, f, indent=2)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, choices=("bundle", "single-model", "trivial"), help="training mode")
    parser.add_argument("--source", required=True, choices=("synthetic", "real", "dataset"), help="data source")
    parser.add_argument("--representation", default="scaled", choices=("scaled", "split"), help="representation")
    parser.add_argument("--output-dir", "-o", required=True, help="output/cache directory")
    parser.add_argument("--split-file", help="path to split.json (for real data)")
    parser.add_argument("--dataset", help="path to .npz dataset (for source=dataset)")
    parser.add_argument("--model-type", default="minirocket", choices=("minirocket", "tsf"), help="for single-model")
    parser.add_argument("--target-length", type=int, default=5000)
    parser.add_argument("--segment-length", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=20_000, help="for bundle+synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2, help="for trivial+dataset")
    parser.add_argument("--internal-test-ratio", type=float, default=None, help="for trivial+real, internal split")
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    if args.stride is None:
        args.stride = args.segment_length

    if args.trainer == "bundle" and args.source == "synthetic":
        return train_bundle_synth(args)
    if args.trainer == "bundle" and args.source == "real":
        if not args.split_file:
            return 1
        args.split_file = os.path.expanduser(args.split_file)
        return train_bundle_real(args)
    if args.trainer == "single-model" and args.source == "real":
        if not args.split_file:
            return 1
        args.split_file = os.path.expanduser(args.split_file)
        return train_single_real(args)
    if args.trainer == "trivial" and args.source == "real" and args.representation == "scaled":
        if not args.split_file:
            return 1
        args.split_file = os.path.expanduser(args.split_file)
        return train_trivial_real_scaled(args)
    if args.trainer == "trivial" and args.source == "dataset" and args.representation == "split":
        if not args.dataset:
            return 1
        args.dataset = os.path.expanduser(args.dataset)
        return train_trivial_dataset_split(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())

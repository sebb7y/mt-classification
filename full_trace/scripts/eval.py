import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

RESULT_FIELDNAMES = [
    "path", "experiment_id", "source", "n_beads", "correct", "accuracy", "method",
    "n_predicted_good", "n_predicted_bad", "tp", "tn", "fp", "fn",
]


def confusion(lab, pred):
    lab = np.asarray(lab).ravel()
    pred = np.asarray(pred).ravel()
    good, bad = 1, 0
    tp = int(((pred == good) & (lab == good)).sum())
    tn = int(((pred == bad) & (lab == bad)).sum())
    fp = int(((pred == good) & (lab == bad)).sum())
    fn = int(((pred == bad) & (lab == good)).sum())
    n_pred_good = int((pred == good).sum())
    n_pred_bad = int((pred == bad).sum())
    return tp, tn, fp, fn, n_pred_good, n_pred_bad


def row_result(path, experiment_id, source, n, correct, method, pred, lab):
    tp, tn, fp, fn, n_pred_good, n_pred_bad = confusion(lab, pred)
    return {
        "path": path,
        "experiment_id": experiment_id,
        "source": source,
        "n_beads": n,
        "correct": correct,
        "accuracy": correct / n if n else 0,
        "method": method,
        "n_predicted_good": n_pred_good,
        "n_predicted_bad": n_pred_bad,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def score_array_bundle(args):
    import joblib
    from experimental_loader import prep_exp_traces_for_model
    from features import extract_feats4 as extract_feats
    from features import extract_feats_extra, extract_feats_full
    from rules_baseline import classify_traces as rules_classify_traces

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
    test_rows = split_data["test_rows"]
    if args.task_id < 0 or args.task_id >= len(test_rows):
        return 1
    row = test_rows[args.task_id]
    path = row.get("path", "").strip()
    if not path or not os.path.isdir(path):
        return 1
    cache_dir = os.path.abspath(args.cache_dir)
    with open(os.path.join(cache_dir, "training_stats.json")) as f:
        training_stats = json.load(f)
    feature_medians = np.array(training_stats.get("feature_medians", [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    feature_medians_extra = np.array(training_stats.get("feature_medians_extra", []), dtype=np.float64)
    scaler = joblib.load(os.path.join(cache_dir, "scaler.joblib"))
    scaler_extra = None
    if os.path.isfile(os.path.join(cache_dir, "scaler_extra.joblib")):
        scaler_extra = joblib.load(os.path.join(cache_dir, "scaler_extra.joblib"))
    cache_methods = []
    for f in os.listdir(cache_dir):
        if f.startswith("clf_") and f.endswith(".joblib"):
            cache_methods.append(f[len("clf_"): -len(".joblib")])
    if not cache_methods:
        return 1
    classifiers = {m: joblib.load(os.path.join(cache_dir, f"clf_{m}.joblib")) for m in cache_methods}
    extra_methods = [m for m in cache_methods if m in ("baseline_rf_extra", "baseline_dt_extra")]
    target_length = args.target_length

    traces, labels = load_exp(path, target_length, None, return_traces=True)
    need_scaled = "approach1_scaled_dt" in classifiers
    X_scaled = None
    if need_scaled:
        X_scaled, _ = load_exp(path, target_length, training_stats=training_stats)
    if traces is None or labels is None or len(labels) == 0:
        return 1
    n = len(labels)
    lab = np.asarray(labels, dtype=np.intp)
    X = extract_feats(traces)
    for j in range(min(X.shape[1], len(feature_medians))):
        bad = ~np.isfinite(X[:, j])
        if np.any(bad):
            X[bad, j] = feature_medians[j]
    if X_scaled is not None and len(X_scaled) == n:
        for j in range(min(X_scaled.shape[1], len(feature_medians))):
            bad = ~np.isfinite(X_scaled[:, j])
            if np.any(bad):
                X_scaled[bad, j] = feature_medians[j]
    X_full = None
    if extra_methods and scaler_extra is not None and len(feature_medians_extra) >= 8:
        X_full = extract_feats_full(traces)
        for j in range(min(X_full.shape[1], len(feature_medians_extra))):
            bad = ~np.isfinite(X_full[:, j])
            if np.any(bad):
                X_full[bad, j] = feature_medians_extra[j]

    results = []
    for method in cache_methods:
        clf = classifiers[method]
        if method == "approach1_scaled_dt" and X_scaled is not None and len(X_scaled) == n:
            pred = clf.predict(scaler.transform(X_scaled))
        elif method in extra_methods and X_full is not None and len(X_full) == n and scaler_extra is not None:
            pred = clf.predict(scaler_extra.transform(X_full))
        else:
            pred = clf.predict(scaler.transform(X))
        correct = int((pred == lab).sum())
        results.append(row_result(path, row.get("experiment_id", os.path.basename(path.rstrip("/"))), row.get("source", ""), n, correct, method, pred, lab))
    pred = rules_classify_traces(traces)
    results.append(row_result(path, row.get("experiment_id", os.path.basename(path.rstrip("/"))), row.get("source", ""), n, int((pred == lab).sum()), "approach4_rules", pred, lab))
    if not results:
        return 1
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"result_{args.task_id}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        w.writeheader()
        w.writerows(results)
    return 0


def score_array_model(args):
    from experimental_loader import prep_exp_traces_for_model
    from eval_stuff import read_model_run, predict_batch, build_traces_labels_split, aggr_pred_bead

    with open(args.split_file) as f:
        split_data = json.load(f)
    test_rows = split_data["test_rows"]
    if args.task_id < 0 or args.task_id >= len(test_rows):
        return 1
    row = test_rows[args.task_id]
    path = row.get("path", "").strip()
    if not path or not os.path.isdir(path):
        return 1
    run = read_model_run(args.model_dir)
    config = run.get("config") or {}
    use_split = config.get("representation") == "split" or config.get("segment_length") is not None
    segment_length = config.get("segment_length") or run.get("target_length") or 5000
    stride = config.get("stride", segment_length)
    target_length = run.get("target_length") or segment_length
    training_stats = run.get("training_stats")

    if use_split:
        traces, _sl, bead_id_per_segment, bead_labels, data = build_traces_labels_split(
            path, traces_path=None, good_only=False, target_length=segment_length, stride=stride, training_stats=training_stats
        )
        if traces.size == 0:
            return 1
        n = data["n_traces"]
        y_pred_seg = predict_batch(run, traces)
        y_pred_seg = np.asarray(y_pred_seg).ravel()
        y_pred = aggr_pred_bead(y_pred_seg, bead_id_per_segment, n)
        labels = np.asarray(bead_labels, dtype=np.intp).ravel()
    else:
        data = prep_exp_traces_for_model(
            path, good_only=False, target_length=target_length, use_resample=True, training_stats=training_stats
        )
        traces = data["traces"]
        labels = np.asarray(data["labels"], dtype=np.intp)
        if traces.size == 0 or len(labels) == 0:
            return 1
        y_pred = predict_batch(run, traces)
        y_pred = np.asarray(y_pred).ravel()
        n = len(labels)
        if len(y_pred) != n:
            return 1
        labels = np.asarray(labels).ravel()

    correct = int((y_pred == labels).sum())
    method = args.method_name or "minirocket"
    res = row_result(path, row.get("experiment_id", os.path.basename(path.rstrip("/"))), row.get("source", ""), n, correct, method, y_pred, labels)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"result_{args.task_id}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        w.writeheader()
        w.writerow(res)
    return 0


def score_array_rules(args):
    from experimental_loader import prep_exp_traces_for_model
    from rules_baseline import classify_traces as rules_classify_traces

    with open(args.split_file) as f:
        split_data = json.load(f)
    test_rows = split_data["test_rows"]
    if args.task_id < 0 or args.task_id >= len(test_rows):
        return 1
    row = test_rows[args.task_id]
    path = row.get("path", "").strip()
    if not path or not os.path.isdir(path):
        return 1
    data = prep_exp_traces_for_model(
        path, good_only=False, target_length=args.target_length, use_resample=True, training_stats=None
    )
    traces = data["traces"]
    labels = data["labels"]
    if traces.size == 0:
        return 1
    lab = np.asarray(labels, dtype=np.intp)
    n = len(lab)
    pred = rules_classify_traces(traces)
    correct = int((pred == lab).sum())
    method = args.method_name or "approach4_rules"
    res = row_result(path, row.get("experiment_id", os.path.basename(path.rstrip("/"))), row.get("source", ""), n, correct, method, pred, lab)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"result_{args.task_id}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        w.writeheader()
        w.writerow(res)
    return 0


def score_bead_acc_model(args):
    from eval_stuff import read_model_run, predict_batch, build_traces_labels_split, aggr_pred_bead, aggr_pred_bead_sum

    model_dir = args.model_dir
    if os.path.isfile(os.path.join(model_dir, "LATEST")):
        with open(os.path.join(model_dir, "LATEST")) as f:
            sub = f.read().strip()
        model_dir = os.path.join(args.model_dir, sub) if not os.path.isabs(sub) else sub
    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        misplaced = os.path.join(repo_root, "~", "tweezcat", "eval_minirocket_real", "cache_minirocket_split")
        if os.path.isdir(misplaced):
            runs = sorted(glob.glob(os.path.join(misplaced, "minirocket_*")))
            if runs:
                model_dir = runs[-1]
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        return 1
    with open(args.split_file) as f:
        split_data = json.load(f)
    test_rows = split_data.get("test_rows", [])
    if not test_rows:
        return 1
    run = read_model_run(model_dir)
    config = run.get("config") or {}
    segment_length = config.get("segment_length") or run.get("target_length") or 5000
    stride = config.get("stride", segment_length)
    training_stats = run.get("training_stats")

    all_labels = []
    all_pred_maj = []
    all_pred_sum = []
    for row in test_rows:
        path = row.get("path", "").strip()
        if not path or not os.path.isdir(path):
            continue
        try:
            traces, _sl, bead_id_per_segment, bead_labels, data = build_traces_labels_split(
                path, traces_path=None, good_only=False, target_length=segment_length, stride=stride, training_stats=training_stats
            )
        except Exception:
            continue
        if traces.size == 0:
            continue
        n = data["n_traces"]
        y_pred_seg = predict_batch(run, traces)
        y_pred_seg = np.asarray(y_pred_seg).ravel()
        pred_maj = aggr_pred_bead(y_pred_seg, bead_id_per_segment, n)
        pred_sum = aggr_pred_bead_sum(y_pred_seg, bead_id_per_segment, n, threshold_frac=args.sum_threshold_frac)
        bead_labels = np.asarray(bead_labels, dtype=np.intp).ravel()
        all_labels.append(bead_labels)
        all_pred_maj.append(pred_maj)
        all_pred_sum.append(pred_sum)
    if not all_labels:
        return 1
    y_true = np.concatenate(all_labels)
    y_maj = np.concatenate(all_pred_maj)
    y_sum = np.concatenate(all_pred_sum)
    acc_maj = float(np.mean(y_true == y_maj))
    acc_sum = float(np.mean(y_true == y_sum))
    results = {
        "n_test_beads": int(len(y_true)),
        "n_test_experiments": len(all_labels),
        "accuracy_majority_vote": acc_maj,
        "accuracy_sum_threshold": acc_sum,
        "sum_threshold_frac": args.sum_threshold_frac,
        "model_dir": model_dir,
        "split_file": args.split_file
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    return 0


def score_bead_acc_trivial(args):
    from features import extract_feats4, sanitize_feature_matrix
    from eval_stuff import build_traces_labels_split, aggr_pred_bead, aggr_pred_bead_sum
    import joblib

    if not os.path.isfile(os.path.join(args.model_dir, "config.json")) or not os.path.isfile(os.path.join(args.model_dir, "model.pkl")):
        return 1
    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)
    segment_length = config.get("segment_length", 5000)
    stride = config.get("stride", segment_length)
    scaler = joblib.load(os.path.join(args.model_dir, "scaler.pkl"))
    clf = joblib.load(os.path.join(args.model_dir, "model.pkl"))

    with open(args.split_file) as f:
        split_data = json.load(f)
    test_rows = split_data.get("test_rows", [])
    if not test_rows:
        return 1
    all_labels = []
    all_pred_maj = []
    all_pred_sum = []
    for row in test_rows:
        path = row.get("path", "").strip()
        if not path or not os.path.isdir(path):
            continue
        try:
            traces, _sl, bead_id_per_segment, bead_labels, data = build_traces_labels_split(
                path, traces_path=None, good_only=False, target_length=segment_length, stride=stride, training_stats=None
            )
        except Exception:
            continue
        if traces.size == 0:
            continue
        n = data["n_traces"]
        X = sanitize_feature_matrix(extract_feats4(traces))
        X = scaler.transform(X)
        y_pred_seg = clf.predict(X).ravel()
        pred_maj = aggr_pred_bead(y_pred_seg, bead_id_per_segment, n)
        pred_sum = aggr_pred_bead_sum(y_pred_seg, bead_id_per_segment, n, threshold_frac=args.sum_threshold_frac)
        bead_labels = np.asarray(bead_labels, dtype=np.intp).ravel()
        all_labels.append(bead_labels)
        all_pred_maj.append(pred_maj)
        all_pred_sum.append(pred_sum)
    if not all_labels:
        return 1
    y_true = np.concatenate(all_labels)
    y_maj = np.concatenate(all_pred_maj)
    y_sum = np.concatenate(all_pred_sum)
    acc_maj = float(np.mean(y_true == y_maj))
    acc_sum = float(np.mean(y_true == y_sum))
    results = {
        "n_test_beads": int(len(y_true)),
        "n_test_experiments": len(all_labels),
        "accuracy_majority_vote": acc_maj,
        "accuracy_sum_threshold": acc_sum,
        "sum_threshold_frac": args.sum_threshold_frac,
        "model_dir": args.model_dir,
        "split_file": args.split_file
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("array-task", "bead-accuracy"))
    parser.add_argument("--evaluator", required=True, choices=("bundle", "saved-model", "rules", "trivial"))
    parser.add_argument("--split-file", required=True, help="path to split.json")
    parser.add_argument("--cache-dir", help="cache dir (for evaluator=bundle)")
    parser.add_argument("--model-dir", help="model run dir (for saved-model / trivial)")
    parser.add_argument("--output-dir", help="directory for result_<id>.csv (array-task)")
    parser.add_argument("--output", "-o", help="output json path (bead-accuracy)")
    parser.add_argument("--method-name", default=None, help="method name in CSV (e.g. minirocket, approach4_rules)")
    parser.add_argument("--target-length", type=int, default=5000)
    parser.add_argument("--sum-threshold-frac", type=float, default=0.5, help="for bead-accuracy")
    parser.add_argument("task_id", type=int, nargs="?", default=None, help="SLURM_ARRAY_TASK_ID (array-task)")
    args = parser.parse_args()
    args.split_file = os.path.expanduser(args.split_file)

    if args.mode == "array-task":
        if args.task_id is None:
            return 1
        if args.evaluator == "bundle":
            if not args.cache_dir:
                return 1
            args.cache_dir = os.path.expanduser(args.cache_dir)
            args.output_dir = args.output_dir or "."
            return score_array_bundle(args)
        if args.evaluator == "saved-model":
            if not args.model_dir:
                return 1
            args.model_dir = os.path.expanduser(args.model_dir)
            args.output_dir = args.output_dir or "."
            return score_array_model(args)
        if args.evaluator == "rules":
            args.output_dir = args.output_dir or "."
            return score_array_rules(args)
        return 1
    if args.mode == "bead-accuracy":
        if not args.model_dir:
            return 1
        args.model_dir = os.path.expanduser(args.model_dir)
        if args.evaluator == "saved-model":
            return score_bead_acc_model(args)
        if args.evaluator == "trivial":
            return score_bead_acc_trivial(args)
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())

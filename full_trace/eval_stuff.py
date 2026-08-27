import json
import os
import pickle

import numpy as np

from experimental_loader import load_exp_traces, split_trace_into_segments


def resolve_model_dir(model_dir):
    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    if os.path.isfile(os.path.join(model_dir, "LATEST")):
        with open(os.path.join(model_dir, "LATEST")) as f:
            sub = f.read().strip()
        if not os.path.isabs(sub):
            sub = os.path.join(model_dir, sub)
        model_dir = os.path.abspath(sub)
    return model_dir


def read_model_run(model_dir):
    model_dir = resolve_model_dir(model_dir)
    run = {"model_dir": model_dir}
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            run["config"] = json.load(f)
    else:
        run["config"] = {}
    run["target_length"] = run["config"].get("target_length")
    run["segment_length"] = run["config"].get("segment_length")
    run["stride"] = run["config"].get("stride", run["segment_length"])
    stats_path = os.path.join(model_dir, "training_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            run["training_stats"] = json.load(f)
    else:
        run["training_stats"] = None

    def load_pkl(path):
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            import joblib
            return joblib.load(path)
    run["model"] = load_pkl(os.path.join(model_dir, "model.pkl"))
    run["scaler"] = load_pkl(os.path.join(model_dir, "scaler.pkl"))
    run["transformer"] = load_pkl(os.path.join(model_dir, "transformer.pkl"))
    return run


def predict_batch(run, traces):
    model = run.get("model")
    scaler = run.get("scaler")
    transformer = run.get("transformer")
    if model is None:
        raise ValueError("no model in run")
    if traces.ndim == 3 and traces.shape[1] == 1:
        traces_2d = traces[:, 0, :]
    else:
        traces_2d = np.asarray(traces, dtype=np.float64)
    if transformer is not None and scaler is not None:
        from train_models import transform_dataset_sktime
        features = transform_dataset_sktime(traces_2d, transformer, fit=False)
        if hasattr(features, "values"):
            features = features.values
        features = np.asarray(features, dtype=np.float64)
        X = scaler.transform(features)
        pred = model.predict(X)
        return np.asarray(pred).ravel()
    if transformer is None and scaler is not None and model is not None:
        from features import extract_feats4, sanitize_feature_matrix
        X = sanitize_feature_matrix(extract_feats4(traces_2d))
        X = scaler.transform(X)
        pred = model.predict(X)
        return np.asarray(pred).ravel()
    if traces_2d.ndim == 2:
        traces_sktime = traces_2d[:, np.newaxis, :]
    else:
        traces_sktime = traces_2d
    pred = model.predict(traces_sktime)
    return np.asarray(pred).ravel()


def build_traces_labels_split(path, traces_path=None, good_only=False, target_length=5000, stride=None, training_stats=None):
    data = load_exp_traces(path, traces_path=traces_path)
    z = data.z
    bead_labels = data.good_mask.astype(np.int64)
    n_frames, n_beads = z.shape
    segment_length = target_length
    stride = stride if stride is not None else segment_length
    all_segments = []
    all_segment_labels = []
    bead_id_per_segment = []
    for bead_idx in range(n_beads):
        trace = z[:, bead_idx].astype(np.float64)
        if np.any(np.isnan(trace)):
            trace = np.nan_to_num(trace, nan=np.nanmean(trace), posinf=0.0, neginf=0.0)
        segments = split_trace_into_segments(trace, segment_length, stride=stride)
        if len(segments) == 0:
            continue
        label = int(bead_labels[bead_idx])
        all_segments.append(segments)
        all_segment_labels.extend([label] * len(segments))
        bead_id_per_segment.extend([bead_idx] * len(segments))
    if not all_segments:
        traces = np.zeros((0, segment_length), dtype=np.float64)
        segment_labels = np.array([], dtype=np.intp)
        bead_id_per_segment = np.array([], dtype=np.intp)
    else:
        traces = np.vstack(all_segments).astype(np.float64, copy=False)
        segment_labels = np.array(all_segment_labels, dtype=np.intp)
        bead_id_per_segment = np.array(bead_id_per_segment, dtype=np.intp)
    bead_labels_arr = np.asarray(bead_labels, dtype=np.intp)
    data_dict = {"n_traces": n_beads, "segment_length": segment_length, "stride": stride}
    return traces, segment_labels, bead_id_per_segment, bead_labels_arr, data_dict


def aggr_pred_bead(y_pred, bead_id_per_segment, n_beads):
    bead_pred = np.zeros(n_beads, dtype=np.int64)
    for b in range(n_beads):
        mask = bead_id_per_segment == b
        if not np.any(mask):
            continue
        votes = y_pred[mask]
        bead_pred[b] = int(np.bincount(votes).argmax())
    return bead_pred


def aggr_pred_bead_sum(y_pred, bead_id_per_segment, n_beads, threshold_frac=0.5):
    bead_pred = np.zeros(n_beads, dtype=np.int64)
    for b in range(n_beads):
        mask = bead_id_per_segment == b
        if not np.any(mask):
            continue
        votes = y_pred[mask]
        n_seg = len(votes)
        s = int(np.sum(votes))
        bead_pred[b] = 1 if s >= threshold_frac * n_seg else 0
    return bead_pred

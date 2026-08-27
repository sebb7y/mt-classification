import json
import os
import time

import numpy as np

from experimental_loader import load_exp_traces, prep_exp_traces_for_model, split_trace_into_segments
from train_models import save_training_outputs, train_model


def read_split_rows(split_file, key="train_rows"):
    with open(split_file) as f:
        split_data = json.load(f)
    return split_data, split_data.get(key, [])


def read_real_traces_resampled(rows, target_length, training_stats=None):
    traces_list = []
    labels_list = []
    for row in rows:
        path = row.get("path", "").strip()
        if not path or not os.path.isdir(path):
            continue
        try:
            data = prep_exp_traces_for_model(
                path, good_only=False, target_length=target_length, use_resample=True, training_stats=training_stats
            )
        except Exception:
            continue
        traces = data["traces"]
        labels = data["labels"]
        if traces.size == 0:
            continue
        traces_list.extend(traces)
        labels_list.extend(labels)
    if not traces_list:
        return np.zeros((0, target_length), dtype=np.float64), np.array([], dtype=np.intp)
    return np.asarray(traces_list, dtype=np.float64), np.asarray(labels_list, dtype=np.intp)


def read_real_segments(rows, segment_length, stride=None, max_segments=None, seed=42):
    stride = stride if stride is not None else segment_length
    all_segments = []
    all_labels = []
    for row in rows:
        path = row.get("path", "").strip()
        if not path or not os.path.isdir(path):
            continue
        try:
            data = load_exp_traces(path)
        except Exception:
            continue
        z = data.z
        bead_labels = data.good_mask.astype(np.int64)
        for bead_idx in range(z.shape[1]):
            trace = z[:, bead_idx].astype(np.float64)
            if np.any(np.isnan(trace)):
                finite = np.isfinite(trace)
                fill_value = float(np.mean(trace[finite])) if np.any(finite) else 0.0
                trace = np.nan_to_num(trace, nan=fill_value, posinf=0.0, neginf=0.0)
            segments = split_trace_into_segments(trace, segment_length, stride=stride)
            if len(segments) == 0:
                continue
            all_segments.append(segments)
            all_labels.extend([int(bead_labels[bead_idx])] * len(segments))
    if not all_segments:
        return np.zeros((0, segment_length), dtype=np.float64), np.array([], dtype=np.intp)
    segments = np.vstack(all_segments).astype(np.float64, copy=False)
    labels = np.asarray(all_labels, dtype=np.intp)
    if max_segments is not None and len(segments) > max_segments:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(segments), size=max_segments, replace=False)
        segments = segments[idx]
        labels = labels[idx]
    return segments, labels


def save_fixed_length_dataset(dataset_path, traces, labels):
    if traces.ndim != 2:
        raise ValueError(f"expected traces shape (n, T), got {traces.shape}")
    n_traces, target_length = traces.shape
    times = np.broadcast_to(np.arange(target_length, dtype=np.float64), (n_traces, target_length)).copy()
    traces_sktime = traces[:, np.newaxis, :]
    os.makedirs(os.path.dirname(dataset_path) or ".", exist_ok=True)
    np.savez_compressed(
        dataset_path,
        traces=traces,
        times=times,
        labels=labels,
        n_traces=n_traces,
        traces_sktime=traces_sktime,
        traces_sktime_2d=traces,
        median_length=target_length,
        max_length=target_length,
        format_version="2.0",
    )
    return {"dataset_path": dataset_path, "n_traces": n_traces, "target_length": target_length}


def train_real_model(*, split_file, output_dir, model_type, representation, target_length=None, segment_length=None, stride=None, seed=42, max_segments=None, command_line=None):
    _, train_rows = read_split_rows(split_file, key="train_rows")
    if not train_rows:
        raise ValueError("no train rows in split")
    os.makedirs(output_dir, exist_ok=True)
    if representation == "scaled":
        if target_length is None:
            raise ValueError("target_length required for scaled")
        traces, labels = read_real_traces_resampled(train_rows, target_length=target_length)
        dataset_name = "real_dataset.npz"
    elif representation == "split":
        if segment_length is None:
            raise ValueError("segment_length required for split")
        stride = stride if stride is not None else segment_length
        traces, labels = read_real_segments(train_rows, segment_length=segment_length, stride=stride, max_segments=max_segments, seed=seed)
        target_length = segment_length
        dataset_name = "real_dataset_split.npz"
    else:
        raise ValueError(f"unknown representation: {representation}")
    if len(traces) == 0:
        raise ValueError("no traces loaded")
    dataset_path = os.path.join(output_dir, dataset_name)
    dataset_info = save_fixed_length_dataset(dataset_path, traces, labels)
    start_time = time.time()
    results = train_model(
        model_type=model_type,
        use_existing_dataset=dataset_path,
        test_ratio=0.2,
        seed=seed,
        save_dataset_path=None,
        auto_cache=False,
        force_regenerate=False,
    )
    train_time = time.time() - start_time
    config = {
        "model_type": model_type,
        "use_existing_dataset": dataset_path,
        "test_ratio": 0.2,
        "seed": seed,
        "target_length": target_length
    }
    if representation == "split":
        config["representation"] = "split"
        config["segment_length"] = segment_length
        config["stride"] = stride
        config["n_segments"] = int(dataset_info["n_traces"])
    else:
        config["n_traces"] = int(dataset_info["n_traces"])
    runtime_info = {"total_runtime_seconds": train_time}
    run_dir = save_training_outputs(train_result=results, config=config, runtime_info=runtime_info, output_dir=output_dir, save_model=True, command_line=command_line)
    with open(os.path.join(output_dir, "LATEST"), "w") as f:
        f.write(run_dir)
    return run_dir

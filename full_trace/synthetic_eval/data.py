import csv
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from experimental_loader import prep_exp_traces_for_model, read_good_bead_mask
from scripts.eval_manifest import find_exp_folders, top_level_source, traces_shape


def discover_labelled_experiments(root, max_depth=8, sources=None, require_both_classes=True):
    root = os.path.abspath(root)
    allowed = {s.strip() for s in sources.split(",") if s.strip()} if isinstance(sources, str) and sources else None
    rows = []
    for folder in find_exp_folders(root, max_depth=max_depth):
        source = top_level_source(folder, root)
        if allowed is not None and source not in allowed:
            continue
        shape = traces_shape(os.path.join(folder, "traces.npy"))
        if shape is None:
            continue
        _n_frames, n_beads = int(shape[0]), int(shape[1])
        try:
            good_mask = read_good_bead_mask(folder, n_beads)
        except Exception:
            continue
        has_labels = good_mask is not None and len(good_mask) == n_beads
        has_both = bool(np.any(good_mask) and np.any(~good_mask)) if has_labels else False
        if require_both_classes and not has_both:
            continue
        rows.append({
            "path": folder,
            "rel_path": os.path.relpath(folder, root),
            "experiment_id": os.path.basename(folder),
            "source": source,
            "n_frames": _n_frames,
            "n_beads": n_beads,
            "has_labels": bool(has_labels),
            "has_both_classes": bool(has_both),
        })
    return rows


def split_experiments(rows, train_frac=0.8, seed=42):
    if len(rows) < 2:
        raise ValueError("need at least 2 labelled experiments for an experiment-level split")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n_train = max(1, int(len(rows) * train_frac))
    if n_train >= len(rows):
        n_train = len(rows) - 1
    train_idx = set(idx[:n_train])
    train_rows = [rows[i] for i in range(len(rows)) if i in train_idx]
    test_rows = [rows[i] for i in range(len(rows)) if i not in train_idx]
    return train_rows, test_rows


def minimal_row(row):
    return {
        "path": row.get("path", ""),
        "experiment_id": row.get("experiment_id", ""),
        "source": row.get("source", ""),
    }


def save_split(path, train_rows, test_rows, seed, train_frac):
    split = {
        "seed": seed,
        "train_frac": train_frac,
        "train_rows": [minimal_row(r) for r in train_rows],
        "test_rows": [minimal_row(r) for r in test_rows],
    }
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    return split


def write_manifest_csv(path, rows):
    fieldnames = ["path", "rel_path", "experiment_id", "source", "n_frames", "n_beads", "has_labels", "has_both_classes"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_real_dataset(rows, target_length=5000, max_beads_per_experiment=None, seed=42):
    traces_list = []
    labels_list = []
    metadata = []
    rng = np.random.default_rng(seed)
    errors = []
    for row in rows:
        path = row.get("path", "").strip()
        if not path:
            continue
        try:
            data = prep_exp_traces_for_model(path, good_only=False, target_length=target_length, use_resample=True)
        except Exception as exc:
            errors.append({"path": path, "error": str(exc)})
            continue
        traces = np.asarray(data["traces"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.intp)
        bead_indices = np.asarray(data.get("bead_indices", np.arange(len(labels))), dtype=np.intp)
        if len(traces) == 0:
            continue
        idx = np.arange(len(traces))
        if max_beads_per_experiment is not None and len(idx) > max_beads_per_experiment:
            idx = rng.choice(idx, size=max_beads_per_experiment, replace=False)
        traces_list.append(traces[idx])
        labels_list.append(labels[idx])
        for local_i in idx:
            metadata.append({
                "source": "real",
                "experiment_id": row.get("experiment_id", os.path.basename(path)),
                "experiment_source": row.get("source", ""),
                "path": path,
                "bead_id": int(bead_indices[local_i]),
            })
    if not traces_list:
        return empty_dataset(target_length), errors
    traces = np.vstack(traces_list).astype(np.float64, copy=False)
    labels = np.concatenate(labels_list).astype(np.intp, copy=False)
    return {"traces": traces, "labels": labels, "metadata": metadata, "target_length": target_length}, errors


def empty_dataset(target_length):
    return {
        "traces": np.zeros((0, target_length), dtype=np.float64),
        "labels": np.array([], dtype=np.intp),
        "metadata": [],
        "target_length": target_length,
    }


def save_npz_dataset(path, dataset):
    traces = np.asarray(dataset["traces"], dtype=np.float64)
    labels = np.asarray(dataset["labels"], dtype=np.intp)
    traces_sktime = traces[:, np.newaxis, :]
    times = np.broadcast_to(np.arange(traces.shape[1], dtype=np.float64), traces.shape).copy()
    np.savez_compressed(
        path,
        traces=traces,
        traces_sktime=traces_sktime,
        traces_sktime_2d=traces,
        times=times,
        times_sktime=times,
        labels=labels,
        n_traces=len(labels),
        metadata=np.asarray(dataset.get("metadata", []), dtype=object),
        median_length=traces.shape[1] if traces.ndim == 2 else 0,
        max_length=traces.shape[1] if traces.ndim == 2 else 0,
        format_version="synthetic_eval_1.0",
    )


def load_npz_dataset(path):
    data = np.load(path, allow_pickle=True)
    if "traces_sktime_2d" in data:
        traces = np.asarray(data["traces_sktime_2d"], dtype=np.float64)
    elif "traces_sktime" in data:
        traces = np.asarray(data["traces_sktime"], dtype=np.float64)[:, 0, :]
    else:
        traces = np.asarray(data["traces"], dtype=np.float64)
        if traces.ndim == 3 and traces.shape[1] == 1:
            traces = traces[:, 0, :]
    labels = np.asarray(data["labels"], dtype=np.intp)
    metadata = []
    if "metadata" in data:
        metadata_raw = data["metadata"]
        if metadata_raw.shape == ():
            metadata = metadata_raw.item()
        else:
            metadata = metadata_raw.tolist()
    return {
        "traces": traces,
        "labels": labels,
        "metadata": metadata,
        "target_length": traces.shape[1] if traces.ndim == 2 else None,
    }


def class_counts(labels):
    labels = np.asarray(labels, dtype=np.intp)
    return {"bad_0": int(np.sum(labels == 0)), "good_1": int(np.sum(labels == 1)), "n": int(len(labels))}


def check_no_path_overlap(train_rows: Iterable[dict], test_rows: Iterable[dict]):
    train_paths = {os.path.abspath(r.get("path", "")) for r in train_rows}
    test_paths = {os.path.abspath(r.get("path", "")) for r in test_rows}
    overlap = sorted(p for p in train_paths.intersection(test_paths) if p)
    if overlap:
        raise ValueError(f"train/test experiment leakage: {overlap[:5]}")


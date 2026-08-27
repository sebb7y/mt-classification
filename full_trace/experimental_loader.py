import io
import os
import pickle
import re
from pathlib import Path

import numpy as np

from npy_loader import read_settings, load_traces

BEAD_KEY_PATTERN = re.compile(r"^Bead_(\d+)_z$")

def load_pickle(path):
    with open(path, "rb") as f:
        data = f.read()

    for attempt in [
        lambda: pickle.loads(data),
        lambda: pickle.loads(data, encoding="latin-1"),
    ]:
        try:
            return attempt()
        except (UnicodeDecodeError, AttributeError, TypeError):
            continue
        except ModuleNotFoundError as e:
            if "numpy._core" in str(e):
                break
            raise

    class NumPyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if "numpy._core" in module:
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)

    return NumPyCompatUnpickler(io.BytesIO(data)).load()

def bead_key_to_index(key):
    m = BEAD_KEY_PATTERN.match(key)
    if m is None:
        return None
    return int(m.group(1)) - 1

def find_pickle_path(exp_folder):
    exp_folder = os.path.abspath(exp_folder)
    found = []
    for dirpath, _dirnames, filenames in os.walk(exp_folder):
        rel = os.path.relpath(dirpath, exp_folder)
        depth = 0 if rel == "." else len(Path(rel).parts)
        for name in ("traces_filtered.pickle", "traces_selected.pickle"):
            if name in filenames:
                path = os.path.join(dirpath, name)
                found.append((path, name == "traces_filtered.pickle", depth))
    if not found:
        return None
    found.sort(key=lambda x: (
        "original_prepost" not in x[0],
        not x[1],
        x[2],
    ))
    return found[0][0]

def find_all_pickle_paths(exp_folder):
    exp_folder = os.path.abspath(exp_folder)
    found = []
    for dirpath, _dirnames, filenames in os.walk(exp_folder):
        rel = os.path.relpath(dirpath, exp_folder)
        depth = 0 if rel == "." else len(Path(rel).parts)
        for name in ("traces_filtered.pickle", "traces_selected.pickle"):
            if name in filenames:
                path = os.path.join(dirpath, name)
                found.append((path, name == "traces_filtered.pickle", depth))
    if not found:
        return []
    found.sort(key=lambda x: (
        "original_prepost" not in x[0],
        not x[1],
        x[2],
    ))
    return [x[0] for x in found]

def read_good_bead_mask(exp_folder, n_beads):
    good_mask = np.ones(n_beads, dtype=bool)
    ordered_paths = find_all_pickle_paths(exp_folder)
    if not ordered_paths:
        return good_mask

    good_indices = set()
    for pickle_path in ordered_paths:
        data = load_pickle(pickle_path)
        good_indices = set()
        for key in data:
            if "Bead" not in str(key):
                continue
            idx = bead_key_to_index(str(key))
            if idx is not None:
                good_indices.add(idx)
        if good_indices:
            break
    if not good_indices:
        return good_mask

    good_mask[:] = False
    for i in good_indices:
        if 0 <= i < n_beads:
            good_mask[i] = True
    return good_mask

def load_z_over_time(exp_folder, traces_path=None, correction_factor=None, reference_id=None, framerate=None):
    if traces_path is None:
        traces_path = os.path.join(exp_folder, "traces.npy")
    if not os.path.isfile(traces_path):
        raise FileNotFoundError(f"traces.npy not found: {traces_path}")

    config_folder = os.path.dirname(os.path.abspath(traces_path))
    settings, ref_cfg, corr_cfg, fr_cfg = read_settings(config_folder)
    if framerate is None:
        framerate = fr_cfg
    if correction_factor is None:
        correction_factor = corr_cfg
    if reference_id is None:
        reference_id = ref_cfg

    traces = load_traces(traces_path)
    traces = traces / 1000.0
    reshaped = np.transpose(traces, (1, 2, 0))[:, 2, :]
    reshaped = reshaped * (-float(correction_factor))

    n_beads, n_frames = reshaped.shape
    ref_int = int(reference_id)
    if ref_int < 0:
        ref_int = n_beads + ref_int
    if ref_int >= n_beads:
        ref_int = n_beads - 1
    if reference_id and 0 <= ref_int < n_beads:
        reshaped = reshaped - reshaped[ref_int, :]

    time_s = np.arange(n_frames, dtype=np.float64) / float(framerate)
    z = reshaped.T
    return z, time_s

class ExpTraceData:
    def __init__(self, z, time_s, good_mask, n_beads, n_frames, exp_folder, traces_path):
        self.z = z
        self.time_s = time_s
        self.good_mask = good_mask
        self.n_beads = n_beads
        self.n_frames = n_frames
        self.exp_folder = exp_folder
        self.traces_path = traces_path

    def goodz(self):
        return self.z[:, self.good_mask]

    def good_indices(self):
        return np.where(self.good_mask)[0]

def load_exp_traces(exp_folder, traces_path=None, correction_factor=None, reference_id=None, framerate=None):
    if traces_path is None:
        traces_path = os.path.join(exp_folder, "traces.npy")

    z, time_s = load_z_over_time(
        exp_folder,
        traces_path,
        correction_factor=correction_factor,
        reference_id=reference_id,
        framerate=framerate,
    )
    n_frames, n_beads = z.shape
    good_mask = read_good_bead_mask(exp_folder, n_beads)

    return ExpTraceData(
        z=z,
        time_s=time_s,
        good_mask=good_mask,
        n_beads=n_beads,
        n_frames=n_frames,
        exp_folder=os.path.abspath(exp_folder),
        traces_path=os.path.abspath(traces_path),
    )

def resample_trace(trace, target_length):
    trace = np.asarray(trace, dtype=np.float64)
    if len(trace) == target_length:
        return trace.copy()
    x_old = np.linspace(0, 1, len(trace), dtype=np.float64)
    x_new = np.linspace(0, 1, target_length, dtype=np.float64)
    return np.interp(x_new, x_old, trace)

def split_trace_into_segments(trace, segment_length, stride=None):
    trace = np.asarray(trace, dtype=np.float64)
    stride = stride if stride is not None else segment_length
    n = len(trace)
    if n < segment_length:
        return np.empty((0, segment_length), dtype=np.float64)
    starts = np.arange(0, n - segment_length + 1, stride)
    segments = np.array([trace[s : s + segment_length] for s in starts])
    return segments

def prep_exp_traces_for_model(exp_folder, traces_path=None, good_only=False, target_length=None, pad_mode='edge', use_resample=False, training_stats=None):
    data = load_exp_traces(exp_folder, traces_path=traces_path)
    z = data.z
    time_s = data.time_s
    good_mask = data.good_mask
    n_frames, n_beads = z.shape

    if good_only:
        z = data.goodz()
        good_mask_sub = np.ones(z.shape[1], dtype=bool)
        bead_indices = data.good_indices()
    else:
        bead_indices = np.arange(n_beads)
        good_mask_sub = good_mask

    n_traces = z.shape[1]
    if target_length is None:
        target_length = n_frames

    traces_list = []
    for j in range(n_traces):
        trace = z[:, j].astype(np.float64)
        if np.any(np.isnan(trace)):
            fill = np.nanmean(trace)
            if np.isnan(fill):
                fill = 0.0
            trace = np.nan_to_num(trace, nan=fill, posinf=0.0, neginf=0.0)
        if use_resample and target_length is not None:
            trace = resample_trace(trace, target_length)
        elif len(trace) > target_length:
            trace = trace[:target_length]
        elif len(trace) < target_length:
            trace = np.pad(trace, (0, target_length - len(trace)), mode=pad_mode)
        traces_list.append(trace)

    traces = np.array(traces_list)
    labels = good_mask_sub.astype(np.int64)

    if training_stats is not None and "mean" in training_stats and "std" in training_stats:
        exp_mean = np.nanmean(traces)
        exp_std = np.nanstd(traces)
        if exp_std <= 0:
            exp_std = 1.0
        train_mean = float(training_stats["mean"])
        train_std = float(training_stats["std"])
        if train_std <= 0:
            train_std = 1.0
        traces = (traces - exp_mean) / exp_std * train_std + train_mean

    out = {"traces": traces, "labels": labels, "good_mask": good_mask_sub}
    out["time_s"] = time_s
    out["n_traces"] = n_traces
    out["target_length"] = target_length
    out["exp_folder"] = data.exp_folder
    out["bead_indices"] = bead_indices
    return out

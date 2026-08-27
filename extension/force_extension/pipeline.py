import os
import numpy as np
import pandas as pd

from . import config
from .load import discover, load_exp_from_txt, load_exp_from_npy
from .preprocess import attach_script_regions
from .process import energy_mat_for_ext, cluster_ext, rot_extension_labels, cluster_rot_extension

def clip_to_region_s(time_s, region, *arrs):
    time_s = np.asarray(time_s).flatten()
    t0, t1 = region[0], region[1]
    mask = (time_s >= t0) & (time_s <= t1)
    ind = np.where(mask)[0]
    if len(ind) == 0:
        return [None] * len(arrs), None
    start, end = int(ind[0]), int(ind[-1]) + 1
    out = []
    for a in arrs:
        if a is None:
            out.append(None)
        elif isinstance(a, pd.DataFrame):
            out.append(a.iloc[start:end].reset_index(drop=True))
        elif isinstance(a, np.ndarray):
            a = np.asarray(a)
            if a.ndim == 1:
                out.append(a[start:end])
            else:
                out.append(a[start:end])
        else:
            out.append(a[start:end] if hasattr(a, "__getitem__") else a)
    time_clipped = time_s[start:end]
    return out, time_clipped

def _trim_ext_to_length(ext, min_len):
    ext["gb_df"] = ext["gb_df"].iloc[:min_len].reset_index(drop=True)
    if ext.get("gb_time") is not None:
        ext["gb_time"] = ext["gb_time"].iloc[:min_len].reset_index(drop=True)
    if ext.get("forces") is not None and len(ext["forces"]) > 0:
        ext["forces"] = ext["forces"][:min_len]
    if ext.get("all_beads_df") is not None:
        ext["all_beads_df"] = ext["all_beads_df"].iloc[:min_len].reset_index(drop=True)
    if ext.get("mag_rots") is not None and len(ext["mag_rots"]) > 0:
        ext["mag_rots"] = ext["mag_rots"][:min_len]


def build_exts_from_exp(experiment, region_type='force'):
    if region_type == "force":
        regions = experiment.get("force_extension_regions") or []
    elif region_type == "rotation":
        regions = experiment.get("rotation_regions") or []
    else:
        regions = experiment.get("rel_regions") or []
    if not regions:
        return []

    gb_df = experiment.get("gb_df")
    gb_time = experiment.get("gb_time")
    exp_time = experiment.get("exp_time")
    exp_z = experiment.get("exp_z")

    if gb_df is None or len(gb_df) == 0:
        if exp_z is None or len(exp_z) == 0:
            return []
        gb_df = exp_z
    if gb_time is None or len(gb_time) == 0:
        gb_time = exp_time
    if gb_time is None or len(gb_time) == 0:
        return []

    time_col = gb_time.iloc[:, 0]
    time_s = np.asarray(time_col, dtype=float)
    forces = experiment.get("forces")
    mag_rots = experiment.get("mag_rots")
    all_df = experiment.get("all_df")
    exp_z = experiment.get("exp_z")
    has_all_df = all_df is not None and len(all_df) > 0
    beads_df_for_clustering = all_df if has_all_df else exp_z

    extensions = []
    for reg in regions:
        to_clip = [gb_df, forces]
        if beads_df_for_clustering is not None and len(beads_df_for_clustering) == len(time_s):
            to_clip.append(beads_df_for_clustering)
        if region_type == "rotation" and mag_rots is not None and len(mag_rots) == len(time_s):
            to_clip.append(np.asarray(mag_rots).flatten())

        clipped_list, time_clipped = clip_to_region_s(time_s, reg, *to_clip)
        if clipped_list[0] is None or len(clipped_list[0]) == 0:
            continue

        gb_clipped = clipped_list[0]
        forces_clipped = clipped_list[1]
        if forces_clipped is not None and hasattr(forces_clipped, "flatten"):
            forces_clipped = np.asarray(forces_clipped).flatten()
        all_beads_clipped = clipped_list[2] if len(to_clip) >= 3 else None
        mag_rots_clipped = None
        if region_type == "rotation" and len(to_clip) == 4:
            mag_rots_clipped = clipped_list[-1]

        gb_time_val = pd.DataFrame(time_clipped, columns=[0]) if time_clipped is not None else None
        ext = {
            "region": reg,
            "region_kind": region_type,
            "gb_df": gb_clipped,
            "forces": forces_clipped if forces_clipped is not None else np.array([]),
            "gb_time": gb_time_val,
        }
        if all_beads_clipped is not None:
            ext["all_beads_df"] = all_beads_clipped
        if mag_rots_clipped is not None:
            ext["mag_rots"] = np.asarray(mag_rots_clipped).flatten()

        min_len = len(ext["gb_df"])
        if ext["gb_time"] is not None:
            min_len = min(min_len, len(ext["gb_time"]))
        if ext["forces"] is not None and len(ext["forces"]) > 0:
            min_len = min(min_len, len(ext["forces"]))
        if min_len == 0:
            continue

        _trim_ext_to_length(ext, min_len)
        extensions.append(ext)
    return extensions

def load_exp(path, require_script=True, **kwargs):
    path = os.path.abspath(path)
    if os.path.isfile(path) and path.endswith(".npy"):
        root = os.path.dirname(path)
        return load_exp_from_npy(root, npy_path=path, require_script=require_script)
    if os.path.isdir(path):
        manifest = discover(path, require_traces=False, require_script=False)
        paths = manifest.get("paths") or {}
        if paths.get("traces_npy"):
            return load_exp_from_npy(path, require_script=require_script)
        return load_exp_from_txt(path, require_script=require_script, **kwargs)
    if os.path.isfile(path):
        root = os.path.dirname(path)
        return load_exp_from_txt(root, require_script=require_script, **kwargs)
    raise FileNotFoundError(f"not found: {path}")
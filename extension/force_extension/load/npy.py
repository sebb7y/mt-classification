import os

import numpy as np
import pandas as pd

from .npy_loader import load_npy_data, load_settings
from .. import config
from .discovery import discover, has_traces_beadpos, has_beadpos_only, beadcount

def read_beadpos_xy(filepath):
    out = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x, y = parts[0].strip(), parts[1].strip()
                    if x == "" or y == "":
                        continue
                    out.append((int(x), int(y)))
                except (ValueError, TypeError):
                    continue
    return out

def npy_subdirs_traces_beadpos(folder_path):
    if not os.path.isdir(folder_path):
        return []
    out = []
    for name in os.listdir(folder_path):
        sub = os.path.join(folder_path, name)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "traces.npy")) and os.path.isfile(os.path.join(sub, "beadpos_xy.txt")):
            out.append(name)
    return out

def npy_two_subdirs_all_good_separate(root):
    if not os.path.isdir(root):
        return None
    subdirs = [n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)) and not n.startswith(".")]
    if len(subdirs) != 2:
        return None
    p0, p1 = os.path.join(root, subdirs[0]), os.path.join(root, subdirs[1])
    has_t0 = has_traces_beadpos(p0)
    has_t1 = has_traces_beadpos(p1)
    if has_t0 and (has_beadpos_only(p1) or has_traces_beadpos(p1)):
        return subdirs[0], subdirs[1]
    if has_t1 and (has_beadpos_only(p0) or has_traces_beadpos(p0)):
        return subdirs[1], subdirs[0]
    return None

def get_all_df_safe(gb, bb):
    gb_df = gb.copy()
    gb_df.columns = ["gb_" + str(col) for col in gb_df.columns]
    bb_df = bb.copy()
    bb_df.columns = ["bb_" + str(col) for col in bb_df.columns]
    return pd.concat([gb_df, bb_df], axis=1)

def load_exp_from_npy(root, npy_path=None, require_script=True):
    root = os.path.abspath(root)
    manifest = discover(root, require_traces=False, require_script=False)
    paths = manifest.get("paths", {})

    if npy_path and os.path.isfile(npy_path):
        traces_npy = npy_path
        subdir_for_script = None
    elif "traces_npy" in paths:
        traces_npy = paths["traces_npy"]
        subdir_for_script = paths.get("traces_npy_subdir")
    else:
        for cand in config.TRACES_NPY_CANDIDATES:
            p = os.path.join(root, cand)
            if os.path.isfile(p):
                traces_npy = p
                break
        else:
            raise FileNotFoundError(f"no traces.npy found under {root}")

    folder_path = root
    subdirs = npy_subdirs_traces_beadpos(root)
    root_has_traces_beadpos = has_traces_beadpos(root)
    separate = npy_two_subdirs_all_good_separate(root)

    if len(subdirs) == 2:
        n0, n1 = beadcount(os.path.join(root, subdirs[0])), beadcount(os.path.join(root, subdirs[1]))
        all_subdir, good_subdir = (subdirs[0], subdirs[1]) if n0 >= n1 else (subdirs[1], subdirs[0])
        all_path = os.path.join(root, all_subdir)
        good_path = os.path.join(root, good_subdir)

        all_positions = read_beadpos_xy(os.path.join(all_path, "beadpos_xy.txt"))
        good_positions = set(read_beadpos_xy(os.path.join(good_path, "beadpos_xy.txt")))
        good_indices = [i for i, p in enumerate(all_positions) if p in good_positions]
        bad_indices = [i for i in range(len(all_positions)) if i not in good_indices]

        npy_path_all = os.path.join(all_path, "traces.npy")
        exp_z, exp_time, exp_x, exp_y = load_npy_data(npy_path_all)

        gb_df = exp_z.iloc[:, good_indices].copy()
        gb_df.columns = range(gb_df.shape[1])
        bb_df = exp_z.iloc[:, bad_indices].copy()
        bb_df.columns = range(bb_df.shape[1])

        traces_dir_for_script = all_path
        good_bead_indices = good_indices

    elif separate is not None:
        all_subdir, good_subdir = separate
        all_path = os.path.join(root, all_subdir)
        good_path = os.path.join(root, good_subdir)

        all_positions = read_beadpos_xy(os.path.join(all_path, "beadpos_xy.txt"))
        good_positions = set(read_beadpos_xy(os.path.join(good_path, "beadpos_xy.txt")))
        good_indices = [i for i, p in enumerate(all_positions) if p in good_positions]
        bad_indices = [i for i in range(len(all_positions)) if i not in good_indices]

        npy_path_all = os.path.join(all_path, "traces.npy")
        exp_z, exp_time, exp_x, exp_y = load_npy_data(npy_path_all)

        gb_df = exp_z.iloc[:, good_indices].copy()
        gb_df.columns = range(gb_df.shape[1])
        bb_df = exp_z.iloc[:, bad_indices].copy()
        bb_df.columns = range(bb_df.shape[1])

        traces_dir_for_script = all_path
        good_bead_indices = good_indices

    elif len(subdirs) == 1 and (root_has_traces_beadpos or has_beadpos_only(root)):
        all_path = os.path.join(root, subdirs[0])
        good_path = root

        all_positions = read_beadpos_xy(os.path.join(all_path, "beadpos_xy.txt"))
        good_positions = set(read_beadpos_xy(os.path.join(good_path, "beadpos_xy.txt")))
        good_indices = [i for i, p in enumerate(all_positions) if p in good_positions]
        bad_indices = [i for i in range(len(all_positions)) if i not in good_indices]

        npy_path_all = os.path.join(all_path, "traces.npy")
        exp_z, exp_time, exp_x, exp_y = load_npy_data(npy_path_all)

        gb_df = exp_z.iloc[:, good_indices].copy()
        gb_df.columns = range(gb_df.shape[1])
        bb_df = exp_z.iloc[:, bad_indices].copy()
        bb_df.columns = range(bb_df.shape[1])

        traces_dir_for_script = all_path
        for script_dir in (root, all_path):
            if any(
                os.path.isfile(os.path.join(script_dir, c))
                for c in config.SCRIPT_NPY_CANDIDATES + config.CFG_YAML_CANDIDATES
            ):
                traces_dir_for_script = script_dir
                break

        good_bead_indices = good_indices

    else:
        exp_z, exp_time, exp_x, exp_y = load_npy_data(traces_npy)
        gb_df = pd.DataFrame()
        bb_df = pd.DataFrame()
        good_bead_indices = None
        traces_dir_for_script = os.path.dirname(traces_npy)

    script_txt_path = None
    for cand in config.SCRIPT_NPY_CANDIDATES:
        p = os.path.join(traces_dir_for_script, cand)
        if os.path.isfile(p):
            script_txt_path = p
            break

    gb_time = exp_time
    all_df = get_all_df_safe(gb_df, bb_df) if len(gb_df) > 0 and len(bb_df) > 0 else pd.DataFrame()

    time_s = np.asarray(exp_time.iloc[:, 0])
    trace_data = {
        "z": exp_z.values,
        "time": np.broadcast_to(time_s.reshape(-1, 1), exp_z.shape),
        "x": exp_x.values,
        "y": exp_y.values
    }

    try:
        _settings, ref_id, corr, fr = load_settings(traces_dir_for_script)
        framerate = float(fr) if fr is not None else 58.0
        reference_id = ref_id
        correction_factor = corr
    except Exception:
        framerate = 58.0
        reference_id = -1
        correction_factor = 0.88


    exp_config = {
        'folder_path': folder_path,
        'framerate': framerate,
        'reference_id': reference_id,
        'correction_factor': correction_factor
    }

    path_built = {'traces_npy': traces_npy}
    path_built.update(paths)
    if script_txt_path:
        path_built['script'] = script_txt_path
        path_built['script_npy_path'] = script_txt_path
    paths = path_built

    experiment = {
        'trace_data': trace_data,
        'source': 'npy',
        'config': exp_config,
        'folder_path': folder_path,
        'exp_z': exp_z,
        'exp_time': exp_time,
        'exp_x': exp_x,
        'exp_y': exp_y,
        'gb_df': gb_df,
        'bb_df': bb_df,
        'all_df': all_df,
        'gb_time': gb_time,
        'paths': paths
    }

    if good_bead_indices is not None:
        experiment["good_bead_indices"] = good_bead_indices
    return experiment

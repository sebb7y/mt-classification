import os
import re
import numpy as np
import pandas as pd
from .. import config as _config
from .discovery import discover



def get_ref_id_from_cfg(root):
    default_ref = -1
    default_corr = 1.0
    for cand in _config.CFG_YAML_CANDIDATES:
        path = os.path.join(root, cand)
        if os.path.isfile(path):
            try:
                import yaml
                with open(path, "r") as f:
                    settings = yaml.safe_load(f) or {}
            except Exception:
                return default_ref, default_corr
            ref_val = settings.get("refIndices", default_ref)
            if ref_val is None:
                return default_ref, default_corr
            if isinstance(ref_val, (int, float)):
                reference_id = int(ref_val)
            elif isinstance(ref_val, list):
                reference_id = int(ref_val[-1])
            else:
                ref_list = str(ref_val).strip().split(",")
                reference_id = int(ref_list[-1].strip().strip("'").strip('"'))
            corr = settings.get("zcorrection", settings.get("processing_zcorrection"))
            correction_factor = float(corr) if corr is not None else default_corr
            return reference_id, correction_factor
    return default_ref, default_corr


def apply_reference_subtraction_z(exp_z, reference_id):
    n = exp_z.shape[1]
    ref_idx = (n - 1) if reference_id == -1 else reference_id
    if ref_idx < 0 or ref_idx >= n:
        return exp_z
    exp_z = exp_z.copy()
    ref_col = exp_z.iloc[:, ref_idx].values.reshape(-1, 1)
    exp_z.iloc[:, :] = exp_z.values - ref_col
    return exp_z

def read_exp_file(filepath):
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        na_values=["-1.#IND000"],
    )
    time_df = df[[1]]
    df = df.drop(columns=[0, 1])
    df.columns = range(df.shape[1])
    return df, time_df

def read_goodbeads_file(filepath):
    df = pd.read_csv(
        filepath,
        sep="\t",
        na_values=["-1.#IND000"],
    )
    time_df = df[["time(ms)"]]
    df = df.drop(columns=["time(ms)"])
    return df, time_df

def split_exp_df(df):
    n = df.shape[1]
    df_x = df.iloc[:, 0:n:3]
    df_y = df.iloc[:, 1:n:3]
    df_z = df.iloc[:, 2:n:3]
    df_x.columns = range(df_x.shape[1])
    df_y.columns = range(df_y.shape[1])
    df_z.columns = range(df_z.shape[1])
    return df_x, df_y, df_z

def get_bb_df(exp_z, gb_df):
    cols = gb_df.columns.tolist()
    col_nums = [int(re.search(r"\d+", str(c)).group()) - 1 for c in cols]
    return exp_z.drop(exp_z.columns[col_nums], axis=1)

def get_all_df(gb, bb, safe=False):
    gb_df = gb.copy()
    if safe:
        gb_df.columns = ["gb_" + str(col) for col in gb_df.columns]
    else:
        gb_df.columns = [
            "gb_" + str(int(str(col).split("_")[1]) - 1) for col in gb_df.columns
        ]
    bb_df = bb.copy()
    bb_df.columns = ["bb_" + str(col) for col in bb_df.columns]
    return pd.concat([gb_df, bb_df], axis=1)

def get_all_df_coilable_uncoilable(exp_z, gc_path, uc_path):
    gc_df, _ = read_goodbeads_file(gc_path)
    uc_df, _ = read_goodbeads_file(uc_path)
    gc_col_nums = [int(re.search(r"\d+", str(c)).group()) - 1 for c in gc_df.columns]
    uc_col_nums = [int(re.search(r"\d+", str(c)).group()) - 1 for c in uc_df.columns]

    drop_cols = sorted(set(gc_col_nums) | set(uc_col_nums))

    gc_df = gc_df.copy()
    gc_df.columns = [f"gc_{i}" for i in range(gc_df.shape[1])]
    uc_df = uc_df.copy()
    uc_df.columns = [f"uc_{i}" for i in range(uc_df.shape[1])]
    bb_df = exp_z.drop(exp_z.columns[drop_cols], axis=1)
    bb_df.columns = [f"bb_{i}" for i in range(bb_df.shape[1])]
    all_df = pd.concat([gc_df, uc_df, bb_df], axis=1)
    return gc_df, uc_df, bb_df, all_df

def load_exp_from_txt(root, require_script=True, require_good_beads=False, goodbeads_coilable_path=None, goodbeads_uncoilable_path=None):
    manifest = discover(
        root,
        require_traces=True,
        require_script=require_script,
        require_good_beads=require_good_beads,
    )
    if manifest.get("errors"):
        raise FileNotFoundError("; ".join(manifest["errors"]))

    traces_path = manifest["paths"].get("traces_txt")
    if not traces_path:
        raise FileNotFoundError(f"no traces file found under {root}")

    exp_df, exp_time = read_exp_file(traces_path)
    exp_x, exp_y, exp_z = split_exp_df(exp_df)

    folder_path = os.path.abspath(root)
    reference_id, correction_factor = get_ref_id_from_cfg(folder_path)
    exp_z = apply_reference_subtraction_z(exp_z, reference_id)

    time_col = exp_time.iloc[:, 0].values.astype(float)
    time_s = time_col / 1000.0
    goodbeads_path = manifest["paths"].get("goodbeads_txt")

    use_coilable_uncoilable = (
        goodbeads_coilable_path and goodbeads_uncoilable_path
        and os.path.isfile(goodbeads_coilable_path)
        and os.path.isfile(goodbeads_uncoilable_path)
    )

    if use_coilable_uncoilable:
        gc_df, uc_df, bb_df, all_df = get_all_df_coilable_uncoilable(
            exp_z, goodbeads_coilable_path, goodbeads_uncoilable_path
        )
        gb_df = gc_df
        gb_time_df = pd.DataFrame(time_s, columns=[0])
        _gc_raw, _ = read_goodbeads_file(goodbeads_coilable_path)
        good_bead_indices = [int(re.search(r"\d+", str(c)).group()) - 1 for c in _gc_raw.columns]

    elif goodbeads_path and os.path.isfile(goodbeads_path):
        gb_df, gb_time = read_goodbeads_file(goodbeads_path)
        bb_df = get_bb_df(exp_z, gb_df)
        all_df = get_all_df(gb_df, bb_df, safe=False)
        gb_time_col = gb_time.iloc[:, 0].values.astype(float)
        gb_time_s = gb_time_col / 1000.0
        gb_time_df = pd.DataFrame(gb_time_s, columns=[0])
        good_bead_indices = [int(re.search(r"\d+", str(c)).group()) - 1 for c in gb_df.columns]

    else:
        gb_df = pd.DataFrame()
        bb_df = pd.DataFrame()
        all_df = pd.DataFrame()
        gb_time_df = pd.DataFrame(time_s, columns=[0])
        good_bead_indices = None

    trace_data = {
        "z": exp_z.values,
        "time": np.broadcast_to(time_s.reshape(-1, 1), (len(time_s), exp_z.shape[1])),
        "x": exp_x.values,
        "y": exp_y.values
    }

    experiment = {
        'trace_data': trace_data,
        'source': 'txt',
        'config': {
            'folder_path': folder_path,
            'framerate': 58.0,
            'reference_id': reference_id,
            'correction_factor': correction_factor
        },
        'folder_path': folder_path,
        'exp_z': exp_z,
        'exp_time': pd.DataFrame(time_s, columns=[0]),
        'exp_x': exp_x,
        'exp_y': exp_y,
        'gb_df': gb_df,
        'bb_df': bb_df,
        'all_df': all_df,
        'gb_time': gb_time_df,
        'paths': manifest['paths']
    }
    if good_bead_indices is not None:
        experiment["good_bead_indices"] = good_bead_indices
    return experiment

import numpy as np
import pandas as pd

from .. import config

def get_bead_df(ext, use_all_beads=True):
    all_beads_df = ext.get("all_beads_df")
    if use_all_beads and all_beads_df is not None and len(all_beads_df) > 0:
        return all_beads_df
    return ext["gb_df"]

def get_aligned_force_ext_inputs(ext, use_all_beads=True):
    df = get_bead_df(ext, use_all_beads=use_all_beads)
    forces = np.asarray(ext["forces"]).flatten()
    if len(forces) == len(df):
        return forces, df

    min_len = min(len(forces), len(df))
    return forces[:min_len], df.iloc[:min_len]

def energy_from_force_trapz(force, extension, sort_input=True, remove_nan=True):
    f = np.asarray(force).flatten()
    x = np.asarray(extension).flatten()
    if f.shape != x.shape:
        raise ValueError("force and extension must have the same shape")
    if remove_nan:
        mask = ~(np.isnan(f) | np.isnan(x))
        f = f[mask]
        x = x[mask]
    if sort_input:
        idx = np.argsort(x)
        x = x[idx]
        f = f[idx]

    E = np.zeros_like(x) # array of zero same shape as x
    if x.size >= 2:
        dx = np.diff(x)
        nonzero = dx != 0
        trape = 0.5 * (f[1:] + f[:-1]) * dx
        trape[~nonzero] = 0.0
        E[1:] = np.cumsum(trape)
    return x, E

def detect_outlier_beads_range(df, z_thresh=3.0):
    mins = df.min(axis=0)
    maxs = df.max(axis=0)
    ranges = maxs - mins
    mean_range = ranges.mean()
    std_range = ranges.std()
    if std_range == 0 or not np.isfinite(std_range):
        std_range = 1.0
    threshold_low = mean_range - z_thresh * std_range
    threshold_high = mean_range + z_thresh * std_range
    good_mask = (ranges >= threshold_low) & (ranges <= threshold_high)
    good_cols = df.columns[good_mask].tolist()
    bad_cols = df.columns[~good_mask].tolist()
    return good_cols, bad_cols, mins, maxs

def energy_mat_for_ext(ext, n_grid=None, z_thresh=None, sort_input=True, remove_nan=True, verbose=False, use_all_beads=True, use_full_extension_range=True, exclude_obvious_outliers=True):
    if n_grid is None:
        n_grid = config.DEFAULT_N_GRID
    if z_thresh is None:
        z_thresh = config.DEFAULT_Z_THRESH_RANGE

    forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
    if len(df) == 0:
        return np.array([]), np.empty((0, 0)), [], [], {"by_range": [], "by_energy": []}

    good_cols, bad_cols_range, mins, maxs = detect_outlier_beads_range(df, z_thresh=z_thresh)
    cols_to_try = good_cols if exclude_obvious_outliers else list(df.columns)
    if len(cols_to_try) == 0:
        return np.array([]), np.empty((0, 0)), [], list(df.columns), {"by_range": list(df.columns), "by_energy": []}

    if use_full_extension_range:
        min_common = float(np.nanmin([mins[col] for col in cols_to_try]))
        max_common = float(np.nanmax([maxs[col] for col in cols_to_try]))
    else:
        min_common = max(mins[col] for col in cols_to_try)
        max_common = min(maxs[col] for col in cols_to_try)
    if min_common >= max_common or not np.isfinite(min_common) or not np.isfinite(max_common):
        min_common = float(np.nanmean([mins[col] for col in cols_to_try]))
        max_common = float(np.nanmean([maxs[col] for col in cols_to_try]))
    x_grid = np.linspace(min_common, max_common, n_grid)

    E_rows = []
    cols_kept = []
    for col in cols_to_try:
        extension = df[col].values
        if len(forces) != len(extension):
            min_len = min(len(forces), len(extension))
            forces_use = forces[:min_len]
            extension_use = extension[:min_len]
        else:
            forces_use = forces
            extension_use = extension
        x_bead, E_bead = energy_from_force_trapz(
            forces_use, extension_use, sort_input=sort_input, remove_nan=remove_nan
        )
        if x_bead.size < 2:
            continue
        if np.all(np.isnan(x_bead)) or np.all(np.isnan(E_bead)):
            continue
        E_interp = np.interp(x_grid, x_bead, E_bead, left=np.nan, right=np.nan)
        if np.all(np.isnan(E_interp)):
            continue
        E_rows.append(E_interp)
        cols_kept.append(col)

    if not E_rows:
        return x_grid, np.empty((0, n_grid)), [], bad_cols_range, {"by_range": bad_cols_range, "by_energy": []}

    E_arr = np.array(E_rows)
    max_e = np.nanmax(E_arr, axis=1)
    range_e = np.nanmax(E_arr, axis=1) - np.nanmin(E_arr, axis=1)
    range_e = np.where(np.isfinite(range_e), range_e, 0.0)
    med_max = np.median(max_e)
    mad_max = np.median(np.abs(max_e - med_max))
    # if mad is 0 we use std, otherwise we use mad

    mad_max = mad_max if mad_max > 0 else (np.nanstd(max_e) or 1.0)
    med_range = np.median(range_e)
    mad_range = np.median(np.abs(range_e - med_range))
    mad_range = mad_range if mad_range > 0 else (np.nanstd(range_e) or 1.0)
    z_energy = 2.5 # need to tune


    ok_max = (max_e >= med_max - z_energy * mad_max) & (max_e <= med_max + z_energy * mad_max)
    ok_range = (range_e >= med_range - z_energy * mad_range) & (range_e <= med_range + z_energy * mad_range)
    keep = ok_max & ok_range
    E_rows = [E_rows[i] for i in range(len(E_rows)) if keep[i]]
    excluded_by_energy = [cols_kept[i] for i in range(len(cols_kept)) if not keep[i]]
    cols_kept = [c for i, c in enumerate(cols_kept) if keep[i]]
    bad_cols = bad_cols_range + excluded_by_energy
    exclusion_info = {"by_range": bad_cols_range, "by_energy": excluded_by_energy}

    E_mat = np.vstack(E_rows) if E_rows else np.empty((0, n_grid))
    return x_grid, E_mat, cols_kept, bad_cols, exclusion_info

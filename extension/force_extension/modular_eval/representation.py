import numpy as np
import pandas as pd

from ..process.energy import energy_from_force_trapz, energy_mat_for_ext as _build_energy_matrix_for_ext
from ..process.rot_analysis import build_rot_curve_matrix

ROT_REPRESENTATIONS = frozenset({
    "rotation_grid", "common_grid_intersection", "common_grid_union",
    "percentile_grid", "arc_length_percentile", "landmark_quartiles"
})

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

def stack_feature_rows(rows, n_features):
    if rows:
        return np.vstack(rows)
    return np.empty((0, n_features))

def normalise_if_possible(values):
    value_range = np.nanmax(values) - np.nanmin(values)
    if value_range <= 0:
        return values
    return (values - np.nanmin(values)) / value_range

def compute_slope(values, sample_grid):
    delta_values = np.diff(values)
    delta_grid = np.diff(sample_grid)

    slope = np.zeros_like(values)
    slope[:-1] = delta_values / np.where(delta_grid != 0, delta_grid, np.nan)
    slope[-1] = slope[-2] if len(slope) > 1 else 0.0
    return slope

def build_energy_pct_curve(forces, ext_col, percentile_grid):
    x_bead, E_bead = build_energy_curve(forces, ext_col)
    if x_bead is None:
        return None

    bead_percentiles = np.linspace(0, 100, x_bead.size)
    curve = np.interp(percentile_grid, bead_percentiles, E_bead, left=np.nan, right=np.nan)
    if np.all(np.isnan(curve)):
        return None

    return curve

def build_energy_curve(forces, ext_col):
    x_bead, E_bead = energy_from_force_trapz(
        forces[: len(ext_col)],
        ext_col,
        sort_input=True,
        remove_nan=True,
    )
    if x_bead.size < 2:
        return None, None
    return x_bead, E_bead

def resample_curve(values, target_len):
    values_len = len(values)
    if values_len <= target_len:
        out = np.full(target_len, np.nan, dtype=float)
        out[:values_len] = values
        return out

    old_idx = np.linspace(0, values_len - 1, target_len)
    return np.interp(old_idx, np.arange(values_len), values)

def build_force_pct_curve(forces, ext_col, percentile_grid):
    x_bead = np.asarray(ext_col, dtype=float)
    n_points = len(x_bead)
    if n_points < 2:
        return None

    f_bead = np.asarray(forces[:n_points], dtype=float)
    valid = np.isfinite(f_bead) & np.isfinite(x_bead)
    if np.sum(valid) < 2:
        return None

    x_bead = x_bead[valid]
    f_bead = f_bead[valid]

    sort_idx = np.argsort(x_bead)
    x_bead = x_bead[sort_idx]
    f_bead = f_bead[sort_idx]

    bead_percentiles = np.linspace(0, 100, x_bead.size)
    curve = np.interp(percentile_grid, bead_percentiles, f_bead, left=np.nan, right=np.nan)
    if np.all(np.isnan(curve)):
        return None

    return curve

def get_sorted_ext_curve(ext_col, sort_idx):
    ext_values = np.asarray(ext_col, dtype=float)
    if np.sum(np.isfinite(ext_values)) < 2:
        return None

    ext_sorted = ext_values[sort_idx]
    if np.sum(np.isfinite(ext_sorted)) < len(ext_sorted) * 0.5:
        return None

    return ext_sorted

def build_rot_feature_matrix(ext, n_grid=200, representation='rotation_grid'):
    all_beads = ext.get("all_beads_df")
    if all_beads is None:
        all_beads = ext.get("gb_df")
    mag_rots = ext.get("mag_rots")
    if all_beads is None or mag_rots is None or len(all_beads) == 0:
        return np.empty((0, n_grid)), [], np.linspace(0, 1, n_grid), {"representation": "rotation_grid"}
    z = np.asarray(all_beads)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    mag_rots = np.asarray(mag_rots).flatten()
    n_time = min(len(mag_rots), z.shape[0])
    z = z[:n_time]
    mag_rots = mag_rots[:n_time]
    cols = all_beads.columns.tolist()
    n_beads = z.shape[1]

    if representation not in ROT_REPRESENTATIONS:
        representation = "rotation_grid"

    if representation in ("rotation_grid", "common_grid_intersection"):
        n_total = z.shape[0]
        sample_i = np.arange(n_total) if n_total <= n_grid else np.linspace(0, n_total - 1, n_grid, dtype=int)
        n_out = len(sample_i)
        rows_list, cols_kept = [], []
        for b in range(n_beads):
            z_b = z[sample_i, b].astype(float)
            if np.sum(np.isfinite(z_b)) < max(2, n_out * 0.5):
                continue
            rows_list.append(z_b)
            cols_kept.append(cols[b])
        X = np.vstack(rows_list) if rows_list else np.empty((0, n_out))
        time_grid = np.linspace(0, 1, n_out)
        meta = {"representation": representation, "n_grid": n_out}
        return X, cols_kept, time_grid, meta

    if representation == "common_grid_union":
        r_min, r_max = float(np.nanmin(mag_rots)), float(np.nanmax(mag_rots))
        margin = 0.05 * (r_max - r_min) if r_max > r_min else 1.0
        rot_grid, Z_mat, cols_kept_bool = build_rot_curve_matrix(
            z, mag_rots, n_grid=n_grid, rot_min=r_min - margin, rot_max=r_max + margin
        )
        cols_kept = [c for c, k in zip(cols, cols_kept_bool) if k]
        X = Z_mat[:, cols_kept_bool].T
        meta = {"representation": "common_grid_union", "n_grid": n_grid}
        return X, cols_kept, rot_grid, meta

    if representation == "percentile_grid":
        p_grid = np.linspace(0, 100, n_grid)
        rows_list, cols_kept = [], []
        for b in range(n_beads):
            rot_b = mag_rots.copy()
            z_b = z[:, b].astype(float)
            valid = np.isfinite(rot_b) & np.isfinite(z_b)
            if np.sum(valid) < 2:
                continue
            rot_b = rot_b[valid]
            z_b = z_b[valid]
            idx = np.argsort(rot_b)
            rot_b, z_b = rot_b[idx], z_b[idx]
            p_bead = np.linspace(0, 100, rot_b.size)
            z_at_p = np.interp(p_grid, p_bead, z_b, left=np.nan, right=np.nan)
            if np.all(np.isnan(z_at_p)):
                continue
            rows_list.append(z_at_p)
            cols_kept.append(cols[b])
        X = np.vstack(rows_list) if rows_list else np.empty((0, n_grid))
        meta = {"representation": "percentile_grid", "n_grid": n_grid}
        return X, cols_kept, p_grid, meta

    if representation == "arc_length_percentile":

        s_grid = np.linspace(0, 100, n_grid)
        rows_list, cols_kept = [], []
        for b in range(n_beads):
            rot_b = mag_rots.copy().astype(float)
            z_b = z[:, b].astype(float)
            valid = np.isfinite(rot_b) & np.isfinite(z_b)
            

            if np.sum(valid) < 2:
                continue
            rot_b = rot_b[valid]
            z_b = z_b[valid]
            idx = np.argsort(rot_b)
            rot_b, z_b = rot_b[idx], z_b[idx]


            dr = np.diff(rot_b)
            dz = np.diff(z_b)
            seg = np.sqrt(dr * dr + dz * dz)
            seg = np.where(np.isfinite(seg), seg, 0.0)
            arc_cum = np.zeros(rot_b.size)
            arc_cum[1:] = np.cumsum(seg)
            total = arc_cum[-1]
            if total <= 0:
                continue

            s_bead = 100.0 * arc_cum / total
            z_at_s = np.interp(s_grid, s_bead, z_b, left=np.nan, right=np.nan)
            if np.all(np.isnan(z_at_s)):
                continue
            rows_list.append(z_at_s)
            cols_kept.append(cols[b])

        X = np.vstack(rows_list) if rows_list else np.empty((0, n_grid))
        meta = {"representation": "arc_length_percentile", "n_grid": n_grid}
        return X, cols_kept, s_grid, meta

    if representation == "landmark_quartiles":
        features_list, cols_kept = [], []
        for b in range(n_beads):
            rot_b = mag_rots.copy().astype(float)
            z_b = z[:, b].astype(float)
            valid = np.isfinite(rot_b) & np.isfinite(z_b)
            if np.sum(valid) < 2:
                continue

        
            rot_b = rot_b[valid]
            z_b = z_b[valid]
            idx = np.argsort(rot_b)
            rot_b, z_b = rot_b[idx], z_b[idx]
            n_pt = rot_b.size


            p_target = np.array([25.0, 50.0, 75.0])
            p_bead = np.linspace(0, 100, n_pt)
            rot_land = np.interp(p_target, p_bead, rot_b, left=np.nan, right=np.nan)
            z_land = np.interp(p_target, p_bead, z_b, left=np.nan, right=np.nan)
            if np.any(np.isnan(rot_land)) or np.any(np.isnan(z_land)):
                continue
            features_list.append(np.concatenate([rot_land, z_land]))
            cols_kept.append(cols[b])


        X = np.vstack(features_list) if features_list else np.empty((0, 6))
        meta = {"representation": "landmark_quartiles", "n_grid": 6}
        return X, cols_kept, np.array([25.0, 50.0, 75.0]), meta

    rot_grid, Z_mat, cols_kept_bool = build_rot_curve_matrix(z, mag_rots, n_grid=n_grid)
    cols_kept = [c for c, k in zip(cols, cols_kept_bool) if k]
    X = Z_mat[:, cols_kept_bool].T
    meta = {"representation": "rotation_grid", "n_grid": n_grid}
    return X, cols_kept, rot_grid, meta

def feat_mat(ext, representation, n_grid=200, z_thresh=3.0, use_full_extension_range=False, exclude_obvious_outliers=True, use_all_beads=True):
    meta = {'representation': representation, 'exclusion_info': {}}
    use_union = use_full_extension_range

    if representation in ("common_grid_intersection", "common_grid_union"):
        x_grid, E_mat, cols_kept, bad_cols, exclusion_info = _build_energy_matrix_for_ext(
            ext,
            n_grid=n_grid,
            z_thresh=z_thresh,
            use_full_extension_range=use_union,
            exclude_obvious_outliers=exclude_obvious_outliers,
            use_all_beads=use_all_beads,
        )
        meta["exclusion_info"] = exclusion_info
        meta["n_grid"] = n_grid
        if E_mat.size == 0:
            return E_mat, [], x_grid if x_grid is not None else None, meta
        return E_mat, cols_kept, x_grid, meta

    if representation == "percentile_grid":
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, n_grid)), [], None, meta

        p_grid = np.linspace(0, 100, n_grid)
        E_rows, cols_kept = [], []

        for col in df.columns:
            E_at_p = build_energy_pct_curve(forces, df[col].values, p_grid)
            if E_at_p is None:
                continue

            E_rows.append(E_at_p)
            cols_kept.append(col)

        X = stack_feature_rows(E_rows, n_grid)
        return X, cols_kept, p_grid, meta

    if representation == "raw_extension_sorted_by_force":
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, len(forces))), [], None, meta

        sort_idx = np.argsort(forces)
        forces_sorted = forces[sort_idx]
        X_rows, cols_kept = [], []

        for col in df.columns:
            ext_sorted = get_sorted_ext_curve(df[col].values, sort_idx)
            if ext_sorted is None:
                continue

            X_rows.append(ext_sorted)
            cols_kept.append(col)

        X = stack_feature_rows(X_rows, len(forces))
        meta["n_features"] = len(forces)
        meta["force_grid"] = forces_sorted
        return X, cols_kept, forces_sorted, meta

    if representation == "raw_extension_sorted_by_force_normalised":
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, len(forces))), [], None, meta

        sort_idx = np.argsort(forces)
        forces_sorted = forces[sort_idx]
        X_rows, cols_kept = [], []

        for col in df.columns:
            ext_sorted = get_sorted_ext_curve(df[col].values, sort_idx)
            if ext_sorted is None:
                continue

            ext_sorted = normalise_if_possible(ext_sorted)
            X_rows.append(ext_sorted)
            cols_kept.append(col)

        X = stack_feature_rows(X_rows, len(forces))
        meta["n_features"] = len(forces)
        meta["force_grid"] = forces_sorted
        return X, cols_kept, forces_sorted, meta

    if representation == "raw_extension_sorted_by_force_subsampled":
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, n_grid)), [], None, meta

        sort_idx = np.argsort(forces)
        forces_sorted = forces[sort_idx]
        n_total = len(forces_sorted)
        if n_total <= n_grid:
            sample_idx = np.arange(n_total)
        else:
            sample_idx = np.linspace(0, n_total - 1, n_grid, dtype=int)
        forces_subsampled = forces_sorted[sample_idx]
        X_rows, cols_kept = [], []

        for col in df.columns:
            ext_sorted = get_sorted_ext_curve(df[col].values, sort_idx)
            if ext_sorted is None:
                continue

            ext_subsampled = ext_sorted[sample_idx]
            X_rows.append(ext_subsampled)
            cols_kept.append(col)

        X = stack_feature_rows(X_rows, len(sample_idx))
        meta["n_features"] = len(sample_idx)
        meta["n_grid"] = n_grid
        meta["force_grid"] = forces_subsampled
        return X, cols_kept, forces_subsampled, meta

    if representation == "force_percentile_grid":
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, n_grid)), [], None, meta

        p_grid = np.linspace(0, 100, n_grid)
        F_rows, cols_kept = [], []

        for col in df.columns:
            F_at_p = build_force_pct_curve(forces, df[col].values, p_grid)
            if F_at_p is None:
                continue

            F_rows.append(F_at_p)
            cols_kept.append(col)

        X = stack_feature_rows(F_rows, n_grid)
        meta["n_grid"] = n_grid
        return X, cols_kept, p_grid, meta

    if representation == "force_percentile_normalised":
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, n_grid)), [], None, meta

        p_grid = np.linspace(0, 100, n_grid)
        F_rows, cols_kept = [], []

        for col in df.columns:
            F_at_p = build_force_pct_curve(forces, df[col].values, p_grid)
            if F_at_p is None:
                continue

            F_at_p = normalise_if_possible(F_at_p)
            F_rows.append(F_at_p)
            cols_kept.append(col)

        X = stack_feature_rows(F_rows, n_grid)
        meta["n_grid"] = n_grid
        return X, cols_kept, p_grid, meta

    if representation == "percentile_grid_normalised":
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, n_grid)), [], None, meta

        p_grid = np.linspace(0, 100, n_grid)
        E_rows, cols_kept = [], []

        for col in df.columns:
            E_at_p = build_energy_pct_curve(forces, df[col].values, p_grid)
            if E_at_p is None:
                continue

            E_at_p = normalise_if_possible(E_at_p)
            E_rows.append(E_at_p)
            cols_kept.append(col)

        X = stack_feature_rows(E_rows, n_grid)
        meta["n_grid"] = n_grid
        return X, cols_kept, p_grid, meta

    if representation == "energy_slope_percentile":
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, n_grid)), [], None, meta

        p_grid = np.linspace(0, 100, n_grid)
        slope_rows, cols_kept = [], []

        for col in df.columns:
            E_at_p = build_energy_pct_curve(forces, df[col].values, p_grid)
            if E_at_p is None:
                continue

            slope = compute_slope(E_at_p, p_grid)
            slope_rows.append(slope)
            cols_kept.append(col)

        X = stack_feature_rows(slope_rows, n_grid)
        meta["n_grid"] = n_grid
        return X, cols_kept, p_grid, meta

    if representation == "force_slope_percentile":
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, n_grid)), [], None, meta

        p_grid = np.linspace(0, 100, n_grid)
        slope_rows, cols_kept = [], []

        for col in df.columns:
            F_at_p = build_force_pct_curve(forces, df[col].values, p_grid)
            if F_at_p is None:
                continue

            slope = compute_slope(F_at_p, p_grid)
            slope_rows.append(slope)
            cols_kept.append(col)

        X = stack_feature_rows(slope_rows, n_grid)
        meta["n_grid"] = n_grid
        return X, cols_kept, p_grid, meta

    if representation == "arc_length_percentile":
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, n_grid)), [], None, meta

        s_grid = np.linspace(0, 100, n_grid)
        E_rows, cols_kept = [], []

        for col in df.columns:
            x_bead, E_bead = build_energy_curve(forces, df[col].values)
            if x_bead is None:
                continue


            dx = np.diff(x_bead)
            dE = np.diff(E_bead)
            seg_len = np.sqrt(dx * dx + dE * dE)
            seg_len = np.where(np.isfinite(seg_len), seg_len, 0.0)

            arc_cum = np.zeros(x_bead.size)
            arc_cum[1:] = np.cumsum(seg_len)
            total = arc_cum[-1]
            if total <= 0:
                continue

            s_bead = 100.0 * arc_cum / total
            E_at_s = np.interp(s_grid, s_bead, E_bead, left=np.nan, right=np.nan)
            if np.all(np.isnan(E_at_s)):
                continue

            E_rows.append(E_at_s)
            cols_kept.append(col)

        X = stack_feature_rows(E_rows, n_grid)
        meta["n_grid"] = n_grid
        return X, cols_kept, s_grid, meta

    if representation == "landmark_quartiles":
        # rubbish method, can remove
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, 6)), [], None, meta

        features_list, cols_kept = [], []
        for col in df.columns:
            ext_col = df[col].values
            n = len(ext_col)
            if n < 2:
                continue

            x_bead, E_bead = build_energy_curve(forces[:n], ext_col)
            if x_bead is None:
                continue

            idx = np.argsort(ext_col)
            f_bead = np.asarray(forces[:n], dtype=float)[idx]
            x_f = np.asarray(ext_col, dtype=float)[idx]
            valid = np.isfinite(x_f) & np.isfinite(f_bead)
            if np.sum(valid) < 2:
                continue

            x_f, f_bead = x_f[valid], f_bead[valid]
            E_min, E_max = np.nanmin(E_bead), np.nanmax(E_bead)
            E_range = E_max - E_min
            if E_range <= 0:
                continue
            ext_at_25 = np.interp(E_min + 0.25 * E_range, E_bead, x_bead)
            ext_at_50 = np.interp(E_min + 0.50 * E_range, E_bead, x_bead)
            ext_at_75 = np.interp(E_min + 0.75 * E_range, E_bead, x_bead)

            x_min, x_max = x_f.min(), x_f.max()
            if x_max <= x_min:
                continue

            f_at_25 = np.interp(x_min + 0.25 * (x_max - x_min), x_f, f_bead)
            f_at_50 = np.interp(x_min + 0.50 * (x_max - x_min), x_f, f_bead)
            f_at_75 = np.interp(x_min + 0.75 * (x_max - x_min), x_f, f_bead)
            features_list.append([ext_at_25, ext_at_50, ext_at_75, f_at_25, f_at_50, f_at_75])
            cols_kept.append(col)

        X = stack_feature_rows(features_list, 6)
        meta["n_features"] = 6
        return X, cols_kept, None, meta

    if representation == "no_partition":
        # just dming the f-ext keeps more data
        forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
        if len(df) == 0:
            return np.empty((0, n_grid)), [], None, meta

        curves = []
        cols_kept = []

        for col in df.columns:
            x_bead, E_bead = build_energy_curve(forces, df[col].values)
            if x_bead is None:
                continue

            curves.append((x_bead, E_bead))
            cols_kept.append(col)

        if not curves:
            return np.empty((0, n_grid)), [], None, meta

        max_len = max(len(c[0]) for c in curves)
        target_len = min(max_len, n_grid)
        X = np.full((len(curves), target_len), np.nan, dtype=float)

        for i, (x_b, E_b) in enumerate(curves):
            X[i, :] = resample_curve(E_b, target_len)

        meta["max_curve_length"] = target_len
        return X, cols_kept, None, meta

    raise ValueError(f"unknown representation: {representation}")

def get_raw_force_ext_curves(ext, cols_kept, use_all_beads=True):
    energy_fn = energy_from_force_trapz
    forces, df = get_aligned_force_ext_inputs(ext, use_all_beads=use_all_beads)
    out = []
    for col in cols_kept:
        if col not in df.columns:
            continue
        ext_col = df[col].values
        x_bead, E_bead = energy_fn(
            forces[: len(ext_col)], ext_col, sort_input=True, remove_nan=True
        )
        if x_bead.size >= 2:
            out.append((np.asarray(x_bead), np.asarray(E_bead)))
        else:
            out.append((np.array([]), np.array([])))
    return out

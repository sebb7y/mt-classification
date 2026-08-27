import numpy as np

ROT_LABEL_NO_COIL = 0
ROT_LABEL_ONE_WAY = 1
ROT_LABEL_BOTH_WAY = 2

def gradient_activity_in_segment(z_segment, rot_segment):
    if rot_segment.size < 2 or z_segment.size < 2:
        return np.nan
    rot_segment = np.asarray(rot_segment).flatten()
    z_segment = np.asarray(z_segment).flatten()
    n = min(len(rot_segment), len(z_segment))
    rot_segment = rot_segment[:n]
    z_segment = z_segment[:n]
    if np.ptp(rot_segment) < 1e-12:
        return np.nan  
    grad = np.gradient(z_segment, rot_segment)
    return float(np.nanstd(grad))

def classify_rot_section(z, mag_rot, gradient_thresh=0.01, rot_zero_tol=1.0, min_points_per_half=5):
    n_beads = z.shape[1]
    labels = np.zeros(n_beads, dtype=int)
    if mag_rot.size == 0 or z.size == 0:
        return labels
    mag_rot = np.asarray(mag_rot).flatten()
    if len(mag_rot) != z.shape[0]:
        return labels

    pos_mask = mag_rot > rot_zero_tol
    neg_mask = mag_rot < -rot_zero_tol
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    for b in range(n_beads):
        zb = z[:, b]
        if n_pos >= min_points_per_half:
            pos_activity = gradient_activity_in_segment(zb[pos_mask], mag_rot[pos_mask])
        else:
            pos_activity = np.nan
        if n_neg >= min_points_per_half:
            neg_activity = gradient_activity_in_segment(zb[neg_mask], mag_rot[neg_mask])
        else:
            neg_activity = np.nan

        pos_active = not np.isnan(pos_activity) and pos_activity >= gradient_thresh
        neg_active = not np.isnan(neg_activity) and neg_activity >= gradient_thresh

        if pos_active and neg_active:
            labels[b] = ROT_LABEL_BOTH_WAY
        elif pos_active or neg_active:
            labels[b] = ROT_LABEL_ONE_WAY
        else:
            labels[b] = ROT_LABEL_NO_COIL
    return labels

def rot_extension_labels(ext, gradient_thresh=0.01, rot_zero_tol=1.0, min_points_per_half=5, use_all_beads=True):
    mag_rots = ext.get("mag_rots")
    if mag_rots is None or mag_rots.size == 0:
        n = 0
        if use_all_beads and ext.get("all_beads_df") is not None:
            n = ext["all_beads_df"].shape[1]
        elif ext.get("gb_df") is not None:
            n = ext["gb_df"].shape[1]
        return np.zeros(n, dtype=int), np.array([]).reshape(0, 0)

    if use_all_beads and ext.get("all_beads_df") is not None:
        z_df = ext["all_beads_df"]
    else:
        z_df = ext.get("gb_df")
    if z_df is None or len(z_df) == 0:
        return np.zeros(0, dtype=int), np.array([]).reshape(0, 0)

    z = np.asarray(z_df)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    mag_rots = np.asarray(mag_rots).flatten()
    n_time = min(len(mag_rots), z.shape[0])
    z = z[:n_time]
    mag_rots = mag_rots[:n_time]

    labels = classify_rot_section(
        z,
        mag_rots,
        gradient_thresh=gradient_thresh,
        rot_zero_tol=rot_zero_tol,
        min_points_per_half=min_points_per_half,
    )
    return labels, z

def build_rot_curve_matrix(z, mag_rot, n_grid=200, rot_min=None, rot_max=None):
    mag_rot = np.asarray(mag_rot).flatten()
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n_time = min(len(mag_rot), z.shape[0])
    mag_rot = mag_rot[:n_time]
    z = z[:n_time]
    if rot_min is None:
        rot_min = float(np.nanmin(mag_rot))
    if rot_max is None:
        rot_max = float(np.nanmax(mag_rot))
    if rot_max <= rot_min:
        rot_max = rot_min + 1e-9
    rot_grid = np.linspace(rot_min, rot_max, n_grid)
    n_beads = z.shape[1]
    Z_mat = np.full((n_grid, n_beads), np.nan, dtype=float)
    for b in range(n_beads):
        zb = z[:, b]
        valid = np.isfinite(zb) & np.isfinite(mag_rot)
        if np.sum(valid) < 2:
            continue
        Z_mat[:, b] = np.interp(rot_grid, mag_rot[valid], zb[valid])
    cols_kept = np.ones(n_beads, dtype=bool)
    for b in range(n_beads):
        col = Z_mat[:, b]
        if np.sum(np.isfinite(col)) < 2 or np.nanstd(col) < 1e-12:
            cols_kept[b] = False
    return rot_grid, Z_mat, cols_kept

def cluster_rot_extension(ext, n_grid=200, n_clusters=None, max_k=3, outlier_z=3.0, random_state=0, use_all_beads=True):
    from .clustering import cluster_energy_kmeans

    mag_rots = ext.get("mag_rots")
    if mag_rots is None or mag_rots.size == 0:
        return
    if use_all_beads and ext.get("all_beads_df") is not None:
        z_df = ext["all_beads_df"]
    else:
        z_df = ext.get("gb_df")
    if z_df is None or len(z_df) == 0:
        return
    z = np.asarray(z_df)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    mag_rots = np.asarray(mag_rots).flatten()
    n_time = min(len(mag_rots), z.shape[0])
    z = z[:n_time]
    mag_rots = mag_rots[:n_time]

    rot_grid, Z_mat, cols_kept = build_rot_curve_matrix(z, mag_rots, n_grid=n_grid)
    Z_for_cluster = Z_mat.T  # rows = beads, cols = rot
    indices_kept = np.where(cols_kept)[0]
    if len(indices_kept) < 2:
        ext["rot_cluster_rot_grid"] = rot_grid
        ext["rot_cluster_z_matrix"] = Z_mat
        ext["rot_cluster_cols_kept"] = cols_kept
        ext["rot_cluster_labels"] = np.full(z.shape[1], -1, dtype=int)
        return

    Z_sub = Z_for_cluster[indices_kept]
    Z_sub = np.nan_to_num(Z_sub, nan=0.0, posinf=0.0, neginf=0.0)
    labels_sub, info = cluster_energy_kmeans(
        Z_sub,
        n_clusters=n_clusters,
        max_k=max_k,
        random_state=random_state,
        outlier_z=outlier_z,
    )
    labels_full = np.full(z.shape[1], -1, dtype=int)
    labels_full[indices_kept] = labels_sub
    ext["rot_cluster_rot_grid"] = rot_grid
    ext["rot_cluster_z_matrix"] = Z_mat
    ext["rot_cluster_cols_kept"] = cols_kept
    ext["rot_cluster_labels"] = labels_full

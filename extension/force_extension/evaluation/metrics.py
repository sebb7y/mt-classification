import numpy as np

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

def y_true_y_pred_three_class(cluster_cols, cluster_labels):
    subset = []
    for i in range(len(cluster_cols)):
        c = cluster_cols[i]
        if not isinstance(c, str):
            continue
        if c.startswith("gc_") or c.startswith("uc_") or c.startswith("bb_"):
            subset.append((i, c))

    if not subset:
        return None

    indices = []
    sub_cols = []
    for s in subset:
        indices.append(s[0])
        sub_cols.append(s[1])

    y_true = np.zeros(len(sub_cols), dtype=int)
    for j in range(len(sub_cols)):
        c = sub_cols[j]
        if c.startswith("gc_"):
            y_true[j] = 1
        elif c.startswith("uc_"):
            y_true[j] = 0
        else:
            y_true[j] = 2

    labels = cluster_labels[indices]
    unique = np.unique(labels)
    non_outlier = []
    for k in unique:
        if k >= 0:
            non_outlier.append(k)
    non_outlier = sorted(non_outlier)
    n = len(y_true)
    best_acc = -1.0
    best_y_pred = None

    if len(non_outlier) == 0:
        pred = np.full(n, 2, dtype=int)
        acc = float((y_true == pred).mean())
        if acc >= 0:
            return y_true, pred, sub_cols
        return None

    def apply_mapping(mapping):
        return np.array([mapping.get(l, 2) for l in labels])

    if len(non_outlier) == 1:
        for cls in (0, 1):
            mapping = {non_outlier[0]: cls}
            mapping[-1] = 2
            pred = apply_mapping(mapping)
            acc = (y_true == pred).mean()
            if acc > best_acc:
                best_acc, best_y_pred = acc, pred
    elif len(non_outlier) == 2:
        for perm in ((0, 1), (1, 0)):
            mapping = {non_outlier[0]: perm[0], non_outlier[1]: perm[1], -1: 2}
            pred = apply_mapping(mapping)
            acc = (y_true == pred).mean()
            if acc > best_acc:
                best_acc, best_y_pred = acc, pred
    else:
        from itertools import permutations

        top3 = non_outlier[:3]
        for perm in permutations([0, 1, 2]):
            mapping = {}
            for j in range(3):
                mapping[top3[j]] = perm[j]
            mapping[-1] = 2
            for k in non_outlier[3:]:
                mapping[k] = 2
            pred = apply_mapping(mapping)
            acc = (y_true == pred).mean()
            if acc > best_acc:
                best_acc, best_y_pred = acc, pred

    if best_y_pred is None:
        return None
    return y_true, best_y_pred, sub_cols

def y_true_y_pred_from_clustering(cluster_cols, cluster_labels, ext=None, label_scheme='good_bad'):
    if not cluster_cols or cluster_labels is None or len(cluster_labels) != len(cluster_cols):
        return None
    cluster_labels = np.asarray(cluster_labels).ravel()
    if label_scheme == "coilable_uncoilable_bad":
        return y_true_y_pred_three_class(cluster_cols, cluster_labels)
    if label_scheme == "coilable_uncoilable":
        subset = []
        for i in range(len(cluster_cols)):
            c = cluster_cols[i]
            if not isinstance(c, str):
                continue
            if c.startswith("gc_") or c.startswith("uc_"):
                subset.append((i, c))
        if not subset:
            return None
        indices = []
        sub_cols = []
        for s in subset:
            indices.append(s[0])
            sub_cols.append(s[1])
        good_flags = []
        for c in sub_cols:
            good_flags.append(c.startswith("gc_"))
        if not good_flags or all(good_flags) or not any(good_flags):
            return None
        cluster_cols = sub_cols
        cluster_labels = cluster_labels[indices]
    else:
        good_flags = []
        all_str = True
        for c in cluster_cols:
            if not isinstance(c, str):
                all_str = False
                break
        if all_str:
            for c in cluster_cols:
                good_flags.append(c.startswith("gb_"))
        elif ext is not None:
            gb_df = ext.get("gb_df", pd.DataFrame())
            gb_cols = set()
            if hasattr(gb_df, "columns"):
                gb_cols = set(gb_df.columns)
            for c in cluster_cols:
                good_flags.append(c in gb_cols)
        else:
            return None

    y_true = np.zeros(len(good_flags), dtype=int)
    for j in range(len(good_flags)):
        if good_flags[j]:
            y_true[j] = 1
        else:
            y_true[j] = 0

    unique = []
    for k in np.unique(cluster_labels):
        if k >= 0:
            unique.append(k)
    n = len(cluster_cols)
    cols_aligned = list(cluster_cols)
    if not unique:
        y_pred = np.zeros(n, dtype=int)
        return y_true, y_pred, cols_aligned
    best_k = None
    best_acc = -1.0
    for k in unique:
        pred_good = set()
        for i in range(n):
            if cluster_labels[i] == k:
                pred_good.add(i)
        pred_bad = set(range(n)) - pred_good
        good_set = set()
        for i in range(len(good_flags)):
            if good_flags[i]:
                good_set.add(i)
        bad_set = set(range(n)) - good_set
        acc = (len(pred_good & good_set) + len(pred_bad & bad_set)) / n
        if acc > best_acc:
            best_acc, best_k = acc, k
    if best_k is None:
        return None
    y_pred = np.zeros(n, dtype=int)
    for i in range(n):
        if cluster_labels[i] == best_k:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_true, y_pred, cols_aligned

def pack_metrics(y_true, y_pred, label_scheme):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    if n < 2:
        return None
    is_three_class = label_scheme == "coilable_uncoilable_bad" or len(np.unique(y_true)) >= 3
    acc = float((y_true == y_pred).mean())
    n_coilable = int((y_true == 1).sum())
    n_uncoilable = int((y_true == 0).sum())
    if is_three_class:
        n_bad = int((y_true == 2).sum())
    else:
        n_bad = 0
    n_good = n_coilable
    if is_three_class:
        n_bad_legacy = n_uncoilable + n_bad
    else:
        n_bad_legacy = int((y_true == 0).sum())
    y_true_bin = (y_true == 1).astype(int)
    y_pred_bin = (y_pred == 1).astype(int)
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    chosen_cluster_size = int((y_pred_bin == 1).sum())
    out = {
        "accuracy": acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_good": n_good,
        "n_bad": n_bad_legacy,
        "chosen_cluster_size": chosen_cluster_size,
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }
    if is_three_class:
        out["n_coilable"] = n_coilable
        out["n_uncoilable"] = n_uncoilable
        out["n_bad_class"] = n_bad
    return out


def bead_df(ext):
    if ext is None:
        return None
    ab = ext.get("all_beads_df")
    if ab is not None and len(ab) > 0:
        return ab
    return ext.get("gb_df")


def label_cols(ext, scheme):
    df = bead_df(ext)
    if df is None or not hasattr(df, "columns"):
        return []
    out = []
    for c in df.columns:
        s = str(c)
        if scheme == "coilable_uncoilable_bad":
            if s.startswith("gc_") or s.startswith("uc_") or s.startswith("bb_"):
                out.append(c)
        elif scheme == "coilable_uncoilable":
            if s.startswith("gc_") or s.startswith("uc_"):
                out.append(c)
        elif s.startswith("gb_") or s.startswith("bb_"):
            out.append(c)
    return out


def truth_from_col(col, scheme):
    s = str(col)
    if scheme == "coilable_uncoilable_bad":
        if s.startswith("gc_"):
            return 1
        if s.startswith("uc_"):
            return 0
        return 2
    if scheme == "coilable_uncoilable":
        return 1 if s.startswith("gc_") else 0
    return 1 if s.startswith("gb_") else 0


def drop_pred_default(scheme):
    return 2 if scheme == "coilable_uncoilable_bad" else 0


def expand_all(ext, kept, y_t, y_p, scheme):
    cols = label_cols(ext, scheme)
    if len(cols) == 0:
        return None
    pm = {kept[i]: int(y_p[i]) for i in range(len(kept))}
    d = drop_pred_default(scheme)
    yt = np.array([truth_from_col(c, scheme) for c in cols], dtype=int)
    yp = np.array([pm.get(c, d) for c in cols], dtype=int)
    return yt, yp


def vote_row(names, pred_by_bead, bead):
    d = pred_by_bead.get(bead) or {}
    if len(d) != len(names) or not all(n in d for n in names):
        return None
    vals = [int(d[n]) for n in names]
    m = len(names)
    if all(v in (0, 1) for v in vals):
        return 1 if sum(vals) >= (m + 1) // 2 else 0
    b = np.bincount(vals, minlength=max(vals) + 1 if vals else 1)
    return int(np.argmax(b))


def ensemble_all(ext, names, pred_by_bead, scheme):
    cols = label_cols(ext, scheme)
    if not cols:
        return None
    d = drop_pred_default(scheme)
    yt = np.array([truth_from_col(c, scheme) for c in cols], dtype=int)
    yp = np.zeros(len(cols), dtype=int)
    for i, c in enumerate(cols):
        v = vote_row(names, pred_by_bead, c)
        yp[i] = v if v is not None else d
    return yt, yp


def cluster_metrics(cluster_cols, cluster_labels, E_mat=None, ext=None, label_scheme='good_bad'):
    pair = y_true_y_pred_from_clustering(cluster_cols, cluster_labels, ext=ext, label_scheme=label_scheme)
    if pair is None:
        return None
    y_true, y_pred, cols_aligned = pair
    n_sub = len(y_true)
    if n_sub < 2:
        return None
    sil = None
    if E_mat is not None and E_mat.shape[0] == n_sub and E_mat.size > 0:
        try:
            from sklearn.preprocessing import StandardScaler
            X = np.nan_to_num(E_mat, nan=0.0)
            X = StandardScaler().fit_transform(X)
            if len(np.unique(cluster_labels)) >= 2:
                sil = float(silhouette_score(X, cluster_labels))
            else:
                sil = float("nan")
        except Exception:
            sil = float("nan")
    expanded = expand_all(ext, cols_aligned, y_true, y_pred, label_scheme) if ext is not None else None
    if expanded is not None:
        y_true, y_pred = expanded
    if len(y_true) < 2:
        return None
    out = pack_metrics(y_true, y_pred, label_scheme)
    if out is None:
        return None
    out["silhouette"] = sil
    return out

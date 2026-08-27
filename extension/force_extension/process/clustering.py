import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean

from .. import config
from .energy import energy_mat_for_ext


MAD_TO_SIGMA = 1.4826 # consistency factor

try:
    from fastdtw import fastdtw
    _DTW_AVAILABLE = True
except ImportError:
    _DTW_AVAILABLE = False

def apply_outlier_z(labels, centre_dists, outlier_z):
    if outlier_z is None or len(labels) == 0:
        return labels.copy(), np.zeros(len(labels), dtype=bool)
    labels_out = labels.copy()
    median_dist = np.median(centre_dists)
    mad = np.median(np.abs(centre_dists - median_dist))
    if mad == 0:
        return labels_out, np.zeros(len(labels), dtype=bool)
    threshold = median_dist + outlier_z * MAD_TO_SIGMA * mad
    outlier_mask = centre_dists > threshold
    labels_out[outlier_mask] = -1
    return labels_out, outlier_mask

def relabel_small_clusters_as_outliers(labels, min_cluster_size=3):
    labels = np.asarray(labels).copy()
    mask_valid = labels >= 0
    unique, counts = np.unique(labels[mask_valid], return_counts=True)
    small_clusters = unique[counts < min_cluster_size]
    mask_small = np.isin(labels, small_clusters)
    labels[mask_small] = -1
    return labels, mask_small

def replace_nan_for_clustering(X):
    X = np.asarray(X, dtype=float)
    if not np.any(np.isnan(X)):
        return X
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    out = X.copy()
    nan_mask = np.isnan(out)
    out[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return out

def cluster_energy_kmeans(E_mat, n_clusters=None, max_k=2, random_state=0, outlier_z=None, min_cluster_size=3):
    E_mat = np.asarray(E_mat)
    n_samples = E_mat.shape[0]
    if n_samples == 0:
        return np.array([]), {"n_clusters_": 0, "method": "kmeans", "distances": np.array([])}

    X = replace_nan_for_clustering(E_mat)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    global_mean = X.mean(axis=0)
    dist_global = np.linalg.norm(X - global_mean, axis=1)

    if n_clusters is None:
        if n_samples < 3:
            n_clusters = min(2, n_samples)
        else:
            best_k, best_score = 1, None
            for k in range(2, min(max_k, n_samples) + 1):
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels_k = km.fit_predict(X)
                if len(np.unique(labels_k)) < 2:
                    continue
                score = silhouette_score(X, labels_k)
                if best_score is None or score > best_score:
                    best_score, best_k = score, k
            n_clusters = best_k

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels_raw = km.fit_predict(X)
    centres = km.cluster_centers_
    dist_centre = np.linalg.norm(X - centres[labels_raw], axis=1)
    labels_out, _ = apply_outlier_z(labels_raw, dist_centre, outlier_z)
    labels_out, _ = relabel_small_clusters_as_outliers(labels_out, min_cluster_size)

    info = {
        "method": "kmeans",
        "n_clusters_": len(set(labels_out) - {-1}),
        "centres": centres,
        "distances": dist_centre,
        "distances_global": dist_global,
        "raw_labels": labels_raw,
        "labels": labels_out
    }
    return labels_out, info

def cluster_energy_gmm(E_mat, n_clusters=None, max_k=2, random_state=0, outlier_z=None):
    E_mat = np.asarray(E_mat)
    n_samples = E_mat.shape[0]
    if n_samples == 0:
        return np.array([]), {"n_clusters_": 0, "method": "gmm", "distances": np.array([])}

    X = replace_nan_for_clustering(E_mat)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if n_clusters is None:
        best_k, best_bic = 1, None
        for k in range(1, min(max_k, n_samples) + 1):
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
            gmm.fit(X)
            bic = gmm.bic(X)
            if best_bic is None or bic < best_bic:
                best_bic, best_k = bic, k
        n_clusters = best_k

    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_state)
    labels = gmm.fit_predict(X)
    centres = gmm.means_
    probs = gmm.predict_proba(X)
    dists = np.linalg.norm(X - centres[labels], axis=1)
    labels_out, _ = apply_outlier_z(labels, dists, outlier_z)
    info = {
        "method": "gmm",
        "n_clusters_": len(set(labels_out) - {-1}),
        "centres": centres,
        "probs": probs,
        "distances": dists,
        "raw_labels": labels,
        "labels": labels_out
    }
    return labels_out, info

def cluster_energy_dtw(E_mat, n_clusters=None, max_k=2, outlier_z=None):
    if not _DTW_AVAILABLE:
        raise ImportError("fastdtw required for DTW clustering: pip install fastdtw")
    E_mat = np.asarray(E_mat)
    n_samples = E_mat.shape[0]
    if n_samples == 0:
        return np.array([]), {"n_clusters_": 0, "method": "dtw", "distances": np.array([])}

    X = replace_nan_for_clustering(E_mat)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist, _ = fastdtw(X[i], X[j], dist=euclidean)
            D[i, j] = D[j, i] = dist

    if n_clusters is None:
        best_k, best_score = 1, None
        for k in range(2, min(max_k, n_samples) + 1):
            ac = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
            labels_k = ac.fit_predict(D)
            if len(np.unique(labels_k)) < 2:
                continue
            score = silhouette_score(D, labels_k, metric="precomputed")
            if best_score is None or score > best_score:
                best_score, best_k = score, k
        n_clusters = best_k

    ac = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    labels = ac.fit_predict(D)
    centre_dists = np.zeros(n_samples)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        submatrix = D[np.ix_(idx, idx)]
        mean_dists = submatrix.mean(axis=1)
        medoid_local = idx[np.argmin(mean_dists)]
        centre_dists[idx] = D[idx, medoid_local]
    labels_out, _ = apply_outlier_z(labels, centre_dists, outlier_z)
    info = {
        "method": "dtw",
        "n_clusters_": len(set(labels_out) - {-1}),
        "distance_matrix": D,
        "distances": centre_dists,
        "raw_labels": labels,
        "labels": labels_out
    }
    return labels_out, info

def cluster_energy_agglomerative(E_mat, n_clusters=None, max_k=2, outlier_z=None, min_cluster_size=3, linkage='ward'):
    E_mat = np.asarray(E_mat)
    n_samples = E_mat.shape[0]
    if n_samples == 0:
        return np.array([]), {"n_clusters_": 0, "method": "agglomerative", "distances": np.array([])}

    X = replace_nan_for_clustering(E_mat)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if n_clusters is None:
        best_k, best_score = 1, None
        for k in range(2, min(max_k, n_samples) + 1):
            ac = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels_k = ac.fit_predict(X)
            if len(np.unique(labels_k)) < 2:
                continue
            score = silhouette_score(X, labels_k)
            if best_score is None or score > best_score:
                best_score, best_k = score, k
        n_clusters = best_k

    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels_raw = ac.fit_predict(X)
    centres = np.zeros((n_clusters, X.shape[1]))
    for c in range(n_clusters):
        mask = labels_raw == c
        if np.any(mask):
            centres[c] = X[mask].mean(axis=0)
    dist_centre = np.linalg.norm(X - centres[labels_raw], axis=1)
    labels_out, _ = apply_outlier_z(labels_raw, dist_centre, outlier_z)
    labels_out, _ = relabel_small_clusters_as_outliers(labels_out, min_cluster_size)

    info = {
        "method": "agglomerative",
        "n_clusters_": len(set(labels_out) - {-1}),
        "centres": centres,
        "distances": dist_centre,
        "raw_labels": labels_raw,
        "labels": labels_out
    }
    return labels_out, info

def cluster_energy_spectral(E_mat, n_clusters=None, max_k=2, random_state=0, outlier_z=None, min_cluster_size=3, affinity='rbf'):
    E_mat = np.asarray(E_mat)
    n_samples = E_mat.shape[0]
    if n_samples == 0:
        return np.array([]), {"n_clusters_": 0, "method": "spectral", "distances": np.array([])}

    X = replace_nan_for_clustering(E_mat)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if n_clusters is None:
        best_k, best_score = 1, None
        for k in range(2, min(max_k, n_samples) + 1):
            sc = SpectralClustering(n_clusters=k, affinity=affinity, random_state=random_state)
            labels_k = sc.fit_predict(X)
            if len(np.unique(labels_k)) < 2:
                continue
            score = silhouette_score(X, labels_k)
            if best_score is None or score > best_score:
                best_score, best_k = score, k
        n_clusters = best_k

    sc = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=random_state)
    labels_raw = sc.fit_predict(X)
    centres = np.zeros((n_clusters, X.shape[1]))
    for c in range(n_clusters):
        mask = labels_raw == c
        if np.any(mask):
            centres[c] = X[mask].mean(axis=0)
    dist_centre = np.linalg.norm(X - centres[labels_raw], axis=1)
    labels_out, _ = apply_outlier_z(labels_raw, dist_centre, outlier_z)
    labels_out, _ = relabel_small_clusters_as_outliers(labels_out, min_cluster_size)

    info = {
        "method": "spectral",
        "n_clusters_": len(set(labels_out) - {-1}),
        "centres": centres,
        "distances": dist_centre,
        "raw_labels": labels_raw,
        "labels": labels_out
    }
    return labels_out, info

def cluster_energy_curves(E_mat, method='kmeans', n_clusters=None, max_k=2, random_state=0, outlier_z=None):
    if method == "kmeans":
        return cluster_energy_kmeans(
            E_mat, n_clusters=n_clusters, max_k=max_k,
            random_state=random_state, outlier_z=outlier_z,
        )
    if method == "gmm":
        return cluster_energy_gmm(
            E_mat, n_clusters=n_clusters, max_k=max_k,
            random_state=random_state, outlier_z=outlier_z,
        )
    if method == "dtw":
        return cluster_energy_dtw(
            E_mat, n_clusters=n_clusters, max_k=max_k, outlier_z=outlier_z,
        )
    if method == "agglomerative":
        return cluster_energy_agglomerative(
            E_mat, n_clusters=n_clusters, max_k=max_k, outlier_z=outlier_z,
        )
    if method == "spectral":
        return cluster_energy_spectral(
            E_mat, n_clusters=n_clusters, max_k=max_k,
            random_state=random_state, outlier_z=outlier_z,
        )
    raise ValueError(f"unknown method: {method}")

def cluster_ext(ext, method='kmeans', n_clusters=None, max_k=2, n_grid=None, z_thresh=None, outlier_z=None, random_state=0):
    if n_grid is None:
        n_grid = config.DEFAULT_N_GRID
    if z_thresh is None:
        z_thresh = config.DEFAULT_Z_THRESH_RANGE
    if outlier_z is None:
        outlier_z = config.DEFAULT_OUTLIER_Z

    x_grid, E_mat, cols_kept, bad_cols, exclusion_info = energy_mat_for_ext(
        ext, n_grid=n_grid, z_thresh=z_thresh, use_full_extension_range=False,
    )
    if E_mat.size == 0:
        ext["cluster_x_grid"] = x_grid
        ext["cluster_energy_matrix"] = E_mat
        ext["cluster_cols"] = cols_kept
        ext["cluster_bad_cols"] = bad_cols
        ext["cluster_exclusion_info"] = exclusion_info
        ext["cluster_summary"] = {
            "n_excluded_by_range": len(exclusion_info.get("by_range", [])),
            "n_excluded_by_energy": len(exclusion_info.get("by_energy", [])),
            "n_clustered": 0,
            "n_outliers": 0,
            "n_clusters": 0
        }
        ext["cluster_labels"] = np.array([])
        ext["cluster_info"] = {"distances": np.array([])}
        return ext

    labels, info = cluster_energy_curves(
        E_mat, method=method, n_clusters=n_clusters, max_k=max_k,
        random_state=random_state, outlier_z=outlier_z,
    )
    labels = np.asarray(labels).ravel()
    n_outliers = int((labels < 0).sum())
    n_clusters = len(set(labels) - {-1})

    ext["cluster_x_grid"] = x_grid
    ext["cluster_energy_matrix"] = E_mat
    ext["cluster_cols"] = cols_kept
    ext["cluster_bad_cols"] = bad_cols
    ext["cluster_exclusion_info"] = exclusion_info
    ext["cluster_summary"] = {
        "n_excluded_by_range": len(exclusion_info.get("by_range", [])),
        "n_excluded_by_energy": len(exclusion_info.get("by_energy", [])),
        "n_clustered": int(len(labels)),
        "n_outliers": n_outliers,
        "n_clusters": n_clusters
    }
    ext["cluster_labels"] = labels
    ext["cluster_info"] = info
    return ext

import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture

from .config import CLUSTER_KMEANS, CLUSTER_GMM, CLUSTER_HIERARCHICAL, CLUSTER_DBSCAN, CLUSTER_HDBSCAN, CLUSTER_OPTICS, CLUSTER_SPECTRAL, PipelineCfg


def replace_nan_for_clustering(X):
    X = np.asarray(X, dtype=float)
    if not np.any(~np.isfinite(X)):
        return X

    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)

    out = X.copy()
    out[~np.isfinite(out)] = np.take(col_mean, np.where(~np.isfinite(out))[1])
    return out


def apply_outlier_z(labels, centre_dists, outlier_z):
    if outlier_z is None or len(labels) == 0:
        return labels.copy(), np.zeros(len(labels), dtype=bool)

    labels_out = labels.copy()
    median_dist = np.median(centre_dists)
    mad = np.median(np.abs(centre_dists - median_dist))
    if mad == 0:
        return labels_out, np.zeros(len(labels), dtype=bool)

    threshold = median_dist + outlier_z * 1.4826 * mad
    outlier_mask = centre_dists > threshold
    labels_out[outlier_mask] = -1
    return labels_out, outlier_mask


def relabel_small_clusters(labels, min_cluster_size):
    labels = np.asarray(labels).copy()
    valid = labels >= 0
    unique, counts = np.unique(labels[valid], return_counts=True)
    small = unique[counts < min_cluster_size]
    labels[np.isin(labels, small)] = -1
    return labels


def centres_from_labels(X, labels, n_clusters):
    centres = np.zeros((n_clusters, X.shape[1]))
    for c in range(n_clusters):
        mask = labels == c
        if np.any(mask):
            centres[c] = X[mask].mean(axis=0)

    return centres


def cluster(X, config):
    X = replace_nan_for_clustering(X)
    n_samples = X.shape[0]
    if n_samples == 0:
        return np.array([]), {"method": config.clustering, "n_clusters_": 0, "distances": np.array([])}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    method = config.clustering
    n_clusters = config.n_clusters
    max_k = config.max_k
    outlier_z = config.outlier_z
    min_cluster_size = config.min_cluster_size
    random_state = config.random_state

    if method == CLUSTER_KMEANS:
        n_init = getattr(config, "extra", {}).get("n_init", 10)
        if n_clusters is None:
            from sklearn.metrics import silhouette_score
            best_k, best_score = 1, None
            for k in range(2, min(max_k, n_samples) + 1):
                km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
                lab = km.fit_predict(X_scaled)
                if len(np.unique(lab)) < 2:
                    continue
                sc = silhouette_score(X_scaled, lab)
                if best_score is None or sc > best_score:
                    best_score, best_k = sc, k
            n_clusters = best_k

        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        labels_raw = km.fit_predict(X_scaled)
        centres = km.cluster_centers_
        dist_centre = np.linalg.norm(X_scaled - centres[labels_raw], axis=1)
        labels_out, _ = apply_outlier_z(labels_raw, dist_centre, outlier_z)
        labels_out = relabel_small_clusters(labels_out, min_cluster_size)

        return labels_out, {
            "method": "kmeans",
            "n_clusters_": len(set(labels_out) - {-1}),
            "centres": centres,
            "distances": dist_centre,
            "raw_labels": labels_raw
        }

    # todo deduplicatae this copy pasted spaghetti
    if method == CLUSTER_GMM:
        if n_clusters is None:
            best_k, best_bic = 1, None
            for k in range(1, min(max_k, n_samples) + 1):
                gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
                gmm.fit(X_scaled)
                bic = gmm.bic(X_scaled)
                if best_bic is None or bic < best_bic:
                    best_bic, best_k = bic, k
            n_clusters = best_k

        gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_state)
        labels_raw = gmm.fit_predict(X_scaled)
        centres = gmm.means_
        dist_centre = np.linalg.norm(X_scaled - centres[labels_raw], axis=1)
        labels_out, _ = apply_outlier_z(labels_raw, dist_centre, outlier_z)

        return labels_out, {
            "method": "gmm",
            "n_clusters_": len(set(labels_out) - {-1}),
            "centres": centres,
            "distances": dist_centre,
            "raw_labels": labels_raw
        }

    if method == CLUSTER_HIERARCHICAL:
        linkage = getattr(config, "hierarchical_linkage", "ward") or "ward"
        from sklearn.metrics import silhouette_score
        if n_clusters is None:
            best_k, best_score = 1, None
            for k in range(2, min(max_k, n_samples) + 1):
                ac = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                lab = ac.fit_predict(X_scaled)
                if len(np.unique(lab)) < 2:
                    continue
                sc = silhouette_score(X_scaled, lab)
                if best_score is None or sc > best_score:
                    best_score, best_k = sc, k
            n_clusters = best_k

        ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels_raw = ac.fit_predict(X_scaled)
        centres = centres_from_labels(X_scaled, labels_raw, n_clusters)
        dist_centre = np.linalg.norm(X_scaled - centres[labels_raw], axis=1)
        labels_out, _ = apply_outlier_z(labels_raw, dist_centre, outlier_z)
        labels_out = relabel_small_clusters(labels_out, min_cluster_size)

        return labels_out, {
            "method": "agglomerative",
            "n_clusters_": len(set(labels_out) - {-1}),
            "centres": centres,
            "distances": dist_centre,
            "raw_labels": labels_raw
        }

    if method == CLUSTER_DBSCAN:
        eps = getattr(config, "dbscan_eps", None)
        min_samples = getattr(config, "dbscan_min_samples", 5)
        if eps is None:
            from sklearn.neighbors import NearestNeighbors
            k = min(min_samples + 1, n_samples)
            nn = NearestNeighbors(n_neighbors=k).fit(X_scaled)
            dists, _ = nn.kneighbors(X_scaled)
            k_dist = dists[:, -1]
            eps = float(np.percentile(k_dist, 90))

        db = DBSCAN(eps=eps, min_samples=min_samples, metric=config.distance_metric)
        labels_out = db.fit_predict(X_scaled)
        labels_out[labels_out == -1] = -1
        n_clusters = len(set(labels_out) - {-1})

        if n_clusters > 0:
            centres = centres_from_labels(X_scaled, labels_out, n_clusters)
            dist_centre = np.linalg.norm(X_scaled - centres[labels_out], axis=1)
            dist_centre[labels_out < 0] = np.nan
        else:
            dist_centre = np.full(n_samples, np.nan)

        return labels_out, {
            "method": "dbscan",
            "n_clusters_": n_clusters,
            "eps": eps,
            "distances": dist_centre,
            "raw_labels": labels_out.copy()
        }

    if method == CLUSTER_HDBSCAN:
        min_cluster_size = getattr(config, "dbscan_min_samples", 5)
        min_cluster_size = max(2, min(min_cluster_size, n_samples))

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=max(1, min_cluster_size - 1), metric=config.distance_metric)
        labels_out = clusterer.fit_predict(X_scaled)
        labels_out = np.asarray(labels_out, dtype=int)
        labels_out[labels_out == -1] = -1
        n_clusters = len(set(labels_out) - {-1})

        if n_clusters > 0:
            centres = centres_from_labels(X_scaled, labels_out, n_clusters)
            dist_centre = np.linalg.norm(X_scaled - centres[labels_out], axis=1)
            dist_centre[labels_out < 0] = np.nan
        else:
            dist_centre = np.full(n_samples, np.nan)

        return labels_out, {
            "method": "hdbscan",
            "n_clusters_": n_clusters,
            "distances": dist_centre,
            "raw_labels": labels_out.copy()
        }

    if method == CLUSTER_OPTICS:
        from sklearn.cluster import OPTICS
        min_samples = getattr(config, "dbscan_min_samples", 5)

        optics = OPTICS(min_samples=min_samples, metric=config.distance_metric, cluster_method="xi")
        labels_out = optics.fit_predict(X_scaled)
        labels_out = np.asarray(labels_out, dtype=int)
        labels_out[labels_out == -1] = -1
        n_clusters = len(set(labels_out) - {-1})

        if n_clusters > 0:
            centres = centres_from_labels(X_scaled, labels_out, n_clusters)
            dist_centre = np.linalg.norm(X_scaled - centres[labels_out], axis=1)
            dist_centre[labels_out < 0] = np.nan
        else:
            dist_centre = np.full(n_samples, np.nan)

        return labels_out, {
            "method": "optics",
            "n_clusters_": n_clusters,
            "distances": dist_centre,
            "raw_labels": labels_out.copy()
        }

    if method == CLUSTER_SPECTRAL:
        from sklearn.metrics import silhouette_score
        affinity = "rbf"
        if n_clusters is None:
            best_k, best_score = 1, None
            for k in range(2, min(max_k, n_samples) + 1):
                sc = SpectralClustering(n_clusters=k, affinity=affinity, random_state=random_state)
                lab = sc.fit_predict(X_scaled)
                if len(np.unique(lab)) < 2:
                    continue
                score = silhouette_score(X_scaled, lab)
                if best_score is None or score > best_score:
                    best_score, best_k = score, k
            n_clusters = best_k

        sp = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=random_state)
        labels_raw = sp.fit_predict(X_scaled)
        centres = centres_from_labels(X_scaled, labels_raw, n_clusters)
        dist_centre = np.linalg.norm(X_scaled - centres[labels_raw], axis=1)
        labels_out, _ = apply_outlier_z(labels_raw, dist_centre, outlier_z)
        labels_out = relabel_small_clusters(labels_out, min_cluster_size)

        return labels_out, {
            "method": "spectral",
            "n_clusters_": len(set(labels_out) - {-1}),
            "centres": centres,
            "distances": dist_centre,
            "raw_labels": labels_raw
        }

    raise ValueError(f"unknown clustering: {method}")

import numpy as np

from .config import DIM_RED_NONE, DIM_RED_FPCA, DIM_RED_MANIFOLD_UMAP, DIM_RED_MANIFOLD_TSNE, DIM_RED_MANIFOLD_ISOMAP, DIM_RED_MANIFOLD_LLE, DIM_RED_MANIFOLD_LAPLACIAN, DIM_RED_DTW_MDS

def replace_nan(X, fill=0.0):
    out = np.asarray(X, dtype=float).copy()
    out[~np.isfinite(out)] = fill
    return out

def require_fastdtw():
    try:
        from fastdtw import fastdtw
    except ImportError as exc:
        raise ImportError("fastdtw required for dtw_mds") from exc
    return fastdtw

def require_umap():
    try:
        import umap
    except ImportError as exc:
        raise ImportError(f"umap failed: {exc} install: pip install umap-learn") from exc
    return umap

def reduce_dimensions(X, method=DIM_RED_NONE, n_components=10, random_state=0, **kwargs):
    if method == DIM_RED_DTW_MDS:
        fastdtw = require_fastdtw()
        from scipy.spatial.distance import euclidean
        from sklearn.manifold import MDS

        def euclidean_1d(u, v):
            u = np.atleast_1d(np.asarray(u, dtype=float).ravel())
            v = np.atleast_1d(np.asarray(v, dtype=float).ravel())
            return float(euclidean(u, v))

        n_samples = X.shape[0]
        if n_samples < 2:
            return replace_nan(X), None
        curves = []
        for i in range(n_samples):
            row = np.asarray(X[i], dtype=float).ravel()
            valid = np.isfinite(row)
            if np.any(valid):
                last = np.where(valid)[0][-1] + 1
                curves.append(row[:last].ravel())
            else:
                curves.append(np.array([0.0]))
        D = np.zeros((n_samples, n_samples))
        use_dtw = True
        try:
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    ci, cj = np.asarray(curves[i]).ravel(), np.asarray(curves[j]).ravel()
                    d, _ = fastdtw(ci, cj, dist=euclidean_1d)
                    D[i, j] = D[j, i] = d
        except Exception:
            use_dtw = False
            X_clean = replace_nan(X)
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    d = float(euclidean(X_clean[i], X_clean[j]))
                    D[i, j] = D[j, i] = d
        n = min(n_components or 10, n_samples - 1, n_samples)
        if n < 1:
            return replace_nan(X), None
        mds = MDS(n_components=n, dissimilarity="precomputed", random_state=random_state, **kwargs)
        Xr = mds.fit_transform(D)
        return Xr, mds

    X = replace_nan(X)
    if method == DIM_RED_NONE or method is None or method == "":
        return X, None

    if method == DIM_RED_FPCA:
        from sklearn.decomposition import PCA
        n = min(n_components or 10, X.shape[0], X.shape[1])
        if n < 1:
            return X, None
        pca = PCA(n_components=n, random_state=random_state)
        Xr = pca.fit_transform(X)
        return Xr, pca

    if method == DIM_RED_MANIFOLD_UMAP:
        umap = require_umap()
        n = min(n_components or 10, X.shape[0] - 1, X.shape[1])
        if n < 1:
            return X, None
        reducer = umap.UMAP(
            n_components=n,
            random_state=random_state,
            n_neighbors=min(15, X.shape[0] - 1) if X.shape[0] > 1 else 2,
            **kwargs,
        )
        Xr = reducer.fit_transform(X)
        return Xr, reducer

    if method == DIM_RED_MANIFOLD_TSNE:
        from sklearn.manifold import TSNE
        n = min(n_components or 2, 3, X.shape[0] - 1, X.shape[1])
        if n < 1:
            return X, None
        tsne = TSNE(
            n_components=n,
            random_state=random_state,
            perplexity=min(30, (X.shape[0] - 1) // 3) if X.shape[0] > 3 else 5, # here be dragons
            **kwargs,
        )
        Xr = tsne.fit_transform(X)
        return Xr, tsne

    if method == DIM_RED_MANIFOLD_ISOMAP:
        from sklearn.manifold import Isomap
        n = min(n_components or 10, X.shape[0] - 1, X.shape[1])
        if n < 1:
            return X, None
        n_neighbors = min(15, X.shape[0] - 1) if X.shape[0] > 1 else 2
        iso = Isomap(n_components=n, n_neighbors=n_neighbors, **kwargs)
        Xr = iso.fit_transform(X)
        return Xr, iso

    if method == DIM_RED_MANIFOLD_LLE:
        from sklearn.manifold import LocallyLinearEmbedding
        n = min(n_components or 10, X.shape[0] - 1, X.shape[1])
        if n < 1:
            return X, None
        n_neighbors = min(15, X.shape[0] - 1) if X.shape[0] > 1 else 2
        lle = LocallyLinearEmbedding(n_components=n, n_neighbors=n_neighbors, random_state=random_state, **kwargs)
        Xr = lle.fit_transform(X)
        return Xr, lle

    if method == DIM_RED_MANIFOLD_LAPLACIAN:
        from sklearn.manifold import SpectralEmbedding
        n = min(n_components or 10, X.shape[0] - 1, X.shape[1])
        if n < 1:
            return X, None
        n_neighbors = min(15, X.shape[0] - 1) if X.shape[0] > 1 else 2
        lap = SpectralEmbedding(n_components=n, n_neighbors=n_neighbors, random_state=random_state, **kwargs)
        Xr = lap.fit_transform(X)
        return Xr, lap

    raise ValueError(f"unknown dim_reduction: {method}")

import json
import os

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")
FORCE_PIPELINES_JSON = os.path.join(CONFIGS_DIR, "force_pipelines.json")

REPRESENTATION_COMMON_GRID_INTERSECTION = "common_grid_intersection"
REPRESENTATION_COMMON_GRID_UNION = "common_grid_union"
REPRESENTATION_PERCENTILE_GRID = "percentile_grid"
REPRESENTATION_RAW_EXTENSION_SORTED_BY_FORCE = "raw_extension_sorted_by_force"
REPRESENTATION_RAW_EXTENSION_SORTED_BY_FORCE_NORMALISED = "raw_extension_sorted_by_force_normalised"
REPRESENTATION_RAW_EXTENSION_SORTED_BY_FORCE_SUBSAMPLED = "raw_extension_sorted_by_force_subsampled"
REPRESENTATION_FORCE_PERCENTILE_GRID = "force_percentile_grid"
REPRESENTATION_FORCE_PERCENTILE_NORMALISED = "force_percentile_normalised"
REPRESENTATION_PERCENTILE_GRID_NORMALISED = "percentile_grid_normalised"
REPRESENTATION_ENERGY_SLOPE_PERCENTILE = "energy_slope_percentile"
REPRESENTATION_FORCE_SLOPE_PERCENTILE = "force_slope_percentile"
REPRESENTATION_ARC_LENGTH_PERCENTILE = "arc_length_percentile"
REPRESENTATION_LANDMARK_QUARTILES = "landmark_quartiles"
REPRESENTATION_NO_PARTITION = "no_partition"
REPRESENTATION_CHOICES = [
    REPRESENTATION_COMMON_GRID_INTERSECTION,
    REPRESENTATION_COMMON_GRID_UNION,
    REPRESENTATION_PERCENTILE_GRID,
    REPRESENTATION_RAW_EXTENSION_SORTED_BY_FORCE,
    REPRESENTATION_RAW_EXTENSION_SORTED_BY_FORCE_NORMALISED,
    REPRESENTATION_RAW_EXTENSION_SORTED_BY_FORCE_SUBSAMPLED,
    REPRESENTATION_FORCE_PERCENTILE_GRID,
    REPRESENTATION_FORCE_PERCENTILE_NORMALISED,
    REPRESENTATION_PERCENTILE_GRID_NORMALISED,
    REPRESENTATION_ENERGY_SLOPE_PERCENTILE,
    REPRESENTATION_FORCE_SLOPE_PERCENTILE,
    REPRESENTATION_ARC_LENGTH_PERCENTILE,
    REPRESENTATION_LANDMARK_QUARTILES,
    REPRESENTATION_NO_PARTITION,
]

DIM_RED_NONE = "none"
DIM_RED_FPCA = "fpca"
DIM_RED_MANIFOLD_UMAP = "umap"
DIM_RED_MANIFOLD_TSNE = "tsne"
DIM_RED_MANIFOLD_ISOMAP = "isomap"
DIM_RED_MANIFOLD_LLE = "lle"
DIM_RED_MANIFOLD_LAPLACIAN = "laplacian"
DIM_RED_DTW_MDS = "dtw_mds"
DIM_RED_CHOICES = [
    DIM_RED_NONE, DIM_RED_FPCA,
    DIM_RED_MANIFOLD_UMAP, DIM_RED_MANIFOLD_TSNE, DIM_RED_MANIFOLD_ISOMAP,
    DIM_RED_MANIFOLD_LLE, DIM_RED_MANIFOLD_LAPLACIAN,
    DIM_RED_DTW_MDS,
]

CLUSTER_KMEANS = "kmeans"
CLUSTER_GMM = "gmm"
CLUSTER_HIERARCHICAL = "hierarchical"
CLUSTER_DBSCAN = "dbscan"
CLUSTER_HDBSCAN = "hdbscan"
CLUSTER_OPTICS = "optics"
CLUSTER_SPECTRAL = "spectral"
CLUSTER_CHOICES = [CLUSTER_KMEANS, CLUSTER_GMM, CLUSTER_HIERARCHICAL, CLUSTER_DBSCAN, CLUSTER_HDBSCAN, CLUSTER_OPTICS, CLUSTER_SPECTRAL]

class PipelineCfg:
    def __init__(self, representation=REPRESENTATION_COMMON_GRID_INTERSECTION,
            dim_reduction=DIM_RED_NONE, dim_reduction_n_components=10,
            clustering=CLUSTER_KMEANS, n_clusters=None, max_k=2, n_grid=200,
            z_thresh=3.0, use_full_extension_range=False, outlier_z=3.0,
            min_cluster_size=3, distance_metric="euclidean", dbscan_eps=None,
            dbscan_min_samples=5, hierarchical_linkage="ward", random_state=0,
            rotation_max_k=3, ensemble_pipelines=None, ensemble_method="majority_vote",
            extra=None):
        self.representation = representation
        self.dim_reduction = dim_reduction
        self.dim_reduction_n_components = dim_reduction_n_components
        self.clustering = clustering
        self.n_clusters = n_clusters
        self.max_k = max_k
        self.n_grid = n_grid
        self.z_thresh = z_thresh
        self.use_full_extension_range = use_full_extension_range
        self.outlier_z = outlier_z
        self.min_cluster_size = min_cluster_size
        self.distance_metric = distance_metric
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.hierarchical_linkage = hierarchical_linkage
        self.random_state = random_state
        self.rotation_max_k = rotation_max_k
        self.ensemble_pipelines = ensemble_pipelines
        self.ensemble_method = ensemble_method
        self.extra = extra if extra is not None else {}

    def to_dict(self):
        out = {
            "representation": self.representation,
            "dim_reduction": self.dim_reduction,
            "dim_reduction_n_components": self.dim_reduction_n_components,
            "clustering": self.clustering,
            "n_clusters": self.n_clusters,
            "max_k": self.max_k,
            "n_grid": self.n_grid,
            "z_thresh": self.z_thresh,
            "use_full_extension_range": self.use_full_extension_range,
            "outlier_z": self.outlier_z,
            "min_cluster_size": self.min_cluster_size,
            "distance_metric": self.distance_metric,
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
            "hierarchical_linkage": self.hierarchical_linkage,
            "random_state": self.random_state,
            "rotation_max_k": self.rotation_max_k,
            **self.extra
        }
        if self.ensemble_pipelines is not None:
            out["ensemble_pipelines"] = self.ensemble_pipelines
            out["ensemble_method"] = self.ensemble_method
        return out

class ScopeCfg:
    def __init__(self, data_root, scope="full", task="force",
            rotation_label_mode="good_bad", experiment_ids=None,
            experiment_paths=None, max_experiments=None, max_depth=6,
            require_labels=True, extension_index=0):
        self.data_root = data_root
        self.scope = scope
        self.task = task
        self.rotation_label_mode = rotation_label_mode
        self.experiment_ids = experiment_ids
        self.experiment_paths = experiment_paths
        self.max_experiments = max_experiments
        self.max_depth = max_depth
        self.require_labels = require_labels
        self.extension_index = extension_index

def ppln_cfg_from_dict(d):
    known = {
        "representation", "dim_reduction", "dim_reduction_n_components",
        "clustering", "n_clusters", "max_k", "n_grid", "z_thresh",
        "use_full_extension_range", "outlier_z", "min_cluster_size",
        "distance_metric", "dbscan_eps", "dbscan_min_samples",
        "hierarchical_linkage", "random_state", "rotation_max_k",
        "ensemble_pipelines", "ensemble_method"
    }
    extra = {k: v for k, v in d.items() if k not in known}
    kwargs = {k: v for k, v in d.items() if k in known}
    return PipelineCfg(extra=extra, **kwargs)

def get_default_cfg():
    return PipelineCfg(
        representation=REPRESENTATION_COMMON_GRID_INTERSECTION,
        dim_reduction=DIM_RED_NONE,
        clustering=CLUSTER_KMEANS,
        n_grid=200,
        z_thresh=3.0,
        use_full_extension_range=False,
        outlier_z=3.0,
        max_k=2,
        random_state=0,
    )

def ppln_preset_manifold_kmeans(**kwargs):
    c = get_default_cfg()
    c.representation = REPRESENTATION_COMMON_GRID_INTERSECTION
    c.dim_reduction = DIM_RED_MANIFOLD_UMAP
    c.dim_reduction_n_components = 10
    c.clustering = CLUSTER_KMEANS
    for k, v in kwargs.items():
        if hasattr(c, k):
            setattr(c, k, v)
    return c

def ppln_preset_manifold_dbscan(**kwargs):
    c = get_default_cfg()
    c.representation = REPRESENTATION_COMMON_GRID_INTERSECTION
    c.dim_reduction = DIM_RED_MANIFOLD_UMAP
    c.dim_reduction_n_components = 10
    c.clustering = CLUSTER_DBSCAN
    for k, v in kwargs.items():
        if hasattr(c, k):
            setattr(c, k, v)
    return c

def ppln_preset_partitioned_kmeans(**kwargs):
    c = get_default_cfg()
    c.representation = REPRESENTATION_COMMON_GRID_INTERSECTION
    c.dim_reduction = DIM_RED_NONE
    c.clustering = CLUSTER_KMEANS
    for k, v in kwargs.items():
        if hasattr(c, k):
            setattr(c, k, v)
    return c

def get_pipelines_for_comparison():
    with open(FORCE_PIPELINES_JSON, encoding="utf-8") as f:
        entries = json.load(f)
    return [(e["name"], ppln_cfg_from_dict({k: v for k, v in e.items() if k != "name"})) for e in entries]

def get_ppln_cfg_by_name(name):
    for n, cfg in get_pipelines_for_comparison():
        if n == name:
            return cfg
    return None

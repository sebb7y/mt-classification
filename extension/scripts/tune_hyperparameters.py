import argparse
import copy
import csv
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TUNING_GRIDS = {
    "lle_kmeans": {
        "param_grid": [
            {"n_grid": 100, "dim_reduction_n_components": 5},
            {"n_grid": 100, "dim_reduction_n_components": 10},
            {"n_grid": 100, "dim_reduction_n_components": 15},
            {"n_grid": 200, "dim_reduction_n_components": 5},
            {"n_grid": 200, "dim_reduction_n_components": 10},
            {"n_grid": 200, "dim_reduction_n_components": 15},
            {"n_grid": 300, "dim_reduction_n_components": 5},
            {"n_grid": 300, "dim_reduction_n_components": 10},
            {"n_grid": 300, "dim_reduction_n_components": 15},
            {"n_grid": 200, "dim_reduction_n_components": 10, "n_neighbors": 8},
            {"n_grid": 200, "dim_reduction_n_components": 10, "n_neighbors": 12},
            {"n_grid": 200, "dim_reduction_n_components": 10, "n_neighbors": 20}
        ],
        "param_distributions": {
            "n_grid": [100, 150, 200, 250, 300],
            "dim_reduction_n_components": [5, 8, 10, 12, 15],
            "n_neighbors": [6, 8, 10, 12, 15, 20]
        }
    },
    "umap_dbscan": {
        "param_grid": [
            {"n_grid": 100, "dim_reduction_n_components": 5, "dbscan_min_samples": 3},
            {"n_grid": 100, "dim_reduction_n_components": 5, "dbscan_min_samples": 5},
            {"n_grid": 100, "dim_reduction_n_components": 5, "dbscan_min_samples": 8},
            {"n_grid": 100, "dim_reduction_n_components": 10, "dbscan_min_samples": 3},
            {"n_grid": 100, "dim_reduction_n_components": 10, "dbscan_min_samples": 5},
            {"n_grid": 100, "dim_reduction_n_components": 10, "dbscan_min_samples": 8},
            {"n_grid": 200, "dim_reduction_n_components": 5, "dbscan_min_samples": 3},
            {"n_grid": 200, "dim_reduction_n_components": 5, "dbscan_min_samples": 5},
            {"n_grid": 200, "dim_reduction_n_components": 5, "dbscan_min_samples": 8},
            {"n_grid": 200, "dim_reduction_n_components": 10, "dbscan_min_samples": 3},
            {"n_grid": 200, "dim_reduction_n_components": 10, "dbscan_min_samples": 5},
            {"n_grid": 200, "dim_reduction_n_components": 10, "dbscan_min_samples": 8}
        ],
        "param_distributions": {
            "n_grid": [100, 150, 200],
            "dim_reduction_n_components": [5, 8, 10, 12],
            "dbscan_min_samples": [3, 5, 6, 8, 10]
        }
    },
    "laplacian_dbscan": {
        "param_grid": [
            {"n_grid": 100, "dim_reduction_n_components": 5, "dbscan_min_samples": 5},
            {"n_grid": 100, "dim_reduction_n_components": 10, "dbscan_min_samples": 5},
            {"n_grid": 100, "dim_reduction_n_components": 15, "dbscan_min_samples": 5},
            {"n_grid": 200, "dim_reduction_n_components": 5, "dbscan_min_samples": 5},
            {"n_grid": 200, "dim_reduction_n_components": 10, "dbscan_min_samples": 5},
            {"n_grid": 200, "dim_reduction_n_components": 15, "dbscan_min_samples": 5},
            {"n_grid": 200, "dim_reduction_n_components": 10, "dbscan_min_samples": 3},
            {"n_grid": 200, "dim_reduction_n_components": 10, "dbscan_min_samples": 8},
            {"n_grid": 200, "dim_reduction_n_components": 10, "n_neighbors": 8, "dbscan_min_samples": 5},
            {"n_grid": 200, "dim_reduction_n_components": 10, "n_neighbors": 12, "dbscan_min_samples": 5},
        ],
        "param_distributions": {
            "n_grid": [100, 150, 200],
            "dim_reduction_n_components": [5, 8, 10, 12, 15],
            "n_neighbors": [6, 8, 10, 12, 15],
            "dbscan_min_samples": [3, 5, 8]
        }
    }
}

def cfg_from_base_params(base, params):
    from force_extension.modular_eval.config import PipelineCfg

    known = {
        "representation", "dim_reduction", "dim_reduction_n_components",
        "clustering", "n_clusters", "max_k", "n_grid", "z_thresh",
        "use_full_extension_range", "outlier_z", "min_cluster_size",
        "distance_metric", "dbscan_eps", "dbscan_min_samples",
        "hierarchical_linkage", "random_state", "ensemble_pipelines", "ensemble_method"
    }
    cfg = copy.deepcopy(base)
    for k, v in params.items():
        if k in known:
            setattr(cfg, k, v)
        else:
            extra = dict(cfg.extra)
            extra[k] = v
            cfg.extra = extra
    return cfg

def main():
    from force_extension.modular_eval.config import ScopeCfg, get_ppln_cfg_by_name
    from force_extension.modular_eval.runner import run_bulk_return_metrics

    parser = argparse.ArgumentParser(
        description="grid or random search over hyperparameters for top pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("data_root", help="data root for experiments")
    parser.add_argument("-o", "--out", default="out/tuning_results.csv", help="output csv of all runs")
    parser.add_argument("--method", choices=["grid", "random"], default="grid")
    parser.add_argument("--pipeline", choices=["lle_kmeans", "umap_dbscan", "laplacian_dbscan", "all"],
                        default="all")
    parser.add_argument("--n-iter", type=int, default=20, help="for random search: number of param combinations")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--extension-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    scope = ScopeCfg(
        data_root=data_root,
        max_experiments=args.max_experiments,
        require_labels=True,
        extension_index=args.extension_index,
    )

    pipelines = ["lle_kmeans", "umap_dbscan", "laplacian_dbscan"] if args.pipeline == "all" else [args.pipeline]
    random.seed(args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_path = os.path.abspath(args.out)
    results = []

    for pipeline_name in pipelines:
        base = get_ppln_cfg_by_name(pipeline_name)
        if base is None:
            print(f"unknown pipeline: {pipeline_name}", file=sys.stderr)
            continue
        grid_spec = TUNING_GRIDS.get(pipeline_name)
        if not grid_spec:
            print(f"no tuning grid for {pipeline_name}", file=sys.stderr)
            continue

        if args.method == "grid":
            param_list = grid_spec["param_grid"]
        else:
            dist = grid_spec["param_distributions"]
            param_list = []
            for _ in range(args.n_iter):
                combo = {}
                for k, values in dist.items():
                    combo[k] = random.choice(values)
                param_list.append(combo)

        print(f"\n{pipeline_name} ({args.method}): {len(param_list)} combinations")
        best_acc = -1.0
        best_params = None

        for i, params in enumerate(param_list):
            cfg = cfg_from_base_params(base, params)
            mean_acc, mean_ari, n_ok, n_fail = run_bulk_return_metrics(scope, cfg, fail_skip=True)
            acc = mean_acc if mean_acc is not None else -1.0
            ari = mean_ari if mean_ari is not None else None
            row = {
                "pipeline": pipeline_name,
                "mean_accuracy": acc,
                "mean_ari": ari if ari is not None else "",
                "n_ok": n_ok,
                "n_fail": n_fail,
                "params": json.dumps(params, sort_keys=True)
            }
            row.update(params)
            results.append(row)
            ari_str = f"{ari:.3f}" if ari is not None else "nan"
            print(f"  [{i+1}/{len(param_list)}] acc={acc:.3f} ari={ari_str}  params={params}")
            if acc > best_acc:
                best_acc = acc
                best_params = params

        print(f"  Best: acc={best_acc:.3f}  params={best_params}")

    if not results:
        print("no results to write", file=sys.stderr)
        return 1

    all_keys = set()
    for r in results:
        all_keys.update(k for k in r if k not in ("params", "mean_accuracy", "mean_ari", "n_ok", "n_fail", "pipeline"))
    fieldnames = ["pipeline", "mean_accuracy", "mean_ari", "n_ok", "n_fail", "params"] + sorted(all_keys)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nwrote {out_path}")

    by_pipeline = {}
    for r in results:
        name = r["pipeline"]
        if name not in by_pipeline:
            by_pipeline[name] = []
        by_pipeline[name].append(r)
    print("\ntuning stats summary")
    for name in sorted(by_pipeline.keys()):
        rows = by_pipeline[name]
        accs = [float(r["mean_accuracy"]) for r in rows if r.get("mean_accuracy") != ""]
        aris = [float(r["mean_ari"]) for r in rows if r.get("mean_ari") not in (None, "")]
        best = max(rows, key=lambda x: float(x.get("mean_accuracy") or -1))
        total_ok = sum(int(r.get("n_ok") or 0) for r in rows)
        total_fail = sum(int(r.get("n_fail") or 0) for r in rows)
        print(f"\n  {name}:")
        print(f"    runs: {len(rows)}, total n_ok={total_ok}, total n_fail={total_fail}")
        if accs:
            print(f"    mean_accuracy: min={min(accs):.3f}, max={max(accs):.3f}, mean={sum(accs)/len(accs):.3f}")
        if aris:
            print(f"    mean_ari:       min={min(aris):.3f}, max={max(aris):.3f}, mean={sum(aris)/len(aris):.3f}")
        print(f"    best params: {json.loads(best.get('params') or '{}')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

import copy
import csv
import os
import sys
import time
from .config import PipelineCfg, ScopeCfg, get_default_cfg, get_ppln_cfg_by_name
from .representation import feat_mat, build_rot_feature_matrix
from .dim_reduction import reduce_dimensions
from .clustering import cluster
from ..pipeline import load_exp, build_exts_from_exp
from ..preprocess import attach_script_regions
from ..evaluation.metrics import cluster_metrics, ensemble_all, label_cols, pack_metrics, y_true_y_pred_from_clustering
from ..evaluation.bulk import disc_force_ext_experiments, disc_txt_experiments
from pathlib import Path


def disc_experiments(data_root, max_depth=6, require_labels=True, task='force', rotation_label_mode='good_bad'):
    if task == "rotation":
        require_cu = rotation_label_mode in ("coilable_uncoilable", "coilable_uncoilable_bad")
        return disc_txt_experiments(
            data_root,
            max_depth=max_depth,
            require_labels=require_labels,
            require_coilable_uncoilable=require_cu,
        )
    return disc_force_ext_experiments(
        data_root,
        max_depth=max_depth,
        require_labels=require_labels,
    )

def run_one_ext(ext, pipeline, label_scheme='good_bad'):
    import numpy as np
    all_beads_df = ext.get("all_beads_df")
    if all_beads_df is not None and hasattr(all_beads_df, "columns"):
        if label_scheme == "coilable_uncoilable_bad":
            n_coilable_real = sum(1 for c in all_beads_df.columns if str(c).startswith("gc_"))
            n_uncoilable_real = sum(1 for c in all_beads_df.columns if str(c).startswith("uc_"))
            n_bad_class_real = sum(1 for c in all_beads_df.columns if str(c).startswith("bb_"))
            n_good_real = n_coilable_real
            n_bad_real = n_uncoilable_real + n_bad_class_real
        elif label_scheme == "coilable_uncoilable":
            n_good_real = sum(1 for c in all_beads_df.columns if str(c).startswith("gc_"))
            n_bad_real = sum(1 for c in all_beads_df.columns if str(c).startswith("uc_"))
            n_coilable_real = n_uncoilable_real = n_bad_class_real = None
        else:
            n_good_real = sum(1 for c in all_beads_df.columns if str(c).startswith("gb_"))
            n_bad_real = sum(1 for c in all_beads_df.columns if str(c).startswith("bb_"))
            n_coilable_real = n_uncoilable_real = n_bad_class_real = None
    else:
        n_good_real = n_bad_real = n_coilable_real = n_uncoilable_real = n_bad_class_real = None
    run_info = {
        "n_good_real": n_good_real,
        "n_bad_real": n_bad_real,
        "n_coilable_real": n_coilable_real,
        "n_uncoilable_real": n_uncoilable_real,
        "n_bad_class_real": n_bad_class_real
    }

    if ext.get("region_kind") == "rotation":
        X, cols_kept, x_grid, meta = build_rot_feature_matrix(
            ext, n_grid=pipeline.n_grid, representation=pipeline.representation
        )
        pipeline_use = copy.copy(pipeline)
        pipeline_use.max_k = getattr(pipeline, "rotation_max_k", 3)
    else:
        X, cols_kept, x_grid, meta = feat_mat(
            ext,
            representation=pipeline.representation,
            n_grid=pipeline.n_grid,
            z_thresh=pipeline.z_thresh,
            use_full_extension_range=pipeline.use_full_extension_range,
        )
        pipeline_use = pipeline

    run_info["representation_meta"] = meta
    lb_cols = label_cols(ext, label_scheme)
    run_info["n_beads"] = len(lb_cols) if lb_cols else len(cols_kept)
    run_info["cols_kept"] = cols_kept
    if X.size == 0 or len(cols_kept) == 0:
        return None, np.array([]), run_info

    X_red, transformer = reduce_dimensions(
        X,
        method=pipeline_use.dim_reduction,
        n_components=pipeline_use.dim_reduction_n_components,
        random_state=pipeline_use.random_state,
        **pipeline_use.extra,
    )
    run_info["dim_reduction_fitted"] = transformer is not None

    labels, cluster_info = cluster(X_red, pipeline_use)
    run_info["cluster_info"] = cluster_info

    metrics_dict = None
    if cols_kept and len(labels) == len(cols_kept):
        metrics_dict = cluster_metrics(
            cols_kept, labels, X_red if X_red.size else None, ext=ext, label_scheme=label_scheme
        )
        if metrics_dict and isinstance(metrics_dict.get("silhouette"), float) and metrics_dict["silhouette"] != metrics_dict["silhouette"]:
            metrics_dict["silhouette"] = None
    return metrics_dict, labels, run_info

def run_one_exp(root_path, pipeline, extension_index=0, require_script=True, task='force', label_scheme='good_bad', load_kwargs=None):
    import numpy as np
    row = {
        "path": root_path,
        "experiment_id": os.path.basename(root_path.rstrip(os.sep)) or root_path,
        "representation": pipeline.representation,
        "dim_reduction": pipeline.dim_reduction,
        "clustering": pipeline.clustering,
        "accuracy": None,
        "ari": None,
        "nmi": None,
        "silhouette": None,
        "n_beads": None,
        "n_extensions": 0,
        "n_clusters": None,
        "cluster_counts": None,
        "n_outliers": None,
        "n_good_in_clustering": None,
        "n_bad_in_clustering": None,
        "tp": None,
        "tn": None,
        "fp": None,
        "fn": None,
        "chosen_cluster_size": None,
        "n_good_real": None,
        "n_bad_real": None,
        "n_coilable_real": None,
        "n_uncoilable_real": None,
        "n_bad_class_real": None,
        "error": None
    }
    try:
        experiment = load_exp(
            root_path,
            require_script=require_script,
            **(load_kwargs or {}),
        )
        attach_script_regions(experiment)
        exts = build_exts_from_exp(experiment, region_type=task)
        if not exts:
            row["error"] = "no_extensions" if task == "force" else "no_rotation_extensions"
            return row
        row["n_extensions"] = len(exts)
        per_ext = []

        if getattr(pipeline, "ensemble_pipelines", None):
            names = list(pipeline.ensemble_pipelines)
            configs = [get_ppln_cfg_by_name(n) for n in names]
            if None in configs:
                row["error"] = "ensemble_unknown_pipeline"
                return row
            for ext in exts:
                pred_by_bead = {}
                last_run_info = {}
                for name, cfg in zip(names, configs):
                    metrics_dict, labels, run_info = run_one_ext(ext, cfg, label_scheme=label_scheme)
                    last_run_info = run_info
                    cols_kept = run_info.get("cols_kept") or []
                    if not cols_kept or len(np.asarray(labels).ravel()) != len(cols_kept):
                        continue
                    pair = y_true_y_pred_from_clustering(
                        cols_kept, labels, ext=ext, label_scheme=label_scheme
                    )
                    if pair is None:
                        continue
                    y_true, y_pred, _ca = pair
                    y_pred = np.asarray(y_pred).ravel()
                    for i, bead in enumerate(cols_kept):
                        pred_by_bead.setdefault(bead, {})[name] = int(y_pred[i])
                ens = ensemble_all(ext, names, pred_by_bead, label_scheme)
                if ens is None:
                    continue
                y_true_full, y_pred_full = ens
                md = pack_metrics(y_true_full, y_pred_full, label_scheme)
                if md is None:
                    continue
                n_good_pred = int((y_pred_full == 1).sum())
                n_bad_pred = len(y_pred_full) - n_good_pred
                per_ext.append({
                    "accuracy": md["accuracy"],
                    "ari": md["ari"],
                    "nmi": md["nmi"],
                    "silhouette": None,
                    "n_beads": len(y_true_full),
                    "n_clusters": 2,
                    "cluster_counts": f"{n_good_pred},{n_bad_pred}",
                    "n_outliers": 0,
                    "n_good_in_clustering": n_good_pred,
                    "n_bad_in_clustering": n_bad_pred,
                    "tp": md["tp"],
                    "tn": md["tn"],
                    "fp": md["fp"],
                    "fn": md["fn"],
                    "chosen_cluster_size": md["chosen_cluster_size"],
                    "n_good_real": last_run_info.get("n_good_real"),
                    "n_bad_real": last_run_info.get("n_bad_real")
                })
            if not per_ext:
                row["error"] = "ensemble_no_common_beads_or_no_gt"
                return row
        else:
            for ext in exts:
                metrics_dict, labels, run_info = run_one_ext(
                    ext, pipeline, label_scheme=label_scheme
                )
                if labels is None:
                    labels_arr = np.array([])
                else:
                    labels_arr = np.asarray(labels).ravel()

                if len(labels_arr) > 0:
                    unique, counts = np.unique(labels_arr, return_counts=True)
                else:
                    unique = np.array([])
                    counts = np.array([])

                if -1 in unique and len(counts) > 0:
                    n_outliers = int(counts[unique == -1].sum())
                else:
                    n_outliers = 0

                cluster_counts_list = []
                for cluster_id in sorted(unique):
                    if cluster_id < 0:
                        continue
                    cluster_counts_list.append(int(counts[unique == cluster_id][0]))

                if cluster_counts_list:
                    cluster_counts_str = ",".join(map(str, cluster_counts_list))
                else:
                    cluster_counts_str = ""
                n_clusters = run_info.get("cluster_info", {}).get("n_clusters_", len(cluster_counts_list))
                if metrics_dict is not None:
                    per_ext.append({
                        "accuracy": metrics_dict["accuracy"],
                        "ari": metrics_dict.get("ari"),
                        "nmi": metrics_dict.get("nmi"),
                        "silhouette": metrics_dict.get("silhouette"),
                        "n_beads": run_info["n_beads"],
                        "n_clusters": n_clusters,
                        "cluster_counts": cluster_counts_str,
                        "n_outliers": n_outliers,
                        "n_good_in_clustering": metrics_dict.get("n_good"),
                        "n_bad_in_clustering": metrics_dict.get("n_bad"),
                        "tp": metrics_dict.get("tp"),
                        "tn": metrics_dict.get("tn"),
                        "fp": metrics_dict.get("fp"),
                        "fn": metrics_dict.get("fn"),
                        "chosen_cluster_size": metrics_dict.get("chosen_cluster_size"),
                        "n_good_real": run_info.get("n_good_real"),
                        "n_bad_real": run_info.get("n_bad_real"),
                        "n_coilable_real": run_info.get("n_coilable_real"),
                        "n_uncoilable_real": run_info.get("n_uncoilable_real"),
                        "n_bad_class_real": run_info.get("n_bad_class_real"),
                        "representation": run_info.get("representation_meta", {}).get("representation", pipeline.representation)
                    })
                else:
                    n_beads = run_info.get("n_beads", 0)
                    per_ext.append({
                        "accuracy": None,
                        "ari": None,
                        "nmi": None,
                        "silhouette": run_info.get("cluster_info", {}).get("silhouette") if n_beads > 0 else None,
                        "n_beads": n_beads,
                        "n_clusters": n_clusters if n_beads > 0 else 0,
                        "cluster_counts": cluster_counts_str if n_beads > 0 else "",
                        "n_outliers": n_outliers if n_beads > 0 else 0,
                        "n_good_in_clustering": None,
                        "n_bad_in_clustering": None,
                        "tp": None,
                        "tn": None,
                        "fp": None,
                        "fn": None,
                        "chosen_cluster_size": None,
                        "n_good_real": run_info.get("n_good_real"),
                        "n_bad_real": run_info.get("n_bad_real"),
                        "n_coilable_real": run_info.get("n_coilable_real"),
                        "n_uncoilable_real": run_info.get("n_uncoilable_real"),
                        "n_bad_class_real": run_info.get("n_bad_class_real"),
                        "representation": run_info.get("representation_meta", {}).get("representation", pipeline.representation)
                    })
        if not per_ext:
            row["error"] = "no_labels_or_empty"
            return row
        if extension_index >= 0:
            idx = min(extension_index, len(per_ext) - 1)
            p = per_ext[idx]
            row["accuracy"] = p["accuracy"]
            row["ari"] = p["ari"]
            row["nmi"] = p["nmi"]
            row["silhouette"] = p["silhouette"]
            row["n_beads"] = p["n_beads"]
            row["n_clusters"] = p["n_clusters"]
            row["cluster_counts"] = p["cluster_counts"]
            row["n_outliers"] = p["n_outliers"]
            row["n_good_in_clustering"] = p["n_good_in_clustering"]
            row["n_bad_in_clustering"] = p["n_bad_in_clustering"]
            row["tp"] = p["tp"]
            row["tn"] = p["tn"]
            row["fp"] = p["fp"]
            row["fn"] = p["fn"]
            row["chosen_cluster_size"] = p["chosen_cluster_size"]
            row["n_good_real"] = p["n_good_real"]
            row["n_bad_real"] = p["n_bad_real"]
            row["n_coilable_real"] = p.get("n_coilable_real")
            row["n_uncoilable_real"] = p.get("n_uncoilable_real")
            row["n_bad_class_real"] = p.get("n_bad_class_real")
            row["representation"] = p.get("representation", row["representation"])
        else:
            accs = [x["accuracy"] for x in per_ext if x["accuracy"] is not None]
            row["accuracy"] = sum(accs) / len(accs) if accs else None
            aris = [x["ari"] for x in per_ext if x["ari"] is not None]
            nmis = [x["nmi"] for x in per_ext if x["nmi"] is not None]
            sils = [x["silhouette"] for x in per_ext if x["silhouette"] is not None]
            row["ari"] = sum(aris) / len(aris) if aris else None
            row["nmi"] = sum(nmis) / len(nmis) if nmis else None
            row["silhouette"] = sum(sils) / len(sils) if sils else None
            row["n_beads"] = sum(x["n_beads"] for x in per_ext)
            row["n_clusters"] = per_ext[0]["n_clusters"]
            row["cluster_counts"] = per_ext[0]["cluster_counts"]
            row["n_outliers"] = per_ext[0]["n_outliers"]
            row["n_good_in_clustering"] = per_ext[0]["n_good_in_clustering"]
            row["n_bad_in_clustering"] = per_ext[0]["n_bad_in_clustering"]
            row["tp"] = per_ext[0]["tp"]
            row["tn"] = per_ext[0]["tn"]
            row["fp"] = per_ext[0]["fp"]
            row["fn"] = per_ext[0]["fn"]
            row["chosen_cluster_size"] = per_ext[0]["chosen_cluster_size"]
            row["n_good_real"] = per_ext[0]["n_good_real"]
            row["n_bad_real"] = per_ext[0]["n_bad_real"]
            row["n_coilable_real"] = per_ext[0].get("n_coilable_real")
            row["n_uncoilable_real"] = per_ext[0].get("n_uncoilable_real")
            row["n_bad_class_real"] = per_ext[0].get("n_bad_class_real")
            row["representation"] = per_ext[0].get("representation", row["representation"])
    except Exception as e:
        row["error"] = str(e)
    return row

def run_bulk(scope, pipeline, output_csv=None, fail_skip=True, overwrite=False):
    rotation_label_mode = getattr(scope, "rotation_label_mode", "good_bad")
    experiments = disc_experiments(
        scope.data_root,
        max_depth=scope.max_depth,
        require_labels=scope.require_labels,
        task=getattr(scope, "task", "force"),
        rotation_label_mode=rotation_label_mode,
    )
    if scope.experiment_paths:
        paths_wanted = set(os.path.abspath(p) for p in scope.experiment_paths)
        experiments = [e for e in experiments if e["path"] in paths_wanted]
    elif scope.experiment_ids:
        ids_set = set(scope.experiment_ids)
        experiments = [e for e in experiments if e["experiment_id"] in ids_set or e["rel_path"] in ids_set]
    if scope.max_experiments is not None:
        experiments = experiments[: scope.max_experiments]
    if not experiments:
        return 0, 0
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
        fieldnames = [
            "path", "rel_path", "experiment_id", "source",
            "representation", "dim_reduction", "clustering",
            "accuracy", "ari", "nmi", "silhouette", "n_beads", "n_extensions",
            "n_clusters", "cluster_counts", "n_outliers",
            "n_good_in_clustering", "n_bad_in_clustering", "tp", "tn", "fp", "fn", "chosen_cluster_size",
            "n_good_real", "n_bad_real",
            "n_coilable_real", "n_uncoilable_real", "n_bad_class_real",
            "error",
        ]
        file_exists = os.path.isfile(output_csv) and not overwrite
        f = open(output_csv, "w" if overwrite else "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore",
            quoting=csv.QUOTE_NONNUMERIC,
        )
        if overwrite or not file_exists:
            writer.writeheader()
    n_ok, n_fail = 0, 0
    for exp in experiments:
        load_kwargs = None
        if rotation_label_mode in ("coilable_uncoilable", "coilable_uncoilable_bad") and exp.get("goodbeads_coilable_txt") and exp.get("goodbeads_uncoilable_txt"):
            load_kwargs = {
                "goodbeads_coilable_path": exp["goodbeads_coilable_txt"],
                "goodbeads_uncoilable_path": exp["goodbeads_uncoilable_txt"]
            }
        row = run_one_exp(
            exp["path"],
            pipeline,
            extension_index=scope.extension_index,
            task=getattr(scope, "task", "force"),
            label_scheme=rotation_label_mode,
            load_kwargs=load_kwargs,
        )
        row["rel_path"] = exp["rel_path"]
        row["experiment_id"] = exp["experiment_id"]
        row["source"] = exp["source"]
        if row.get("error"):
            n_fail += 1
            if not fail_skip:
                if output_csv:
                    f.close()
                raise RuntimeError(f"{exp['path']}: {row['error']}")
        else:
            n_ok += 1
        if output_csv:
            writer.writerow(row)
            f.flush()
    if output_csv:
        f.close()
    return n_ok, n_fail

def run_bulk_return_metrics(scope, pipeline, fail_skip=True):
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        tmp = f.name
    try:
        n_ok, n_fail = run_bulk(scope, pipeline, output_csv=tmp, fail_skip=fail_skip, overwrite=True)
        if n_ok == 0:
            return None, None, 0, n_fail
        with open(tmp, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("error") in (None, "")]
        if not rows:
            return None, None, 0, n_fail
        accs = [float(r["accuracy"]) for r in rows if r.get("accuracy") not in (None, "")]
        aris = [float(r["ari"]) for r in rows if r.get("ari") not in (None, "")]
        mean_acc = sum(accs) / len(accs) if accs else None
        mean_ari = sum(aris) / len(aris) if aris else None
        return mean_acc, mean_ari, n_ok, n_fail
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

def estimate_runtime(scope, pipeline, n_sample=3):
    rotation_label_mode = getattr(scope, "rotation_label_mode", "good_bad")
    experiments = disc_experiments(
        scope.data_root,
        max_depth=scope.max_depth,
        require_labels=scope.require_labels,
        task=getattr(scope, "task", "force"),
        rotation_label_mode=rotation_label_mode,
    )
    if scope.experiment_paths:
        paths_wanted = set(os.path.abspath(p) for p in scope.experiment_paths)
        experiments = [e for e in experiments if e["path"] in paths_wanted]
    elif scope.experiment_ids:
        ids_set = set(scope.experiment_ids)
        experiments = [e for e in experiments if e["experiment_id"] in ids_set or e["rel_path"] in ids_set]
    if scope.max_experiments is not None:
        experiments = experiments[: scope.max_experiments]
    n_total = len(experiments)
    if n_total == 0:
        return 0.0, 0, 0.0
    sample = experiments[: min(n_sample, n_total)]
    start = time.perf_counter()
    for exp in sample:
        load_kwargs = None
        if rotation_label_mode in ("coilable_uncoilable", "coilable_uncoilable_bad") and exp.get("goodbeads_coilable_txt") and exp.get("goodbeads_uncoilable_txt"):
            load_kwargs = {"goodbeads_coilable_path": exp["goodbeads_coilable_txt"], "goodbeads_uncoilable_path": exp["goodbeads_uncoilable_txt"]}
        run_one_exp(
            exp["path"],
            pipeline,
            extension_index=scope.extension_index,
            task=getattr(scope, "task", "force"),
            label_scheme=rotation_label_mode,
            load_kwargs=load_kwargs,
        )
    elapsed = time.perf_counter() - start
    per_exp = elapsed / len(sample) if sample else 0.0
    estimated_total = per_exp * n_total
    return estimated_total, n_total, per_exp

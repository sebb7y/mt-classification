import argparse
import csv
import json
import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_PIPELINE_CFG = os.path.join(REPO_ROOT, "configs", "force_pipelines.json")

def read_ppln_names(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data and isinstance(data[0], dict):
        return [entry["name"] for entry in data]
    return data

def main():
    from force_extension.modular_eval.config import ScopeCfg, get_ppln_cfg_by_name
    from force_extension.modular_eval.runner import run_bulk

    parser = argparse.ArgumentParser(
        description="run modular evaluation (force-extension or rotation) with multiple pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("data_root", nargs="?", default=None,
                        help="data root directory")
    parser.add_argument("--task", choices=["force", "rotation"], default="force",
                        help="force or rotation (default: force)")
    parser.add_argument("--out-dir", "-o", default="out",
                        help="output directory (default: out)")
    parser.add_argument("--pipeline-config", default=None, metavar="PATH",
                        help="json pipeline registry (default: configs/force_pipelines.json)")
    parser.add_argument("--pipelines", nargs="+", metavar="NAME",
                        help="run only these pipeline names")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--no-require-labels", action="store_true")
    parser.add_argument("--extension-index", type=int, default=0)
    parser.add_argument("--rotation-label-mode",
                        choices=["good_bad", "coilable_uncoilable", "coilable_uncoilable_bad"],
                        default="coilable_uncoilable_bad")
    parser.add_argument("--regen", nargs="?", const="all", default=None, metavar="NAME[,NAME]|all",
                        help="force re-run all pipelines (default) or comma-separated pipeline names")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    pipeline_config_path = args.pipeline_config or DEFAULT_PIPELINE_CFG
    all_pipeline_names = read_ppln_names(pipeline_config_path)

    if args.pipelines:
        unknown = [n for n in args.pipelines if n not in all_pipeline_names]
        if unknown:
            parser.error(
                f"unknown pipeline name(s): {', '.join(unknown)}\n"
                f"valid names (from {os.path.relpath(pipeline_config_path)}): "
                f"{', '.join(all_pipeline_names)}"
            )
        selected_names = [n for n in all_pipeline_names if n in set(args.pipelines)]
    else:
        selected_names = list(all_pipeline_names)

    pipelines = []
    for name in selected_names:
        cfg = get_ppln_cfg_by_name(name)
        if cfg is None:
            print(f"warning: skipping '{name}' no config found")
            continue
        pipelines.append((name, cfg))

    if not args.data_root:
        parser.error("data_root required")
    data_root = os.path.abspath(args.data_root)

    scope_kwargs = dict(
        data_root=data_root,
        scope="full",
        max_experiments=args.max_experiments,
        max_depth=args.max_depth,
        require_labels=not args.no_require_labels,
        extension_index=args.extension_index,
    )
    if args.task == "rotation":
        scope_kwargs["task"] = "rotation"
        scope_kwargs["rotation_label_mode"] = args.rotation_label_mode
    scope = ScopeCfg(**scope_kwargs)

    regen_all = args.regen is not None and args.regen.strip() == "all"
    regen_names = (
        {s.strip() for s in args.regen.split(",") if s.strip()}
        if args.regen is not None and not regen_all
        else None
    )

    for name, pipeline in pipelines:
        run_dir = os.path.join(runs_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        results_csv = os.path.join(run_dir, "results.csv")
        force_run = regen_all or (regen_names is not None and name in regen_names)
        if os.path.isfile(results_csv) and not force_run:
            print(f"skipping (already run): {name}")
            continue
        print(f"running pipeline: {name}")
        n_ok, n_fail = run_bulk(scope, pipeline, output_csv=results_csv, fail_skip=True, overwrite=True)
        print(f"  ok={n_ok}, failed={n_fail}")
        if n_fail > 0 and os.path.isfile(results_csv):
            with open(results_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("error"):
                        print(f"  first error: {row['error'][:200]}")
                        break
        try:
            with open(os.path.join(run_dir, "pipeline.json"), "w", encoding="utf-8") as f:
                json.dump(pipeline.to_dict(), f, indent=2)
        except Exception as e:
            print(f"  could not write pipeline.json: {e}")
        try:
            meta = {
                "data_root": data_root,
                "task": args.task,
                "n_experiments_ok": n_ok,
                "n_experiments_fail": n_fail
            }
            with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    return 0

if __name__ == "__main__":
    sys.exit(main())

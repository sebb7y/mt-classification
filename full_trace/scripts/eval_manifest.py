import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

def find_pickle(exp_folder):
    from experimental_loader import find_pickle_path
    return find_pickle_path(exp_folder) is not None

def traces_shape(traces_path):
    import numpy as np
    try:
        with open(traces_path, "rb") as f:
            np.load(f, allow_pickle=True)
            blk = np.load(f)
        if blk.ndim == 3:
            if blk.shape[1] == 3:
                n_beads = int(blk.shape[2])
            elif blk.shape[2] == 3:
                n_beads = int(blk.shape[1])
            else:
                n_beads = int(blk.shape[2])
            return (blk.shape[0], n_beads)
        if blk.ndim == 2:
            return blk.shape
        return None
    except Exception:
        return None

def is_test_subfolder(path):
    return any(p == "test" or p.startswith("test_") for p in Path(path).parts)

def top_level_source(path, root):
    rel_parts = Path(os.path.relpath(path, root)).parts
    if rel_parts:
        return rel_parts[0]
    return "."

def find_exp_folders(root, max_depth=4):
    root = os.path.abspath(root)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if is_test_subfolder(dirpath):
            continue
        rel = os.path.relpath(dirpath, root)
        depth = len(Path(rel).parts) if rel != "." else 0
        if depth > max_depth:
            dirnames.clear()
            continue
        if "traces.npy" in filenames:
            found.append(dirpath)
            dirnames.clear()
    return sorted(found)

def main():
    parser = argparse.ArgumentParser(description="build evaluation manifest (index of experiment folders)")
    parser.add_argument(
        "root",
        nargs="?",
        default=os.path.join(REPO_ROOT, "data"),
        help="root to scan (default: data)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="output csv path (default: root/eval_manifest.csv)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="max depth under root to look for traces.npy (default 8 for nested layouts)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip shape/size check (faster, manifest will have empty n_frames/n_beads/size_mb)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="comma-separated top-level dir names to include (e.g. ssRNA,dsRNA) if not set include all",
    )
    parser.add_argument(
        "--exclude-file",
        default=None,
        help="path to file of experiment_ids to exclude (one per line)",
    )
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"error: root is not a directory: {root}", file=sys.stderr)
        sys.exit(1)
    out_path = args.output or os.path.join(root, "eval_manifest.csv")

    folders = find_exp_folders(root, args.max_depth)
    if args.sources:
        allowed = {s.strip() for s in args.sources.split(",") if s.strip()}
        folders = [f for f in folders if top_level_source(f, root) in allowed]
        print(f"found {len(folders)} experiment folders under {root} (sources: {sorted(allowed)})")
    else:
        print(f"found {len(folders)} experiment folders under {root}")

    rows = []
    for i, folder in enumerate(folders):
        rel = os.path.relpath(folder, root)
        traces_path = os.path.join(folder, "traces.npy")
        experiment_id = os.path.basename(folder) or rel.replace(os.sep, "_")
        source = top_level_source(folder, root)
        has_labels = find_pickle(folder)
        size_mb = None
        n_frames = n_beads = None
        if not args.quick:
            try:
                size_mb = round(os.path.getsize(traces_path) / (1024 * 1024), 2)
            except OSError:
                pass
            shape = traces_shape(traces_path)
            if shape is not None and len(shape) == 2:
                n_frames, n_beads = int(shape[0]), int(shape[1])
        has_both_classes = False
        if has_labels:
            if n_beads is None:
                shape = traces_shape(traces_path)
                if shape is not None and len(shape) == 2:
                    n_beads = int(shape[1])
            if n_beads is not None and n_beads > 0:
                try:
                    from experimental_loader import read_good_bead_mask
                    import numpy as np
                    mask = read_good_bead_mask(folder, int(n_beads))
                    has_both_classes = bool(np.any(mask) and np.any(~mask))
                except Exception:
                    pass
        rows.append({
            "path": folder,
            "rel_path": rel,
            "experiment_id": experiment_id,
            "source": source,
            "size_mb": size_mb if size_mb is not None else "",
            "n_frames": n_frames if n_frames is not None else "",
            "n_beads": n_beads if n_beads is not None else "",
            "has_labels": has_labels,
            "has_both_classes": has_both_classes
        })

    exclude_file = getattr(args, "exclude_file", None) or os.path.join(REPO_ROOT, "scripts", "eval_exclude_experiments.txt")
    if os.path.isfile(exclude_file):
        exclude_ids = set()
        with open(exclude_file) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    exclude_ids.add(line)
        if exclude_ids:
            n_before = len(rows)
            rows = [r for r in rows if (r.get("experiment_id") or "").strip() not in exclude_ids]
            if len(rows) < n_before:
                print(f"excluded {n_before - len(rows)} experiments from {exclude_file}")

    import csv
    fieldnames = ["path", "rel_path", "experiment_id", "source", "size_mb", "n_frames", "n_beads", "has_labels", "has_both_classes"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")
    with_labels = sum(1 for r in rows if r["has_labels"])
    with_both = sum(1 for r in rows if r.get("has_both_classes") is True)
    print(f"  with labels (pickle): {with_labels}")
    print(f"  with both classes (good and bad): {with_both}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

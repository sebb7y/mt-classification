import os
from pathlib import Path
from ..load.discovery import disc_force_ext_roots, discover

def disc_force_ext_experiments(data_root, max_depth=6, require_labels=True):
    data_root = os.path.abspath(data_root)
    roots = disc_force_ext_roots(data_root, max_depth=max_depth, require_labels=require_labels, require_script=True)
    out = []

    
    for root_path, has_labels in roots:
        rel = os.path.relpath(root_path, data_root)
        if rel == ".":
            experiment_id = os.path.basename(root_path) or "root"
            source = "."
        else:
            parts = Path(rel).parts
            if parts:
                experiment_id = parts[-1]
                source = parts[0]
            else:
                experiment_id = rel.replace(os.sep, "_")
                source = "."

        out.append({
            "path": root_path,
            "has_labels": has_labels,
            "rel_path": rel,
            "experiment_id": experiment_id,
            "source": source
        })
    out.sort(key=lambda e: e["path"])
    return out

def disc_txt_experiments(data_root, max_depth=1, require_labels=False, require_coilable_uncoilable=False):
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        return []
    out = []
    try:
        for name in sorted(os.listdir(data_root)):
            if name.startswith("."):
                continue

            path = os.path.join(data_root, name)
            if not os.path.isdir(path):
                continue

            manifest = discover(path, require_traces=True, require_script=True, require_good_beads=False)
            if manifest.get("errors"):
                continue

            if not manifest.get("paths", {}).get("traces_txt") or not manifest.get("paths", {}).get("script"):
                continue

            paths = manifest.get("paths", {})
            if require_coilable_uncoilable:
                gc_path = paths.get("goodbeads_coilable_txt")
                uc_path = paths.get("goodbeads_uncoilable_txt")
                if not gc_path or not uc_path:
                    continue

            rel = os.path.relpath(path, data_root)
            has_labels = manifest.get("has_good_beads", False)
            if require_labels and not has_labels:
                continue

            entry = {
                "path": path,
                "has_labels": has_labels,
                "rel_path": rel,
                "experiment_id": name,
                "source": "txt"
            }
            if require_coilable_uncoilable:
                entry["goodbeads_coilable_txt"] = paths.get("goodbeads_coilable_txt")
                entry["goodbeads_uncoilable_txt"] = paths.get("goodbeads_uncoilable_txt")
            out.append(entry)
    except OSError:
        pass
    return out

__all__ = [
    "disc_force_ext_experiments",
    "disc_txt_experiments",
]

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from force_extension.pipeline import load_exp
from force_extension.preprocess import attach_script_regions
from force_extension.pipeline import build_exts_from_exp

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "data", "npy_luca_3")
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        print(f"not a directory: {path}", file=sys.stderr)
        sys.exit(1)

    exp = load_exp(path, require_script=True)
    attach_script_regions(exp)

    script_path = (exp.get("paths") or {}).get("script") or (exp.get("paths") or {}).get("script_npy_path")
    print(f"script: {script_path}")
    print()

    gb_time = exp.get("gb_time")
    if gb_time is None or (hasattr(gb_time, "empty") and gb_time.empty):
        gb_time = exp.get("exp_time")
    t_min = t_max = None
    if gb_time is not None and not (hasattr(gb_time, "empty") and gb_time.empty):
        import numpy as np
        t = np.asarray(gb_time.iloc[:, 0], dtype=float)
        t_min, t_max = float(np.min(t)), float(np.max(t))
        n_frames = len(t)
        print(f"trace time range: [{t_min:.2f}, {t_max:.2f}] s  ({n_frames} frames)")
    else:
        print("trace time: (no gb_time/exp_time)")
    print()

    regions = exp.get("force_extension_regions") or []
    print(f"force-extension regions: {len(regions)}")
    for i, (t0, t1) in enumerate(regions):
        in_range = "yes" if t_min is not None and t_max is not None and t_min <= t0 and t1 <= t_max else "no (outside trace range)"
        print(f"  ext {i}: [{t0:.2f}, {t1:.2f}] s  duration={t1-t0:.2f}s  in trace range? {in_range}")
    print()

    exts = build_exts_from_exp(exp, region_type="force")
    print(f"built {len(exts)} force-extension extension(s)")
    for i, ext in enumerate(exts):
        reg = ext.get("region", [])
        gb = ext.get("gb_df")
        n_pts = len(gb) if gb is not None else 0
        print(f"  ext {i}: region [{reg[0]:.2f}, {reg[1]:.2f}] s  -> {n_pts} time points per bead")
    return 0

if __name__ == "__main__":
    sys.exit(main())

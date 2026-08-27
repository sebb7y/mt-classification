import os
from pathlib import Path

from .. import config

def has_traces_beadpos(folder):
    if not os.path.isdir(folder):
        return False
    traces_ok = any(
        os.path.isfile(os.path.join(folder, c))
        for c in config.TRACES_NPY_CANDIDATES
    )
    beadpos = os.path.join(folder, config.BEADPOS_XY_NAME)
    return bool(traces_ok and os.path.isfile(beadpos))

def has_script_or_cfg(folder):
    for cand in config.SCRIPT_NPY_CANDIDATES + config.SCRIPT_TXT_CANDIDATES:
        if os.path.isfile(os.path.join(folder, cand)):
            return True
    for cand in config.CFG_YAML_CANDIDATES:
        if os.path.isfile(os.path.join(folder, cand)):
            return True
    return False

def beadcount(path):
    bp = os.path.join(path, config.BEADPOS_XY_NAME)
    try:
        with open(bp, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip() and len(line.split()) >= 2)
    except Exception:
        return 0

def two_subdirs_traces_beadpos(root):
    if not os.path.isdir(root):
        return None
    subdirs = [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
        and has_traces_beadpos(os.path.join(root, name))
    ]
    if len(subdirs) != 2:
        return None
    s0, s1 = os.path.join(root, subdirs[0]), os.path.join(root, subdirs[1])
    n0, n1 = beadcount(s0), beadcount(s1)
    if n0 >= n1:
        return subdirs[0], subdirs[1]
    return subdirs[1], subdirs[0]

def two_subdirs_all_good_separate(root):
    if not os.path.isdir(root):
        return None
    subdirs = [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and not name.startswith(".")
    ]
    if len(subdirs) != 2:
        return None

    p0 = os.path.join(root, subdirs[0])
    p1 = os.path.join(root, subdirs[1])
    has_traces_0 = has_traces_beadpos(p0)
    has_traces_1 = has_traces_beadpos(p1)
    beadpos_only_0 = has_beadpos_only(p0)
    beadpos_only_1 = has_beadpos_only(p1)

    if has_traces_0 and (beadpos_only_1 or has_traces_beadpos(p1)):
        return subdirs[0], subdirs[1]
    if has_traces_1 and (beadpos_only_0 or has_traces_beadpos(p0)):
        return subdirs[1], subdirs[0]
    return None

def has_beadpos_only(folder):
    if not os.path.isdir(folder):
        return False
    beadpos = os.path.join(folder, config.BEADPOS_XY_NAME)
    if not os.path.isfile(beadpos):
        return False

    traces_ok = any(
        os.path.isfile(os.path.join(folder, c))
        for c in config.TRACES_NPY_CANDIDATES
    )
    return not traces_ok

def root_one_test_subdir(root):
    if not os.path.isdir(root):
        return None
    if not has_traces_beadpos(root):
        return None
    subdirs = [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
        and not name.startswith(".")
        and has_traces_beadpos(os.path.join(root, name))
    ]
    if len(subdirs) != 1:
        return None
    return subdirs[0]

def root_beadpos_only_one_test_subdir(root):
    if not os.path.isdir(root):
        return None
    if not has_beadpos_only(root):
        return None
    subdirs = [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
        and not name.startswith(".")
        and has_traces_beadpos(os.path.join(root, name))
    ]
    if len(subdirs) != 1:
        return None
    return subdirs[0]

def disc_force_ext_roots(data_root, max_depth=6, require_labels=False, require_script=True):
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        return []
    results = []

    def scan(current, depth):
        if depth > max_depth:
            return
        try:
            entries = os.listdir(current)
        except OSError:
            return

        subdirs = []
        for name in entries:
            if os.path.isdir(os.path.join(current, name)) and not name.startswith("."):
                subdirs.append(name)

        two = two_subdirs_traces_beadpos(current)
        if two is not None:
            script_ok = not require_script or has_script_or_cfg(current)
            if not script_ok:
                all_path = os.path.join(current, two[0])
                good_path = os.path.join(current, two[1])
                script_ok = has_script_or_cfg(all_path) or has_script_or_cfg(good_path)
            if script_ok:
                results.append((current, True))
            return
        two_sep = two_subdirs_all_good_separate(current)
        if two_sep is not None:
            script_ok = not require_script or has_script_or_cfg(current)
            if not script_ok:
                all_path = os.path.join(current, two_sep[0])
                good_path = os.path.join(current, two_sep[1])
                script_ok = has_script_or_cfg(all_path) or has_script_or_cfg(good_path)
            if script_ok:
                results.append((current, True))
            return

        test_subdir = root_one_test_subdir(current)
        if test_subdir is None:
            test_subdir = root_beadpos_only_one_test_subdir(current)
        if test_subdir is not None:
            test_path = os.path.join(current, test_subdir)
            script_ok = not require_script or has_script_or_cfg(current) or has_script_or_cfg(test_path)
            if script_ok:
                results.append((current, True))
            return

        has_traces_here = False
        for c in config.TRACES_NPY_CANDIDATES:
            if os.path.isfile(os.path.join(current, c)):
                has_traces_here = True
                break
        if has_traces_here and has_script_or_cfg(current) and not require_labels:
            results.append((current, False))
            return

        for name in subdirs:
            scan(os.path.join(current, name), depth + 1)

    scan(data_root, 0)
    return results

def discover(root, require_traces=True, require_script=True, require_good_beads=False):
    root = os.path.abspath(root)
    manifest = {'has_traces': False, 'has_script': False, 'has_good_beads': False, 'has_config': False, 'paths': {}, 'missing': [], 'errors': []}

    # this whole bit is kinda ugly, should improve later

    for cand in config.TRACES_TXT_CANDIDATES:
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            manifest["has_traces"] = True
            manifest["paths"]["traces_txt"] = p
            break
    if not manifest["has_traces"] and require_traces:
        manifest["missing"].append("traces")
        manifest["errors"].append(f"no traces file found under {root}")

    for cand in config.GOODBEADS_TXT_CANDIDATES:
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            manifest["has_good_beads"] = True
            manifest["paths"]["goodbeads_txt"] = p
            break
    if not manifest["has_good_beads"] and require_good_beads:
        manifest["missing"].append("good beads file")

    for cand in config.GOODBEADS_COILABLE_TXT_CANDIDATES:
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            manifest["paths"]["goodbeads_coilable_txt"] = p
            break
    for cand in config.GOODBEADS_UNCOILABLE_TXT_CANDIDATES:
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            manifest["paths"]["goodbeads_uncoilable_txt"] = p
            break

    for cand in config.SCRIPT_TXT_CANDIDATES:
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            manifest["has_script"] = True
            manifest["paths"]["script"] = p
            break
    if not manifest["has_script"] and config.SCRIPT_PARENT_FALLBACK:
        parent = os.path.dirname(root)
        for cand in config.SCRIPT_TXT_CANDIDATES:
            p = os.path.join(parent, cand)
            if os.path.isfile(p):
                manifest["has_script"] = True
                manifest["paths"]["script"] = p
                break
    if not manifest["has_script"] and require_script:
        manifest["missing"].append("magnet script")
        manifest["errors"].append(f"no script file found under {root}")

    for cand in config.CFG_YAML_CANDIDATES:
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            manifest["has_config"] = True
            manifest["paths"]["config_yaml"] = p
            break

    for cand in config.TRACES_NPY_CANDIDATES:
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            manifest["paths"]["traces_npy"] = p
            break
    if "traces_npy" not in manifest.get("paths", {}):
        for name in os.listdir(root):
            sub = os.path.join(root, name)
            if os.path.isdir(sub):
                p = os.path.join(sub, "traces.npy")
                if os.path.isfile(p):
                    manifest["paths"]["traces_npy"] = p
                    manifest["paths"]["traces_npy_subdir"] = name
                    break

    return manifest

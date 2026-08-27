import os
import numpy as np

from .script import parse_txt_script_to_unified, parse_npy_script_to_unified, get_mag_positions, get_mag_rotations
from .sections_force import get_force_ext_regions
from .sections_rot import get_rot_regions, get_rel_regions
from .force_equation import mag_pos_to_force

def attach_script_regions(experiment, script_path=None, force_equation='default', extension_speed=None, use_move_duration=None, extension_only=None, extension_when=None):
    paths = experiment.get("paths") or {}
    path = script_path or paths.get("script") or paths.get("script_npy_path")
    source = experiment.get("source", "")
    if not path or not os.path.isfile(path):
        experiment["unified_script"] = []
        experiment["force_extension_regions"] = []
        experiment["rotation_regions"] = []
        experiment["rel_regions"] = []
        experiment["mag_poss"] = np.array([])
        experiment["mag_rots"] = np.array([])
        experiment["forces"] = np.array([])
        if source == "npy":
            gb_time = experiment.get("gb_time")
            exp_time = experiment.get("exp_time")
            trace_data = experiment.get("trace_data")
            if gb_time is not None and len(gb_time) > 0:
                t = np.asarray(gb_time.iloc[:, 0], dtype=float)
            elif exp_time is not None and len(exp_time) > 0:
                t = np.asarray(exp_time.iloc[:, 0], dtype=float)
            elif trace_data is not None and trace_data.get("time") is not None:
                t = np.asarray(trace_data["time"][:, 0], dtype=float)
            else:
                t = None
            if t is not None and len(t) > 0:
                experiment["force_extension_regions"] = [[float(np.min(t)), float(np.max(t))]]
        return

    source = experiment.get("source", "")
    if source == "txt":
        with open(path, encoding="utf-8") as f:
            text = f.read()
        commands = [c.strip() for c in text.split(";") if c.strip()]
        unified = parse_txt_script_to_unified(commands)
    else:
        unified = parse_npy_script_to_unified(path)

    experiment["unified_script"] = unified
    ext_only = extension_only
    if ext_only is None and source == "npy":
        ext_only = True
    ext_when = extension_when
    if ext_when is None and source == "npy":
        ext_when = "mag_pos_decreasing"
    experiment["force_extension_regions"] = get_force_ext_regions(
        unified,
        desired_speed=extension_speed,
        use_move_duration=use_move_duration,
        extension_only=ext_only,
        extension_when=ext_when,
    )
    experiment["rotation_regions"] = get_rot_regions(unified)
    experiment["rel_regions"] = get_rel_regions(unified)

    gb_time = experiment.get("gb_time")
    exp_time = experiment.get("exp_time")
    trace_data = experiment.get("trace_data")
    if experiment.get("source") == "npy" and not experiment["force_extension_regions"]:
        if gb_time is not None and len(gb_time) > 0:
            t = np.asarray(gb_time.iloc[:, 0], dtype=float)
        elif exp_time is not None and len(exp_time) > 0:
            t = np.asarray(exp_time.iloc[:, 0], dtype=float)
        elif trace_data is not None and trace_data.get("time") is not None:
            t = np.asarray(trace_data["time"][:, 0], dtype=float)
        else:
            t = None
        if t is not None and len(t) > 0:
            experiment["force_extension_regions"] = [[float(np.min(t)), float(np.max(t))]]
    if gb_time is not None and len(gb_time) > 0:
        time_col = gb_time.iloc[:, 0]
    elif exp_time is not None and len(exp_time) > 0:
        time_col = exp_time.iloc[:, 0]
    elif trace_data is not None and trace_data.get("time") is not None:
        time_col = np.asarray(trace_data["time"][:, 0])
    else:
        time_col = None
    if time_col is not None:
        time_s = np.asarray(time_col, dtype=float)
        experiment["mag_poss"] = get_mag_positions(unified, time_s)
        experiment["mag_rots"] = get_mag_rotations(unified, time_s)
        experiment["forces"] = mag_pos_to_force(experiment["mag_poss"], equation=force_equation)
    else:
        experiment["mag_poss"] = np.array([])
        experiment["mag_rots"] = np.array([])
        experiment["forces"] = np.array([])

__all__ = [
    "parse_txt_script_to_unified",
    "parse_npy_script_to_unified",
    "get_mag_positions",
    "get_mag_rotations",
    "get_force_ext_regions",
    "get_rot_regions",
    "get_rel_regions",
    "mag_pos_to_force",
    "attach_script_regions",
]

from pathlib import Path

TRACES_TXT_CANDIDATES = [
    "traces/Experiment.txt",
    "Experiment.txt",
]
GOODBEADS_TXT_CANDIDATES = [
    "traces/Experiment_goodbeads.txt",
    "Experiment_goodbeads.txt",
]
GOODBEADS_COILABLE_TXT_CANDIDATES = [
    "traces/Goodbeads/Experiment_goodbeads_coilable.txt",
    "Goodbeads/Experiment_goodbeads_coilable.txt",
]
GOODBEADS_UNCOILABLE_TXT_CANDIDATES = [
    "traces/Goodbeads/Experiment_goodbeads_non-coilable.txt",
    "traces/Goodbeads/Experiment_goodbeads_non_coilable.txt",
    "traces/Goodbeads/Experiment_goodbeads_uncoilable.txt",
    "Goodbeads/Experiment_goodbeads_non-coilable.txt",
    "Goodbeads/Experiment_goodbeads_non_coilable.txt",
    "Goodbeads/Experiment_goodbeads_uncoilable.txt",
]
SCRIPT_TXT_CANDIDATES = [
    "Experiment_script.txt",
    "magnet-script.txt",
]
SCRIPT_NPY_CANDIDATES = [
    "magnet-script.txt",
]
CFG_YAML_CANDIDATES = [
    "config.yaml",
]
TRACES_NPY_CANDIDATES = [
    "traces.npy",
]
BEADPOS_XY_NAME = "beadpos_xy.txt"

SCRIPT_PARENT_FALLBACK = True

FORCE_EQUATION_NAMES = ["default"]

def force_equation_names():
    return list(FORCE_EQUATION_NAMES)

DEFAULT_FORCE_EXTENSION_SPEED = 0.3
DEFAULT_FORCE_EXTENSION_MIN_DISTANCE = 0.0
DEFAULT_FORCE_EXTENSION_USE_MOVE_DURATION = True
DEFAULT_FORCE_EXTENSION_EXTENSION_ONLY = False
DEFAULT_FORCE_EXTENSION_EXTENSION_WHEN = "mag_pos_increasing"
DEFAULT_ROT_ZERO_TOL = 1e-6
DEFAULT_REL_REGIONS_SPEED = 0.3
DEFAULT_REL_REGIONS_ALLOWANCE = 1

DEFAULT_N_GRID = 200
DEFAULT_Z_THRESH_RANGE = 3.0
DEFAULT_OUTLIER_Z = 3.0

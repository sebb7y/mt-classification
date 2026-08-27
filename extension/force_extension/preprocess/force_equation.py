import numpy as np
from .. import config

def mag_pos_to_force(positions, equation='default'):
    x = np.asarray(positions, dtype=float)
    if equation == "default":
        # equation from luca
        return 1.0 * (5.7061 * np.exp(-1.0203 * x) + 3.1215 * np.exp(-0.5843 * x))


        # add the other equations from pytweezer later
    raise ValueError(f"unknown force equation: {equation}, choose from {config.force_equation_names()}")

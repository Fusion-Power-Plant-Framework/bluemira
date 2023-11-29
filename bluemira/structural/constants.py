# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Constants for use in the structural module.
"""

import numpy as np
from matplotlib.pyplot import get_cmap

from bluemira.base.constants import EPS

# Poisson's ratio
NU = 0.33

# Shear deformation limit (r_g/L)
#   Above which shear deformation properties must be used
SD_LIMIT = 0.1

# The proximity tolerance
#   Used for checking node existence
D_TOLERANCE = 1e-5

# Small number tolerance
#   Used for checking if cos / sin of angles is actually zero
#   Chosen to be slightly larger than:
#      * np.cos(3 * np.pi / 2) =  -1.8369701987210297e-16
#      * np.sin(np.pi * 2) = -2.4492935982947064e-16
NEAR_ZERO = 2 * EPS

# The large displacement ratio (denominator)
R_LARGE_DISP = 100

# Global coordinate system

GLOBAL_COORDS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# Mapping of indices to load types
LOAD_MAPPING = {"Fx": 0, "Fy": 1, "Fz": 2, "Mx": 3, "My": 4, "Mz": 5}

# A list of all potential load types
LOAD_TYPES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

# Mapping of load string to direction vectors (for plotting purposes)
LOAD_STR_VECTORS = {
    "Fx": GLOBAL_COORDS[0],
    "Fy": GLOBAL_COORDS[1],
    "Fz": GLOBAL_COORDS[2],
    "Mx": GLOBAL_COORDS[0],
    "My": GLOBAL_COORDS[1],
    "Mz": GLOBAL_COORDS[2],
}

# Mapping of load integer to direction vectors (for plotting purposes)
LOAD_INT_VECTORS = dict(enumerate(LOAD_STR_VECTORS.values()))

# Mapping of indices to displacement/support types
DISP_MAPPING = {"Dx": 0, "Dy": 1, "Dz": 2, "Rx": 3, "Ry": 4, "Rz": 5}

# Number of interpolation points per Element
N_INTERP = 7

# Color map for Element stresses
STRESS_COLOR = get_cmap("seismic", 1000)

# Color map for Element deflections
DEFLECT_COLOR = get_cmap("viridis", 1000)

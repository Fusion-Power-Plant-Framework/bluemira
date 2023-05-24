# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Constants for use in the equilibria module.
"""

# Absolute tolerance on psi_norm values [n.a.]
#     Used to find last closed flux surface
PSI_NORM_TOL = 1e-2

# Relative tolerance on psi [n. a.]
#     Used as a convergence criterion for Picard iterations
PSI_REL_TOL = 2e-3

# Absolute tolerance on position [m]
#     Used to determine whether O- and X-points are the "same"
#     Used as an offset to determine if a point is "on" the edge of a coil
X_TOLERANCE = 1e-5

# Absolute tolerance on field [T]
#     Used as a threshold to determine if the poloidal field is null
B_TOLERANCE = 1e-3

# Absolute minimum radius for grid
X_AXIS_MIN = 0.1

# Relative grid search length sizer: factor of grid size
#      Used for distance from importance points
REL_GRID_SIZER = 0.1

# Minimum current in coil
#       Used when loading eqdsks from e.g. CREATE
I_MIN = 1e-20

# Minimum current density in toroidal current density array
J_TOR_MIN = 1e-36

# Breakdown field limit [T]
#       Used when calculating plasma breakdowns
B_BREAKDOWN = 3e-3

# The number of MN per 1 m arrow (plot length) [m per N]
#       Used when plotting force arrows
M_PER_MN = 100e6

# The maximum allowable field at a NbTi superconducting coil [T]
#       Used when calculating coil constraints
NBTI_B_MAX = 11.5  # Eyeball // Louis Zani mentioned this once?

# The maximum allowable current density at a NbTi superconducting coil [A/m^2]
#       Used when calculating coil constraints
NBTI_J_MAX = 12.5e6  # A classic EUROfusion assumption

# The maximum allowable field at a Nb3Sn superconducting coil [T]
#       Used when calculating coil constraints
NB3SN_B_MAX = 13  # Eyeball

# The max allowable current density at a Nb3Sn superconducting coil [A/m^2]
#       Used when calculating coil constraints
NB3SN_J_MAX = 16.5e6  # From Simon McIntosh CS calc, 2019

# Dots per inch for GIFs
DPI_GIF = 200

# Matplotlib event plotting pause (shitty stupid API)
PLT_PAUSE = 0.001  # Completely arbitrary number of seconds?


# Minimum discretisation for finite difference grids.
#       In practice should be much higher than this
MIN_N_DISCR = 10

# Minimum discretisation of a coil [m], limit imposed to avoid massive memory
# usage
COIL_DISCR = 0.05

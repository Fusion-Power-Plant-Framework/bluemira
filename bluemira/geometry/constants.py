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
Constants for the geometry module
"""

# Absolute tolerance for equality in distances
D_TOLERANCE = 1e-5  # [m]

# Minimum length of a wire or sub-wire (edge)
MINIMUM_LENGTH = 1e-5  # [m]

# Cross product tolerance
CROSS_P_TOL = 1e-14

# Dot product tolerance
DOT_P_TOL = 1e-6

# Very big number (for large distance projection) - can't go too large because
# of clipperlib conversions
VERY_BIG = 10e4  # [m]

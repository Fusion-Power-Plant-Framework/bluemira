# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

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

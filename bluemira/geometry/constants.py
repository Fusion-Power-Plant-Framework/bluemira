# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Constants for the geometry module
"""

import numpy as np

# Absolute tolerance for equality in distances
D_TOLERANCE = 1e-5  # [m]

# Relative slack for the volume a splitter removal may move. Merging faces
# re-approximates the surfaces involved, and on a shape reassembled from
# fragments it also closes seam gaps, so small movements are legitimate: ~1e-8
# of the volume on a revolved spline vessel, ~2e-6 when it heals the seams of a
# fragment fuse. A wall that has actually moved is ~1e-3 and up.
SPLITTER_VOLUME_TOL = 1e-5

# FreeCAD's default tolerance is 1E-7, so correspondingly, we need a larger epsilon.
# The following value happens to equal 1.1920929E-7, which satisifies the requirement.
# We think that FreeCAD stores number as float 32 but we're not sure.
EPS_FREECAD = np.finfo(np.float32).eps

# Minimum length of a wire or sub-wire (edge)
MINIMUM_LENGTH = 1e-5  # [m]

# Cross product tolerance
CROSS_P_TOL = 1e-14

# Dot product tolerance
DOT_P_TOL = 1e-6

# Very big number (for large distance projection) - can't go too large because
# of clipperlib conversions
VERY_BIG = 10e4  # [m]

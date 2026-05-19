# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Type aliases and numerical tolerances shared by the CadQuery backend submodules.

Kept in its own module to avoid import cycles between ``_core`` and
``_placement`` (both reference the tolerances; ``_core`` also references the
type aliases).
"""

from __future__ import annotations

import cadquery as cq

apiVertex = cq.Vertex
apiVector = cq.Vector
apiEdge = cq.Edge
apiWire = cq.Wire
apiShell = cq.Shell
apiSolid = cq.Solid
apiShape = cq.Shape
apiCompound = cq.Compound

#: Tolerance for collapsing tiny floating-point residuals to zero (matrix
#: cleanup, axis component snapping).
_GEOM_NEAR_ZERO_TOL = 1e-12
#: Generic angular / cross-product tolerance for "is this rotation effectively
#: zero" or "are two vectors parallel" checks.
_ANGLE_PARALLEL_TOL = 1e-10
#: Generic point-coincidence / parameter-equality tolerance.
_POINT_COINCIDENCE_TOL = 1e-9
#: Default tolerance for OCC algorithms that take a precision argument
#: (``BRepClass3d``, classifiers, edge length filters, sewing).
_OCC_DEFAULT_TOL = 1e-6
#: Threshold for selecting an alternative reference axis when the natural
#: choice is too close to the input direction.
_AXIS_DOMINANCE_TOL = 0.9


__all__ = [
    "apiCompound",
    "apiEdge",
    "apiShape",
    "apiShell",
    "apiSolid",
    "apiVector",
    "apiVertex",
    "apiWire",
]

# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Base types for CAD"""

from __future__ import annotations

from enum import Enum, auto


class CurveType(Enum):
    """OCC Curve types"""

    LINE = auto()
    CIRCLE = auto()
    ELLIPSE = auto()
    HYPERBOLA = auto()
    PARABOLA = auto()
    BEZIER = auto()
    BSPLINE = auto()
    OFFSET = auto()
    OTHER = auto()

    @classmethod
    def _missing_(cls, value: object | str) -> CurveType:
        if isinstance(value, str):
            return cls[value.upper()]
        return super()._missing_(value)

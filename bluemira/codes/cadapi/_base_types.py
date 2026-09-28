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

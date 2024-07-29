# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Constants for use in the structural module.
"""

from __future__ import annotations

from enum import Enum, IntEnum, auto
from types import DynamicClassAttribute

import numpy as np
from matplotlib.pyplot import get_cmap

from bluemira.base.constants import EPS
from bluemira.structural.error import StructuralError

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

# Number of interpolation points per Element
N_INTERP = 7

# Color map for Element stresses
STRESS_COLOR = get_cmap("seismic", 1000)

# Color map for Element deflections
DEFLECT_COLOR = get_cmap("viridis", 1000)


class LoadKind(Enum):
    """Enumeration of types of loads."""

    ELEMENT_LOAD = auto()
    DISTRIBUTED_LOAD = auto()
    NODE_LOAD = auto()

    @classmethod
    def _missing_(cls, value: str) -> LoadKind:
        try:
            return cls[value.replace(" ", "_").upper()]
        except KeyError:
            raise StructuralError(
                f"{cls.__name__} has no load type {value}"
                f"please select from {(*cls._member_names_,)}"
            ) from None


class SubLoadType(Enum):
    """Enumeration of sub load types."""

    FORCE = auto()
    MOMENT = auto()
    ALL = auto()

    @classmethod
    def _missing_(cls, value: str) -> SubLoadType:
        try:
            return cls[value.upper()]
        except KeyError:
            raise StructuralError(
                f"{cls.__name__} has no load type {value}"
                f"please select from {(*cls._member_names_,)}"
            ) from None


class LoadType(IntEnum):
    """Mapping of indices to load types"""

    Fx = 0
    Fy = auto()
    Fz = auto()
    Mx = auto()
    My = auto()
    Mz = auto()

    @classmethod
    def _missing_(cls, value: str) -> LoadType:
        try:
            return cls[value.capitalize()]
        except KeyError:
            raise StructuralError(
                f"{cls.__name__} has no load type {value}"
                f"please select from {(*cls._member_names_,)}"
            ) from None

    @DynamicClassAttribute
    def vector(self):
        """Direction vector of load"""
        # indexing loop 0 1 2
        return GLOBAL_COORDS[self % 3]


class DisplacementType(IntEnum):
    """Mapping of indices to displacement/support types"""

    Dx = 0
    Dy = auto()
    Dz = auto()
    Rx = auto()
    Ry = auto()
    Rz = auto()

    @classmethod
    def _missing_(cls, value: str) -> DisplacementType:
        try:
            return cls[value.capitalize()]
        except KeyError:
            raise StructuralError(
                f"{cls.__name__} has no load type {value}"
                f"please select from {(*cls._member_names_,)}"
            ) from None

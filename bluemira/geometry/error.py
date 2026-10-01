# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Errors for geometry module
"""

from bluemira.base.error import BluemiraError


class GeometryError(BluemiraError):
    """
    Error class for use in the geometry module
    """


class NotClosedWireError(BluemiraError):
    """
    Not Closed Wire Error
    """


class MixedOrientationWireError(BluemiraError):
    """
    Mixed Orientation Wire Error
    """


class DisjointedFaceError(BluemiraError):
    """
    Disjointed Face Error
    """


class DisjointedSolidError(BluemiraError):
    """
    Disjointed Solid Error
    """


class GeometryParameterisationError(GeometryError):
    """
    Error class for parametric shapes.
    """


class OutOfBoundsError(GeometryParameterisationError):
    """A parameterisation was asked to build geometry outside its valid domain.

    Raised by ``create_shape`` when the requested variables do not describe a
    constructable shape (e.g. arc angles that sum past the symmetry limit).
    Carries ``excess``: a dimensionless, non-negative measure of how far the
    request is out of bounds, so an optimiser probing an infeasible point can
    penalise proportionally and recover rather than crash.
    """

    def __init__(self, message: str, excess: float = 1.0):
        super().__init__(message)
        self.excess = max(float(excess), 0.0)


class CoordinatesError(GeometryError):
    """
    Error class for use in Coordinates
    """

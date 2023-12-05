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


class CoordinatesError(GeometryError):
    """
    Error class for use in Coordinates
    """

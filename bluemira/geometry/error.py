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
Errors for geometry module
"""

from bluemira.base.error import BluemiraError


class GeometryError(BluemiraError):
    """
    Error class for use in the geometry module
    """

    pass


class NotClosedWire(BluemiraError):
    """
    Not Closed Wire Error
    """

    pass


class MixedOrientationWireError(BluemiraError):
    """
    Mixed Orientation Wire Error
    """

    pass


class DisjointedFace(BluemiraError):
    """
    Disjointed Face Error
    """

    pass


class DisjointedSolid(BluemiraError):
    """
    Disjointed Solid Error
    """

    pass


class GeometryParameterisationError(GeometryError):
    """
    Error class for parametric shapes.
    """

    pass


class CoordinatesError(GeometryError):
    """
    Error class for use in Coordinates
    """

    pass

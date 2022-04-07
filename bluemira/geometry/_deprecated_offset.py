# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Deprecated discretised offset operations.
"""

import numpy as np
from pyclipper import (
    ET_CLOSEDPOLYGON,
    ET_OPENROUND,
    ET_OPENSQUARE,
    JT_MITER,
    JT_ROUND,
    JT_SQUARE,
    PolyTreeToPaths,
    PyclipperOffset,
    PyPolyNode,
    scale_from_clipper,
    scale_to_clipper,
)

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.coordinates import Coordinates, rotation_matrix_v1v2
from bluemira.geometry.error import GeometryError

__all__ = ["offset_clipper"]

# =============================================================================
# Pyclipper utilities
# =============================================================================


def coordinates_to_pyclippath(coordinates):
    """
    Transforms a bluemira Coordinates object into a Path for use in pyclipper

    Parameters
    ----------
    coordinates: Coordinates
        The Coordinates to be used in pyclipper

    Returns
    -------
    path: [(x1, z1), (x2, z2), ...]
        The vertex polygon path formatting required by pyclipper
    """
    return scale_to_clipper(coordinates.xz.T)


def pyclippath_to_coordinates(path):
    """
    Transforms a pyclipper path into a bluemira Coordinates object

    Parameters
    ----------
    path: [(x1, z1), (x2, z2), ...]
        The vertex polygon path formatting used in pyclipper

    Returns
    -------
    coordinates: Coordinates
        The Coordinates from the path object
    """
    p2 = scale_from_clipper(np.array(path).T)
    return Coordinates({"x": p2[0], "z": p2[1]})


def pyclippolytree_to_coordinates(polytree):
    """
    Converts a ClipperLib PolyTree into a list of Coordinates

    Parameters
    ----------
    polytree: ClipperLib::PolyTree
        The polytree to convert to Coordinates
    """
    paths = PolyTreeToPaths(polytree)
    return [pyclippath_to_coordinates(path) for path in paths]


class PyclipperMixin:
    """
    Mixin class for typical pyclipper operations and processing
    """

    name = NotImplemented

    def perform(self):
        """
        Perform the pyclipper operation
        """
        raise NotImplementedError

    def raise_warning(self):
        """
        Raise a warning if None is to be returned.
        """
        bluemira_warn(
            f"{self.name} operation on 2-D polygons returning None.\n"
            "Nothing to perform."
        )

    def handle_solution(self, solution):
        """
        Handles the output of the Pyclipper.Execute(*) algorithms, turning them
        into Coordaintes objects. NOTE: These are closed by default.

        Parameters
        ----------
        solution: Paths
            The tuple of tuple of tuple of path vertices

        Returns
        -------
        coords: List(Coordinates)
            The list of Coordinates objects produced by the pyclipper operations
        """
        if not solution:
            self.raise_warning()
            return None
        else:
            coords = []
            if isinstance(solution, PyPolyNode):
                coords = pyclippolytree_to_coordinates(solution)
            else:
                for path in solution:
                    c = pyclippath_to_coordinates(path)
                    c.close()
                    coords.append(c)

            # Sort open coordinates by length
            return sorted(coords, key=lambda x: -x.length)

    @property
    def result(self):
        """
        The result of the Pyclipper operation.
        """
        return self._result


# =============================================================================
# Offset operations
# =============================================================================


class OffsetOperationManager(PyclipperMixin):
    """
    Abstract base class for offset operations

    Parameters
    ----------
    coordinates: Coordinates
        The Coordinates upon which to perform the offset operation
    delta: float
        The value of the offset [m]. Positive for increasing size, negative for
        decreasing
    """

    method = NotImplemented
    closed_method = ET_CLOSEDPOLYGON
    open_method = NotImplementedError

    def __init__(self, coordinates, delta):
        self.tool = PyclipperOffset()
        path = coordinates_to_pyclippath(coordinates)
        self._scale = path[0][0] / coordinates.x[0]  # Store scale

        if coordinates.closed:
            co_method = self.closed_method
        else:
            co_method = self.open_method

        self.tool.AddPath(path, self.method, co_method)
        self._result = self.perform(delta)

    def perform(self, delta):
        """
        Perform the offset operation.
        """
        delta = int(round(delta * self._scale))  # approximation
        solution = self.tool.Execute(delta)
        return self.handle_solution(solution)


class RoundOffset(OffsetOperationManager):
    """
    Offset class for rounded offsets.
    """

    name = "Round Offset"
    method = JT_ROUND
    open_method = ET_OPENROUND


class SquareOffset(OffsetOperationManager):
    """
    Offset class for squared offsets.
    """

    name = "Square Offset"
    method = JT_SQUARE
    open_method = ET_OPENSQUARE


class MiterOffset(OffsetOperationManager):
    """
    Offset class for mitered offsets.
    """

    name = "Miter Offset"
    method = JT_MITER
    open_method = ET_OPENROUND

    def __init__(self, coordinates, delta, miter_limit=2.0):
        super().__init__(coordinates, delta)

        self.tool.MiterLimit = miter_limit


def offset_clipper(coordinates: Coordinates, delta, method="square", miter_limit=2.0):
    """
    Carries out an offset operation on the Coordinates using the ClipperLib library

    Parameters
    ----------
    coordinates: Coordinates
        The Coordinates upon which to perform the offset operation
    delta: float
        The value of the offset [m]. Positive for increasing size, negative for
        decreasing
    method: str from ['square', 'round', 'miter'] (default = 'square')
        The type of offset to perform
    miter_limit: float (default = 2.0)
        The ratio of delta to used when mitering acute corners. Only used if
        method == 'miter'

    Returns
    -------
    result: Coordinates
        The offset Coordinates result
    """
    if not coordinates.is_planar:
        raise GeometryError("Cannot offset non-planar coordinates.")

    # Transform coordinates to x-z plane
    t_coordinates = transform_coordinates(coordinates, (0.0, 1.0, 0.0))

    if method == "square":
        tool = SquareOffset(t_coordinates, delta)
    elif method == "round":
        bluemira_warn("I don't know why, but this is very slow...")
        tool = RoundOffset(t_coordinates, delta)
    elif method == "miter":
        tool = MiterOffset(t_coordinates, delta, miter_limit=miter_limit)
    else:
        raise GeometryError(
            "Please choose an offset method from:\n" " round \n square \n miter"
        )

    result = tool.result[0]

    # Transform offset coordinates back to original plane
    result = transform_coordinates(result, coordinates.normal_vector)
    return result


def transform_coordinates(coordinates, direction):
    """
    Rotate coordinates to the x-z plane.
    """
    r = rotation_matrix_v1v2(coordinates.normal_vector, np.array(direction))
    x, y, z = r.T @ coordinates
    return Coordinates({"x": x, "y": y, "z": z})

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
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError

# =============================================================================
# Pyclipper utilities
# =============================================================================


def coordinates_to_pyclippath(coordinates):
    """
    Transforms a bluemira Coordinates object into a Path for use in pyclipper

    Parameters
    ----------
    loop: Coordinates
        The Coordinates to be used in pyclipper

    Returns
    -------
    path: [(x1, z1), (x2, z2), ...]
        The vertex polygon path formatting required by pyclipper
    """
    return scale_to_clipper(coordinates.xz.T)


def pyclippath_to_coordinates(path, dims=None):
    """
    Transforms a pyclipper path into a bluemira Coordinates object

    Parameters
    ----------
    path: [(x1, z1), (x2, z2), ...]
        The vertex polygon path formatting used in pyclipper
    dims: [str, str] (default = ['x', 'z'])
        The planar dimensions of the Coordinates. Will default to X, Z

    Returns
    -------
    coordinates: Coordinates
        The Coordinates from the path object
    """
    if dims is None:
        dims = ["x", "z"]
    p2 = scale_from_clipper(np.array(path).T)
    dict_ = {d: p for d, p in zip(dims, p2)}
    return Coordinates(**dict_)


def pyclippolytree_to_coordinates(polytree, dims=None):
    """
    Converts a ClipperLib PolyTree into a list of Coordinates

    Parameters
    ----------
    polytree: ClipperLib::PolyTree
        The polytree to convert to loops
    dims: None or iterable(str, str)
        The dimensions of the Loops
    """
    if dims is None:
        dims = ["x", "z"]
    paths = PolyTreeToPaths(polytree)
    return [pyclippath_to_coordinates(path, dims) for path in paths]


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
        into Loop objects. NOTE: These are closed by default.

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
                coords = pyclippolytree_to_coordinates(solution, dims=self.dims)
            else:
                for path in solution:
                    c = pyclippath_to_coordinates(path, dims=self.dims)
                    c.close()
                    coords.append(c)

        if coords[0].closed:
            return sorted(coords, key=lambda x: -x.area)
        else:
            # Sort open loops by length
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
        self.dims = coordinates.plan_dims
        self.tool = PyclipperOffset()
        path = coordinates_to_pyclippath(coordinates)
        self._scale = path[0][0] / coordinates.d2[0][0]  # Store scale

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

    def __init__(self, loop, delta, miter_limit=2.0):
        super().__init__(loop, delta)

        self.tool.MiterLimit = miter_limit


def offset_clipper(coordinates, delta, method="square", miter_limit=2.0):
    """
    Carries out an offset operation on the Loop using the ClipperLib library

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
    if method == "square":
        tool = SquareOffset(coordinates, delta)
    elif method == "round":
        bluemira_warn("I don't know why, but this is very slow...")
        tool = RoundOffset(coordinates, delta)
    elif method == "miter":
        tool = MiterOffset(coordinates, delta, miter_limit=miter_limit)
    else:
        raise GeometryError(
            "Please choose an offset method from:\n" " round \n square \n miter"
        )
    return tool.result[0]

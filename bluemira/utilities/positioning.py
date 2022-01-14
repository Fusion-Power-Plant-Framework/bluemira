# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
A collection of tools used for position interpolation.
"""

import abc

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull

from bluemira.base.constants import EPS
from bluemira.geometry._deprecated_tools import vector_lengthnorm_2d
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import slice_shape
from bluemira.utilities.error import PositionerError


class XZGeometryInterpolator(abc.ABC):
    """
    Abstract base class for 2-D x-z geometry interpolation to normalised [0, 1] space.

    By convention, normalised x-z space is oriented counter-clockwise w.r.t. [0, 1, 0].

    Parameters
    ----------
    geometry: BluemiraWire
        Geometry to interpolate with
    """

    def __init__(self, geometry):
        self.geometry = geometry

    def _get_xz_coordinates(self):
        """
        Get discretised x-z coordinates of the geometry.
        """
        coordinates = self.geometry.discretize(
            byedges=True, dl=self.geometry.length / 1000
        )
        coordinates.set_ccw([0, 1, 0])
        return coordinates.xz

    @abc.abstractmethod
    def to_xz(self, l_value):
        """
        Convert parametric-space 'L' values to physical x-z space.
        """
        pass

    @abc.abstractmethod
    def to_L(self, x, z):
        """
        Convert physical x-z space values to parametric-space 'L' values.
        """
        pass


class PathInterpolator(XZGeometryInterpolator):
    """
    Sets up an x-z path for a point to move along.

    The path is treated as flat in the x-z plane.

    Parameters
    ----------
    geometry: BluemiraWire
        Path to interpolate along
    """

    def __init__(self, geometry):
        super().__init__(geometry)
        x, z = self._get_xz_coordinates()
        ln = vector_lengthnorm_2d(x, z)
        self.x_ius = InterpolatedUnivariateSpline(ln, x)
        self.z_ius = InterpolatedUnivariateSpline(ln, z)

    @staticmethod
    def _f_min(l_value, f_x, f_z, x, z):
        dx = f_x(l_value) - x
        dz = f_z(l_value) - z
        return dx ** 2 + dz ** 2

    def to_xz(self, l_value):
        """
        Convert parametric-space 'L' values to physical x-z space.
        """
        l_value = np.clip(l_value, 0.0, 1.0)
        return float(self.x_ius(l_value)), float(self.z_ius(l_value))

    def to_L(self, x, z):
        """
        Convert physical x-z space values to parametric-space 'L' values.
        """
        return minimize_scalar(
            self._f_min,
            args=(self.x_ius, self.z_ius, x, z),
            bounds=[0, 1],
            method="bounded",
            options={"xatol": 1e-8},
        ).x


class RegionInterpolator(XZGeometryInterpolator):
    """
    Sets up an x-z region for a point to move within.

    The region is treated as a flat x-z surface.

    The normalisation occurs by cutting the shape in two axes and
    normalising over the cut length within the region.

    Currently this is limited to convex polygons.

    Generalisation to all polygons is possible but unimplemented
    and possibly quite slow when converting from normalised to real coordinates.

    When the point position provided is outside the given region the point will
    be moved to the closest edge of the region.

    The mapping from outside to the edge of the region is not strictly defined.
    The only certainty is that the point will be moved into the region.

    Parameters
    ----------
    geometry: BluemiraWire
        Region to interpolate within
    """

    def __init__(self, geometry):
        super().__init__(geometry)
        self._check_geometry_feasibility(geometry)
        self.z_min = geometry.bounding_box.z_min
        self.z_max = geometry.bounding_box.z_max

    def _check_geometry_feasibility(self, geometry):
        """
        Checks the provided region is convex.

        This is a current limitation of RegionInterpolator
        not providing a 'smooth' interpolation surface.

        Parameters
        ----------
        geometry: BluemiraWire
            Region to check

        Raises
        ------
        PositionerError
            When geometry is not a convex
        """
        if not self.geometry.is_closed:
            raise PositionerError("RegionInterpolator can only handle closed wires.")

        xz_coordinates = self._get_xz_coordinates()
        hull = ConvexHull(xz_coordinates.T)
        # Yes, the "area" of a 2-D scipy ConvexHull is its perimeter...
        if not np.allclose(hull.area, geometry.length, atol=EPS):
            raise PositionerError(
                "RegionInterpolator can only handle convex geometries. Perimeter "
                f"difference between convex hull and geometry: {hull.volume - geometry.area}"
            )

    def to_xz(self, l_values):
        """
        Convert parametric-space 'L' values to physical x-z space.

        Parameters
        ----------
        l_values: Tuple[float, float]
            Coordinates in normalised space

        Returns
        -------
        x: float
            x coordinate in real space
        z: float
            z coordinate in real space

        Raises
        ------
        GeometryError
            When loop is not a Convex Hull

        """
        l_0, l_1 = l_values
        z = self.z_min + (self.z_max - self.z_min) * l_1

        plane = BluemiraPlane.from_3_points([0, 0, z], [1, 0, z], [0, 1, z])

        intersect = slice_shape(self.geometry, plane)
        if len(intersect) == 1:
            x = intersect[0][0]
        elif len(intersect) == 2:
            x_min, x_max = sorted([intersect[0][0], intersect[1][0]])
            x = x_min + (x_max - x_min) * l_0
        else:
            raise PositionerError(
                "Unexpected number of intersections in x-z conversion."
            )

        return x, z

    def to_L(self, x, z):
        """
        Convert physical x-z space values to parametric-space 'L' values.

        Parameters
        ----------
        x: float
            x coordinate in real space
        z: float
            z coordinate in real space

        Returns
        -------
        l_values: Tuple[float, float]
            Coordinates in normalised space

        Raises
        ------
        GeometryError
            When loop is not a Convex Hull

        """
        l_1 = (z - self.z_min) / (self.z_max - self.z_min)
        l_1 = np.clip(l_1, 0.0, 1.0)

        plane = BluemiraPlane.from_3_points([x, 0, z], [x + 1, 0, z], [x, 1, z])
        intersect = slice_shape(self.geometry, plane)

        return self._intersect_filter(x, l_1, intersect)

    def _intersect_filter(self, x, l_1, intersect):
        """
        Checks where points are based on number of intersections
        with a plane. Should initially be called with a plane involving z.

        No intersection could mean above 1 edge therefore a plane in xy
        is checked before recalling this function.
        If there is one intersection point we are on an edge (either bottom or top),
        if there is two intersection points we are in the region,
        otherwise the region is not a convex hull.

        Parameters
        ----------
        x: float
            x coordinate
        l_1: float
            Normalised z coordinate
        intersect: Plane
            A plane through xz

        Returns
        -------
        l_values: Tuple[float, float]
            Coordinates in normalised space

        Raises
        ------
        PositionerError
            When geometry is not a convex
        """
        if intersect is None:
            plane = BluemiraPlane.from_3_points([x, 0, 0], [x + 1, 0, 0], [x, 1, 0])
            intersect = slice_shape(self.geometry, plane)
            l_0, l_1 = self._intersect_filter(
                x, l_1, [False] if intersect is None else intersect
            )
        elif len(intersect) == 2:
            x_min, x_max = sorted([intersect[0][0], intersect[1][0]])
            l_0 = np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)
        elif len(intersect) == 1:
            l_0 = float(l_1 == 1.0)
        else:
            raise PositionerError("Unexpected number of intersections in L conversion.")
        return l_0, l_1


class PositionMapper:
    """
    Positioning tool for use in optimisation

    Parameters
    ----------
    interpolators: List[XZGeometryInterpolator]
        The ordered list of geometry interpolators
    """

    def __init__(self, interpolators):
        self.interpolators = interpolators

    def _check_length(self, thing):
        """
        Check that something is the same length as the number of available interpolators.
        """
        if len(thing) != len(self.interpolators):
            raise PositionerError(
                f"Object of length: {len(thing)} not of length {len(self.interpolators)}"
            )

    def to_xz(self, l_values):
        """
        Convert a set of parametric-space values to physical x-z coordinates.

        Parameters
        ----------
        l_values: Union[List[float],
                        List[Tuple[float]],
                        List[Union[float,
                        Tuple[float]]]]

            The set of parametric-space values to convert

        Returns
        -------
        x: np.ndarray
            Array of x coordinates
        z: np.ndarray
            Array of z coordinates
        """
        self._check_length(l_values)
        return np.array(
            [tool.to_xz(l_values[i]) for i, tool, in enumerate(self.interpolators)]
        ).T

    def to_L(self, x, z):
        """
        Convert a set of physical x-z coordinates to parametric-space values.

        Parameters
        ----------
        x: Iterable
            The x coordinates to convert
        z: Iterable
            The z coordinates to convert

        Returns
        -------
        l_values: Union[List[float],
                        List[Tuple[float]],
                        List[Union[float,
                        Tuple[float]]]]

            The set of parametric-space values
        """
        self._check_length(x)
        self._check_length(z)
        return [tool.to_L(x[i], z[i]) for i, tool in enumerate(self.interpolators)]


class ZLineDivider:
    """
    Sets up a vertical line along which multiple points can move, such that their
    inscribed circles fill the length whilst touching. The circles can be separated by
    a gap.

    By convention, the first point is at the top of the line.

    Parameters
    ----------
    z_min: float
        Minimum vertical coordinate of the line
    z_max: float
        Maximum vertical coordinate of the line
    n_divisions: int
        Number of divisions along the vertical line
    z_gap: float
        Vertical gap between the inscribed circles
    """

    def __init__(self, z_min, z_max, n_divisions, z_gap=0.0):
        if z_max < z_min:
            z_min, z_max = z_max, z_min

        if n_divisions <= 1:
            raise PositionerError(
                "No point making a ZLineDivider with 1 or fewer divisions..."
            )

        self.n_divisions = n_divisions
        self.z_min = z_min
        self.z_max = z_max
        self.z_gap = z_gap
        z = [z_max, z_min]
        self.z_interpolator = interp1d([0, 1], z)
        self.l_interpolator = interp1d(z, [0, 1])

    def _check_length(self, thing):
        """
        Check that something is the same length as the number of divisions.
        """
        if len(thing) != self.n_divisions:
            raise PositionerError(
                f"Object of length: {len(thing)} not of length {self.n_divisions}"
            )

    def to_zdz(self, l_values):
        """
        Convert parametric-space 'L' values to physical z-dz space.
        """
        self._check_length(l_values)

        l_values = np.clip(l_values, 0, 1)
        l_values = np.sort(l_values)
        z_edge = self.z_interpolator(l_values)
        dz, zc = np.zeros(len(l_values)), np.zeros(len(l_values))
        dz[0] = 0.5 * abs(self.z_max - z_edge[0])
        zc[0] = self.z_max - dz[0]
        for i in range(1, len(l_values)):
            dz[i] = 0.5 * abs(z_edge[i - 1] - z_edge[i] - self.z_gap)
            zc[i] = z_edge[i - 1] - dz[i] - self.z_gap
        return zc[::-1], dz[::-1]

    def to_L(self, zc_values):
        """
        Convert physical z-dz space values to parametric-space 'L' values.
        """
        self._check_length(zc_values)

        zc_values = np.sort(zc_values)[::-1]
        z_edge = np.zeros(len(zc_values))
        z_edge[0] = self.z_max - 2 * abs(self.z_max - zc_values[0])
        for i in range(1, len(zc_values) - 1):
            z_edge[i] = zc_values[i] - (z_edge[i - 1] - zc_values[i] - self.z_gap)
        z_edge[len(zc_values) - 1] = self.z_min
        return self.l_interpolator(z_edge)

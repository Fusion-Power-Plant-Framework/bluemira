# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
A collection of geometry tools.
"""

import numpy as np
import numba as nb
from numba.np.extensions import cross2d

EPS = np.finfo(np.float).eps  # from bluemira.base.constants import EPS
from bluemira.geometry.base import GeometryError
from bluemira.geometry.constants import CROSS_P_TOL, DOT_P_TOL

# =============================================================================
# Boolean checks
# =============================================================================


@nb.jit(cache=True, nopython=True)
def check_linesegment(point_a, point_b, point_c):
    """
    Check that point C is on the line between points A and B.

    Parameters
    ----------
    point_a: np.array(2)
        The first line segment point
    point_b: np.array(2)
        The second line segment point
    point_c: np.array(2)
        The point which to check is on A--B

    Returns
    -------
    check: bool
        True: if C on A--B, else False
    """
    a_c, a_b = point_c - point_a, point_b - point_a
    distance = np.sqrt(np.sum(a_b ** 2))
    # Numba doesn't like doing cross-products of things with size 2
    cross = cross2d(a_b, a_c)
    if np.abs(cross) > CROSS_P_TOL * distance:
        return False
    k_ac = np.dot(a_b, a_c)
    k_ab = np.dot(a_b, a_b)
    if k_ac < 0:
        return False
    elif k_ac > k_ab:
        return False
    else:
        return True


@nb.jit(cache=True, nopython=True)
def in_polygon(x, z, poly, include_edges=False):
    """
    Determine if a point (x, z) is inside a 2-D polygon.

    Parameters
    ----------
    x, z: float, float
        Point coordinates
    poly: np.array(2, N)
        The array of polygon point coordinates
    include_edges: bool
        Whether or not to return True if a point is on the perimeter of the
        polygon

    Returns
    -------
    inside: bool
        Whether or not the point is in the polygon
    """
    n = len(poly)
    inside = False
    x1, z1, x_inter = 0, 0, 0
    x0, z0 = poly[0]
    for i in range(n + 1):
        x1, z1 = poly[i % n]

        if x == x1 and z == z1:
            return include_edges

        if z > min(z0, z1):
            if z <= max(z0, z1):
                if x <= max(x0, x1):

                    if z0 != z1:
                        x_inter = (z - z0) * (x1 - x0) / (z1 - z0) + x0
                        if x == x_inter:
                            return include_edges
                    if x0 == x1 or x <= x_inter:

                        inside = not inside  # Beautiful
        elif z == min(z0, z1):
            if z0 == z1:
                if (x <= max(x0, x1)) and (x >= min(x0, x1)):
                    return include_edges

        x0, z0 = x1, z1
    return inside


@nb.jit(cache=True, nopython=True)
def polygon_in_polygon(poly1, poly2, include_edges=False):
    """
    Determine what points of a polygon are inside another polygon.

    Parameters
    ----------
    poly1: np.array(2, N1)
        The array of polygon1 point coordinates
    poly2: np.array(2, N2)
        The array of polygon2 point coordinates
    include_edges: bool
        Whether or not to return True if a point is on the perimeter of the
        polygon

    Returns
    -------
    inside: np.array(N1, dtype=bool)
        The array of boolean values per index of polygon1
    """
    inside_array = np.empty(len(poly1), dtype=np.bool_)
    for i in range(len(poly1)):
        inside_array[i] = in_polygon(
            poly1[i][0], poly1[i][1], poly2, include_edges=include_edges
        )
    return inside_array


@nb.jit(cache=True, forceobj=True)
def on_polygon(x, z, poly):
    """
    Determine if a point (x, z) is on the perimeter of a closed 2-D polygon.

    Parameters
    ----------
    x, z: float, float
        Point coordinates
    poly: np.array(2, N)
        The array of polygon point coordinates

    Returns
    -------
    on_edge: bool
        Whether or not the point is on the perimeter of the polygon
    """
    on_edge = False
    for i, (point_a, point_b) in enumerate(zip(poly[:-1], poly[1:])):
        c = check_linesegment(np.array(point_a), np.array(point_b), np.array([x, z]))

        if c is True:
            return True
    return on_edge


@nb.jit(cache=True, nopython=True)
def check_ccw(x, z):
    """
    Check that a set of x, z coordinates are counter-clockwise.

    Parameters
    ----------
    x, z: 1-D np.array, 1-D np.array
        Coordinates of polygon

    Returns
    -------
    ccw: bool
        True if polygon counterclockwise
    """
    a = 0
    for n in range(len(x) - 1):
        a += (x[n + 1] - x[n]) * (z[n + 1] + z[n])
    return a < 0


# =============================================================================
# Coordinate analysis
# =============================================================================


def distance_between_points(p1, p2):
    """
    Calculates the distance between two points

    Parameters
    ----------
    p1: (float, float)
        The coordinates of the first point
    p2: (float, float)
        The coordinates of the second point

    Returns
    -------
    d: float
        The distance between the two points [m]
    """
    if len(p1) != len(p2):
        raise GeometryError("Need two points of the same number of coordinates.")

    if (len(p1) not in [2, 3]) or (len(p2) not in [2, 3]):
        raise GeometryError("Need 2- or 3-D sized points.")

    return np.sqrt(sum([(p2[i] - p1[i]) ** 2 for i in range(len(p2))]))


def get_normal_vector(x, y, z):
    """
    Calculate the normal vector from a series of planar points.

    Parameters
    ----------
    x: np.array
        The x coordinates
    y: np.array
        The y coordinates
    z: np.array
        The z coordinates

    Returns
    -------
    n_hat: np.array(3)
        The normalised normal vector
    """
    if len(x) < 3:
        raise GeometryError(
            "Cannot get a normal vector for a set of points with" "length less than 3."
        )
    if (len(x) != len(y)) or (len(x) != len(z)) or (len(y) != len(z)):
        raise GeometryError("Point coordinate vectors must be of equal length.")

    p1 = np.array([x[0], y[0], z[0]])
    p2 = np.array([x[1], y[1], z[1]])
    v1 = p2 - p1

    # Force length 3 vectors to access index 2 without raising IndexErrors elsewhere
    i_max = max(3, len(x) - 1)
    for i in range(2, i_max):
        p3 = np.array([x[i], y[i], z[i]])
        v2 = p3 - p2
        if np.allclose(v2, 0):
            v2 = p3 - p1
            if np.allclose(v2, 0):
                continue
        n_hat = np.cross(v1, v2)
        if not np.allclose(n_hat, 0):
            break
    else:
        raise GeometryError("Unable to find a normal vector from set of points.")

    return n_hat / np.linalg.norm(n_hat)


def bounding_box(x, y, z):
    """
    Calculates a bounding box for a set of 3-D coordinates

    Parameters
    ----------
    x: np.array(N)
        The x coordinates
    y: np.array(N)
        The y coordinates
    z: np.array(N)
        The z coordinates

    Returns
    -------
    x_b: np.array(8)
        The x coordinates of the bounding box rectangular cuboid
    y_b: np.array(8)
        The y coordinates of the bounding box rectangular cuboid
    z_b: np.array(8)
        The z coordinates of the bounding box rectangular cuboid
    """
    xmax, xmin = np.max(x), np.min(x)
    ymax, ymin = np.max(y), np.min(y)
    zmax, zmin = np.max(z), np.min(z)

    size = max([xmax - xmin, ymax - ymin, zmax - zmin])

    x_b = 0.5 * size * np.array([-1, -1, -1, -1, 1, 1, 1, 1]) + 0.5 * (xmax + xmin)
    y_b = 0.5 * size * np.array([-1, -1, 1, 1, -1, -1, 1, 1]) + 0.5 * (ymax + ymin)
    z_b = 0.5 * size * np.array([-1, 1, -1, 1, -1, 1, -1, 1]) + 0.5 * (zmax + zmin)
    return x_b, y_b, z_b


# =============================================================================
# Coordinate manipulation
# =============================================================================


def close_coordinates(x, y, z):
    """
    Close an ordered set of coordinates.

    Parameters
    ----------
    x: np.array
        The x coordinates
    y: np.array
        The y coordinates
    z: np.array
        The z coordinates

    Returns
    -------
    x: np.array
        The closed x coordinates
    y: np.array
        The closed y coordinates
    z: np.array
        The closed z coordinates
    """
    if distance_between_points([x[0], y[0], z[0]], [x[-1], y[-1], z[-1]]) > EPS:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])
    return x, y, z


# =============================================================================
# Intersection tools
# =============================================================================


def loop_plane_intersect(loop, plane):
    """
    Calculate the intersection of a loop with a plane.

    Parameters
    ----------
    loop: Loop
        The loop to calculate the intersection on
    plane: Plane
        The plane to calculate the intersection with

    Returns
    -------
    inter: np.array(3, n_intersections) or None
        The xyz coordinates of the intersections with the loop. Returns None if
        there are no intersections detected
    """
    out = _loop_plane_intersect(loop.xyz.T[:-1], plane.p1, plane.n_hat)
    if not out:
        return None
    else:
        return np.unique(out, axis=0)  # Drop occasional duplicates


@nb.jit(cache=True, nopython=True)
def _loop_plane_intersect(array, p1, vec2):
    # JIT compiled utility of the above
    out = []
    for i in range(len(array)):
        vec1 = array[i + 1] - array[i]
        dot = np.dot(vec1, vec2)
        if abs(dot) > DOT_P_TOL:
            w = array[i] - p1
            fac = -(np.dot(vec2, w)) / dot
            if (fac >= 0) and (fac <= 1):
                out.append(array[i] + fac * vec1)
    return out

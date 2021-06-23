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
from pyquaternion import Quaternion
from bluemira.base.constants import EPS
from bluemira.geometry.constants import CROSS_P_TOL, DOT_P_TOL
from bluemira.geometry.error import GeometryError


# =============================================================================
# Pre-processing utilities
# =============================================================================


def xyz_process(func):
    """
    Decorator for parsing x, y, z coordinates to numpy float arrays and dimension
    checking.
    """

    def wrapper(x, y, z=None):
        x = np.ascontiguousarray(x, dtype=np.float_)
        y = np.ascontiguousarray(y, dtype=np.float_)
        if z is None:
            if len(x) != len(y):
                raise GeometryError("Coordinate vectors must have same length.")
            return func(x, y, z)
        else:
            z = np.ascontiguousarray(z, dtype=np.float_)

            if not (len(x) == len(y) == len(z)):
                raise GeometryError("Coordinate vectors must have same length.")

            return func(x, y, z)

    return wrapper


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
    x: np.array
        The x coordinates of the polygon
    z: np.array
        The z coordinates of the polygon

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


def get_angle_between_vectors(v1, v2, signed=False):
    """
    Angle between vectors. Will return the signed angle if specified.

    Parameters
    ----------
    v1: np.array
        The first vector
    v2: np.array
        The second vector

    Returns
    -------
    angle: float
        The angle between the vector [radians]
    """
    if not all(isinstance(p, np.ndarray) for p in [v1, v2]):
        v1, v2 = np.array(v1), np.array(v2)
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    cos_angle = np.dot(v1n, v2n)
    # clip to dodge a NaN
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    sign = 1
    if signed:
        det = np.linalg.det(np.stack((v1n[-2:], v2n[-2:])))
        if det == 0:
            # Vectors parallel
            sign = 1
        else:
            sign = np.sign(det)

    return sign * angle


@nb.jit(cache=True, nopython=True)
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

    if not (len(x) == len(y) == len(z)):
        raise GeometryError("Point coordinate vectors must be of equal length.")

    n_hat = np.array([0.0, 0.0, 0.0])  # Force numba to type to floats
    p1 = np.array([x[0], y[0], z[0]])
    p2 = np.array([x[1], y[1], z[1]])
    v1 = p2 - p1

    # Force length 3 vectors to access index 2 without raising IndexErrors elsewhere
    i_max = max(3, len(x) - 1)
    for i in range(2, i_max):
        p3 = np.array([x[i], y[i], z[i]])
        v2 = p3 - p2

        if np.all(np.abs(v2) < EPS):  # np.allclose not available in numba
            v2 = p3 - p1
            if np.all(np.abs(v2) < EPS):
                continue

        n_hat[:] = np.cross(v1, v2)

        if not np.all(np.abs(n_hat) < EPS):
            break
    else:
        raise GeometryError("Unable to find a normal vector from set of points.")

    return n_hat / np.linalg.norm(n_hat)


@xyz_process
def get_perimeter(x, y, z=None):
    """
    Calculate the perimeter of a set of coordinates.

    Parameters
    ----------
    x: np.array
        The x coordinates
    y: np.array
        The y coordinates
    z: Union[None, np.array]
        The z coordinates

    Returns
    -------
    perimeter: float
        The perimeter of the coordinates
    """
    if z is None:
        return get_perimeter_2d(x, y)
    else:
        return get_perimeter_3d(x, y, z)


@nb.jit(cache=True, nopython=True)
def get_perimeter_2d(x, y):
    """
    Calculate the perimeter of a 2-D set of coordinates.

    Parameters
    ----------
    x: np.array
        The x coordinates
    y: np.array
        The y coordinates

    Returns
    -------
    perimeter: float
        The perimeter of the coordinates
    """
    return np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))


@nb.jit(cache=True, nopython=True)
def get_perimeter_3d(x, y, z):
    """
    Calculate the perimeter of a set of 3-D coordinates.

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
    perimeter: float
        The perimeter of the coordinates
    """
    return np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2))


@xyz_process
def get_area(x, y, z=None):
    """
    Calculate the area inside a closed polygon with x, y coordinate vectors.
    `Link Shoelace method <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    Parameters
    ----------
    x: np.array
        The first set of coordinates [m]
    y: np.array
        The second set of coordinates [m]
    z: Union[np.array, None]
        The third set of coordinates or None (for a 2-D polygon)

    Returns
    -------
    area: float
        The area of the polygon [m^2]
    """
    if z is None:
        return get_area_2d(x, y)
    else:
        return get_area_3d(x, y, z)


@nb.jit(cache=True, nopython=True)
def get_area_2d(x, y):
    """
    Calculate the area inside a closed polygon with x, y coordinate vectors.
    `Link Shoelace method <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    Parameters
    ----------
    x: np.array
        The first set of coordinates [m]
    y: np.array
        The second set of coordinates [m]

    Returns
    -------
    area: float
        The area of the polygon [m^2]
    """
    # No np.roll in numba
    x1 = np.append(x[-1], x[:-1])
    y1 = np.append(y[-1], y[:-1])
    return 0.5 * np.abs(np.dot(x, y1) - np.dot(y, x1))


@nb.jit(cache=True, nopython=True)
def get_area_3d(x, y, z):
    """
    Calculate the area inside a closed polygon with x, y coordinate vectors.
    `Link Shoelace method <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    Parameters
    ----------
    x: np.array
        The first set of coordinates [m]
    y: np.array
        The second set of coordinates [m]
    z: np.array
        The third set of coordinates or None (for a 2-D polygon)

    Returns
    -------
    area: float
        The area of the polygon [m^2]
    """
    v3 = get_normal_vector(x, y, z)
    m = np.zeros((3, len(x)))
    m[0, :] = x
    m[1, :] = y
    m[2, :] = z
    a = np.array([0.0, 0.0, 0.0])
    for i in range(len(z)):
        a += np.cross(m[:, i], m[:, (i + 1) % len(z)])
    a *= 0.5
    return abs(np.dot(a, v3))


@xyz_process
def get_centroid(x, y, z=None):
    """
    Calculate the centroid of a non-self-intersecting 2-D counter-clockwise polygon.

    Parameters
    ----------
    x: np.array
        x coordinates of the loop to calculate on
    y: np.array
        y coordinates of the loop to calculate on
    z: Union[None, np.array]

    Returns
    -------
    centroid: np.array
        The x, y, [z] coordinates of the centroid [m]
    """
    if z is None:
        return get_centroid_2d(x, y)
    else:
        return get_centroid_3d(x, y, z)


@nb.jit(cache=True, nopython=True)
def get_centroid_2d(x, z):
    """
    Calculate the centroid of a non-self-intersecting 2-D counter-clockwise polygon.

    Parameters
    ----------
    x: np.array
        x coordinates of the loop to calculate on
    z: np.array
        z coordinates of the loop to calculate on

    Returns
    -------
    centroid: List[float]
        The x, z coordinates of the centroid [m]
    """
    if not check_ccw(x, z):
        x = x[::-1]
        z = z[::-1]
    area = get_area_2d(x, z)

    cx, cz = 0, 0
    for i in range(len(x) - 1):
        a = x[i] * z[i + 1] - x[i + 1] * z[i]
        cx += (x[i] + x[i + 1]) * a
        cz += (z[i] + z[i + 1]) * a

    if area != 0:
        # Zero division protection
        cx /= 6 * area
        cz /= 6 * area

    return [cx, cz]


def get_centroid_3d(x, y, z):
    """
    Calculate the centroid of a non-self-intersecting counterclockwise polygon
    in 3-D.

    Parameters
    ----------
    x: Iterable
        The x coordinates
    y: Iterable
        The y coordinates
    z: Iterable
        The z coordinates

    Returns
    -------
    centroid: List[float]
        The x, y, z coordinates of the centroid [m]
    """
    cx, cy = get_centroid_2d(x, y)
    cx2, cz = get_centroid_2d(x, z)
    cy2, cz2 = get_centroid_2d(y, z)

    # The following is an "elegant" but computationally more expensive way of
    # dealing with the 0-area edge cases
    # (of which there are more than you think)
    cx = np.array([cx, cx2])
    cy = np.array([cy, cy2])
    cz = np.array([cz, cz2])

    def get_rational(i, array):
        """
        Gets rid of infinity and nan coordinates
        """
        args = np.argwhere(np.isfinite(array))
        if len(args) == 0:
            # 2-D shape with a simple axis offset
            # Get the first value of the coordinate set which is equal to the
            # offset
            return [x, y, z][i][0]
        elif len(args) == 1:
            return array[args[0][0]]
        else:
            if not np.isclose(array[0], array[1]):
                # Occasionally the two c values are not the same, and one is 0
                # Take non-trivial value (this works in the case of 2 zeros)
                return array[np.argmax(np.abs(array))]
            else:
                return array[0]

    return [get_rational(i, c) for i, c in enumerate([cx, cy, cz])]


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


def vector_lengthnorm(x, y, z):
    """
    Get a normalised 1-D parameterisation of a set of x-y-z coordinates.

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
    length_: np.array(n)
        The normalised length vector
    """
    length_ = np.append(
        0,
        np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)),
    )
    return length_ / length_[-1]


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


def side_vector(polygon_array):
    """
    Calculates the side vectors of an anti-clockwise polygon

    Parameters
    ----------
    polygon_array: np.array(2, N)
        The array of polygon point coordinates

    Returns
    -------
    sides: np.array(N, 2)
        The array of the polygon side vectors
    """
    return polygon_array - np.roll(polygon_array, 1)


def normal_vector(side_vectors):
    """
    Anti-clockwise

    Parameters
    ----------
    side_vectors: np.array(N, 2)
        The side vectors of a polygon

    Returns
    -------
    a: np.array(2, N)
        The array of 2-D normal vectors of each side of a polygon
    """
    a = -np.array([-side_vectors[1], side_vectors[0]]) / np.sqrt(
        side_vectors[0] ** 2 + side_vectors[1] ** 2
    )
    nan = np.isnan(a)
    a[nan] = 0
    return a


def offset(x, z, offset_value):
    """
    Get a square-based offset of the coordinates (no splines). N-sized output

    Parameters
    ----------
    x: np.array
        The x coordinate vector
    z: np.array
        The x coordinate vector
    offset_value: float
        The offset value [m]

    Returns
    -------
    xo: np.array(N)
        The x offset coordinates
    zo: np.array(N)
        The z offset coordinates
    """
    # check numpy arrays:
    x, z = np.array(x), np.array(z)
    # check closed:
    if (x[-2:] == x[:2]).all() and (z[-2:] == z[:2]).all():
        closed = True
    elif x[0] == x[-1] and z[0] == z[-1]:
        closed = True
        # Need to "double lock" it for closed curves
        x = np.append(x, x[1])
        z = np.append(z, z[1])
    else:
        closed = False
    p = np.array([np.array(x), np.array(z)])
    # Normal vectors for each side
    v = normal_vector(side_vector(p))
    # Construct points offset
    off_p = np.column_stack(p + offset_value * v)
    off_p2 = np.column_stack(np.roll(p, 1) + offset_value * v)
    off_p = np.array([off_p[:, 0], off_p[:, 1]])
    off_p2 = np.array([off_p2[:, 0], off_p2[:, 1]])
    ox = np.empty((off_p2[0].size + off_p2[0].size,))
    oz = np.empty((off_p2[1].size + off_p2[1].size,))
    ox[0::2], ox[1::2] = off_p2[0], off_p[0]
    oz[0::2], oz[1::2] = off_p2[1], off_p[1]
    off_s = np.array([ox[2:], oz[2:]]).T
    pnts = []
    for i in range(len(off_s[:, 0]) - 2)[0::2]:
        pnts.append(vector_intersect(off_s[i], off_s[i + 1], off_s[i + 3], off_s[i + 2]))
    pnts.append(pnts[0])
    pnts = np.array(pnts)[:-1][::-1]  # sorted ccw nicely
    if closed:
        pnts = np.concatenate((pnts, [pnts[0]]))  # Closed
    else:  # Add end points
        pnts = np.concatenate((pnts, [off_s[0]]))
        pnts = np.concatenate(([off_s[-1]], pnts))
    # sorted ccw nicely - i know looks weird but.. leave us kids alone
    # drop nan values
    return pnts[~np.isnan(pnts).any(axis=1)][::-1].T


# =============================================================================
# Rotations
# =============================================================================


def quart_rotate(point, **kwargs):
    """
    Rotate a point cloud by angle theta around vector (right-hand coordinates).
    Uses black quarternion magic.

    Parameters
    ----------
    point: Union[np.array(n, 3), dict('x': [], 'y': [], 'z': [])]
        The coordinates of the points to be rotated
    kwargs:
        theta: float
            Rotation angle [radians]
        p1: [float, float, float]
            Origin of rotation vector
        p2: [float, float, float]
            Second point defining rotation axis
    kwargs: (alternatively)
        theta: float
            Rotation angle [radians]
        xo: [float, float, float]
            Origin of rotation vector
        dx: [float, float, float] or one of 'x', 'y', 'z'
            Direction vector definition rotation axis from origin. If a string
            is specified the dx vector is automatically calculated, e.g.
            'z': (0, 0, 1)
    kwargs: (alternatively)
        quart: Quarternion object
            The rotation quarternion
        xo: [float, float, float]
            Origin of rotation vector

    Returns
    -------
    rpoint: Union[np.array(n, 3), dict('x': [], 'y': [], 'z': [])]
        The rotated coordinates. Output in numpy array or dict, depending on
        input type
    """
    if "quart" in kwargs:
        quart = kwargs["quart"]
        xo = kwargs.get("xo", np.zeros(3))
    else:
        theta = kwargs["theta"]
        if "p1" in kwargs and "p2" in kwargs:
            p1, p2 = kwargs["p1"], kwargs["p2"]
            if not isinstance(p1, np.ndarray) or not isinstance(p1, np.ndarray):
                p1, p2 = np.array(p1), np.array(p2)
            xo = p1
            dx = p2 - p1
            dx = tuple(dx)
        elif "xo" in kwargs and "dx" in kwargs:
            xo, dx = kwargs["xo"], kwargs["dx"]
        elif "dx" in kwargs:
            dx = kwargs["dx"]
            if isinstance(dx, str):
                index = ["x", "y", "z"].index(dx)
                dx = np.zeros(3)
                dx[index] = 1
            xo = np.zeros(3)
        else:
            errtxt = "error in kwargs input\n"
            errtxt += "rotation vector input as ether:\n"
            errtxt += "\tpair of points, p1=[x,y,z] and p2=[x,y,z]\n"
            errtxt += "\torigin and vector, xo=[x,y,z] and dx=[x,y,z]\n"
            raise GeometryError(errtxt)

        dx /= np.linalg.norm(dx)  # normalise rotation axis
        quart = Quaternion(axis=dx, angle=theta)

    if isinstance(point, dict):
        isdict = True
        p = np.zeros((len(point["x"]), 3))
        for i, var in enumerate(["x", "y", "z"]):
            p[:, i] = point[var]
        point = p
    else:
        isdict = False
    if np.ndim(point) == 1 and len(point) == 3:
        point = np.array([point])
    if np.shape(point)[1] != 3:
        errtxt = "point vector required as numpy.array size=(:,3)"
        raise GeometryError(errtxt)

    trans = np.ones((len(point), 1)) * xo  # expand vector origin
    p = point - trans  # translate to rotation vector's origin (xo)
    rpoint = np.zeros(np.shape(point))
    for i, po in enumerate(p):
        rpoint[i, :] = quart.rotate(po)
    rpoint += trans  # translate from rotation vector's origion (xo)

    if isdict:  # return to dict
        p = {}
        for i, var in enumerate(["x", "y", "z"]):
            p[var] = rpoint[:, i]
        rpoint = p
    return rpoint


def rotation_matrix(theta, axis="z"):
    """
    Old-fashioned rotation matrix: :math:`\\mathbf{R_{u}}(\\theta)`
    \t:math:`\\mathbf{x^{'}}=\\mathbf{R_{u}}(\\theta)\\mathbf{x}`

    \t:math:`\\mathbf{R_{u}}(\\theta)=cos(\\theta)\\mathbf{I}+sin(\\theta)[\\mathbf{u}]_{\\times}(1-cos(\\theta))(\\mathbf{u}\\otimes\\mathbf{u})`

    Parameters
    ----------
    theta: float
        The rotation angle [radians] (counter-clockwise about axis!)
    axis: Union[str, iterable(3)]
        The rotation axis (specified by axis label or vector)

    Returns
    -------
    r_matrix: np.array((3, 3))
        The (active) rotation matrix about the axis for an angle theta
    """
    if isinstance(axis, str):
        # I'm leaving all this in here, because it is easier to understand
        # what is going on, and that these are just "normal" rotation matrices
        if axis == "z":
            r_matrix = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
        elif axis == "y":
            r_matrix = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
            )
        elif axis == "x":
            r_matrix = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)],
                ]
            )
        else:
            raise GeometryError(
                f"Incorrect rotation axis: {axis}\n"
                "please select from: ['x', 'y', 'z']"
            )
    else:
        # Cute, but hard to understand!
        axis = np.array(axis) / np.linalg.norm(axis)  # Unit vector
        cos = np.cos(theta)
        sin = np.sin(theta)
        x, y, z = axis
        u_x = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        u_o_u = np.outer(axis, axis)
        r_matrix = cos * np.eye(3) + sin * u_x + (1 - cos) * u_o_u
    return r_matrix


def rotation_matrix_v1v2(v1, v2):
    """
    Get a rotation matrix based off two vectors.
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_angle = np.dot(v1, v2)
    d = np.cross(v1, v2)
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        matrix = np.identity(3) if cos_angle > 0.0 else -np.identity(3)
    else:
        d /= sin_angle

        eye = np.eye(3)
        ddt = np.outer(d, d)
        skew = np.array(
            [[0, d[2], -d[1]], [-d[2], 0, d[0]], [d[1], -d[0], 0]], dtype=np.float64
        )

        matrix = ddt + cos_angle * (eye - ddt) + sin_angle * skew

    return matrix


# =============================================================================
# Intersection tools
# =============================================================================


def vector_intersect(p1, p2, p3, p4):
    """
    Get the intersection point between two vectors.

    Parameters
    ----------
    p1: np.array(2)
        The first point on the first vector
    p2: np.array(2)
        The second point on the first vector
    p3: np.array(2)
        The first point on the second vector
    p4: np.array(2)
        The second point on the second vector

    Returns
    -------
    p_inter: np.array(2)
        The point of the intersection between the two vectors
    """
    da = p2 - p1
    db = p4 - p3

    if np.isclose(np.cross(da, db), 0):  # vectors parallel
        # NOTE: careful modifying this, different behaviour required...
        point = p2
    else:
        dp = p1 - p3
        dap = normal_vector(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        point = num / denom.astype(float) * db + p3
    return point


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


def get_intersect(loop1, loop2):
    """
    Calculates the intersection points between two Loops. Will return unique
    list of x, z intersections (no duplicates in x-z space)

    Parameters
    ----------
    loop1: Loop
        The Loops between which intersection points should be calculated
    loop2: Loop
        The Loops between which intersection points should be calculated

    Returns
    -------
    xi: np.array(N_itersection)
        The x coordinates of the intersection points
    zi: np.array(N_itersection)
        The z coordinates of the intersection points#

    Note
    ----
    D. Schwarz, <https://uk.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections>
    """  # noqa (W505)
    x1, y1 = loop1.d2
    x2, y2 = loop2.d2

    def inner_inter(x_1, x_2):
        n1, n2 = x_1.shape[0] - 1, x_2.shape[0] - 1
        xx1 = np.c_[x_1[:-1], x_1[1:]]
        xx2 = np.c_[x_2[:-1], x_2[1:]]
        return (
            np.less_equal(
                np.tile(xx1.min(axis=1), (n2, 1)).T, np.tile(xx2.max(axis=1), (n1, 1))
            ),
            np.greater_equal(
                np.tile(xx1.max(axis=1), (n2, 1)).T, np.tile(xx2.min(axis=1), (n1, 1))
            ),
        )

    x_x = inner_inter(x1, x2)
    z_z = inner_inter(y1, y2)
    m, k = np.nonzero(x_x[0] & x_x[1] & z_z[0] & z_z[1])
    n = len(m)
    a_m, xz, b_m = np.zeros((4, 4, n)), np.zeros((4, n)), np.zeros((4, n))
    a_m[0:2, 2, :] = -1
    a_m[2:4, 3, :] = -1
    a_m[0::2, 0, :] = np.diff(np.c_[x1, y1], axis=0)[m, :].T
    a_m[1::2, 1, :] = np.diff(np.c_[x2, y2], axis=0)[k, :].T
    b_m[0, :] = -x1[m].ravel()
    b_m[1, :] = -x2[k].ravel()
    b_m[2, :] = -y1[m].ravel()
    b_m[3, :] = -y2[k].ravel()
    for i in range(n):
        try:
            xz[:, i] = np.linalg.solve(a_m[:, :, i], b_m[:, i])
        except np.linalg.LinAlgError:
            # Parallel segments. Will raise numpy RuntimeWarnings
            xz[0, i] = np.nan
    in_range = (xz[0, :] >= 0) & (xz[1, :] >= 0) & (xz[0, :] <= 1) & (xz[1, :] <= 1)
    xz = xz[2:, in_range].T
    x, z = xz[:, 0], xz[:, 1]
    if len(x) > 0:
        x, z = np.unique([x, z], axis=1)
    return x, z


@nb.jit(cache=True, nopython=True)
def _intersect_count(x_inter, z_inter, x2, z2):
    args = []
    for i in range(len(x_inter)):
        for j in range(len(x2) - 1):
            if check_linesegment(
                np.array([x2[j], z2[j]]),
                np.array([x2[j + 1], z2[j + 1]]),
                np.array([x_inter[i], z_inter[i]]),
            ):
                args.append(j)
                break
    return np.array(args)


def join_intersect(loop1, loop2, get_arg=False):
    """
    Add the intersection points between Loop1 and Loop2 to Loop1.

    Parameters
    ----------
    loop1: Loop
        The Loop to which the intersection points should be added
    loop2: Loop
        The intersecting Loop
    get_arg: bool (default = False)

    Returns
    -------
    (if get_arg is True)
    args: list(int, int, ..) of len(N_intersections)
        The arguments of Loop1 in which the intersections were added.

    Notes
    -----
    Modifies loop1
    """
    x_inter, z_inter = get_intersect(loop1, loop2)
    xz = loop1.d2
    args = _intersect_count(x_inter, z_inter, xz[0], xz[1])

    orderr = args.argsort()
    x_int = x_inter[orderr]
    z_int = z_inter[orderr]

    args = _intersect_count(x_int, z_int, xz[0], xz[1])

    # TODO: Check for duplicates and order correctly based on distance
    # u, counts = np.unique(args, return_counts=True)

    count = 0
    for i, arg in enumerate(args):
        if i > 0 and args[i - 1] == arg:
            # Two intersection points, one after the other
            bump = 0
        else:
            bump = 1
        if not loop1._check_already_in([x_int[i], z_int[i]]):
            # Only increment counter if the intersection isn't already in the Loop
            loop1.insert([x_int[i], z_int[i]], pos=arg + count + bump)
            count += 1

    if get_arg:
        args = []
        for x, z in zip(x_inter, z_inter):
            args.append(loop1.argmin([x, z]))
        return list(set(args))


# =============================================================================
# Coordinate creation
# =============================================================================


def make_circle_arc(
    radius, x_centre=0, y_centre=0, angle=2 * np.pi, n_points=200, start_angle=0
):
    """
    Make a circle arc of a specified radius and angle at a given location.

    Parameters
    ----------
    radius: float
        The radius of the circle arc
    x_centre: float
        The x coordinate of the circle arc centre
    y_centre: float
        The y coordinate of the circle arc centre
    angle: float
        The angle of the circle arc [radians]
    n_points: int
        The number of points on the circle
    start_angle: float
        The starting angle of the circle arc

    Returns
    -------
    x: np.array
        The x coordinates of the circle arc
    y: np.array
        The y coordinates of the circle arc
    """
    n = np.linspace(start_angle, start_angle + angle, n_points)
    x = x_centre + radius * np.cos(n)
    y = y_centre + radius * np.sin(n)
    if angle == 2 * np.pi:
        # Small number correction (close circle exactly)
        x[-1] = x[0]
        y[-1] = y[0]
    return x, y


def get_control_point(loop):
    """
    Find an arbitrary control point which sits inside a specified Loop

    If a Shell is given, finds a point which sits on the solid part of the Shell.

    Parameters
    ----------
    loop: Loop or Shell
        The geometry to find a control point for. Must be 2-D.

    Returns
    -------
    float, float
        An arbitrary control point for the Loop or Shell.
    """
    if loop.__class__.__name__ == "Loop":
        cp = [loop.centroid[0], loop.centroid[1]]
        if loop.point_in_poly(cp):
            return cp
        else:
            return _montecarloloopcontrol(loop)
    else:
        raise GeometryError(f"Unrecognised type: {type(loop)}.")


def _montecarloloopcontrol(loop):
    """
    Find an arbitrary point inside a Loop

    If the centroid doesn't work, will use brute force...

    Parameters
    ----------
    loop: Loop
        The geometry to find a control point for. Must be 2-D.

    Returns
    -------
    float, float
        An arbitrary control point for the Loop.
    """
    xmin, xmax = np.min(loop.d2[0]), np.max(loop.d2[0])
    dx = xmax - xmin
    ymin, ymax = np.min(loop.d2[1]), np.max(loop.d2[1])
    dy = ymax - ymin
    i = 0
    while i < 1000:
        i += 1
        n, m = np.random.rand(2)
        x = xmin + n * dx
        y = ymin + m * dy
        if loop.point_in_poly([x, y]):
            return [x, y]
    raise GeometryError(
        "Unable to find a control point for this Loop using brute force."
    )

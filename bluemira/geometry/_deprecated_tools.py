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
A collection of geometry tools.
"""

from functools import partial
from itertools import zip_longest

import numba as nb
import numpy as np
from numba.np.extensions import cross2d
from pyquaternion import Quaternion
from scipy.interpolate import UnivariateSpline, interp1d

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.bound_box import BoundingBox
from bluemira.geometry.constants import CROSS_P_TOL, DOT_P_TOL
from bluemira.geometry.coordinates import _validate_coordinates, get_area
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.tools import flatten_iterable

# =============================================================================
# Errors
# =============================================================================


class MixedFaceAreaError(GeometryError):
    """
    An error to raise when the area of a mixed face does not give a good match to the
    area enclosed by the original coordinates.
    """

    pass


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
    # Do some protection of numba against integers and lists
    a_c = np.array([point_c[0] - point_a[0], point_c[1] - point_a[1]], dtype=np.float_)
    a_b = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]], dtype=np.float_)

    distance = np.sqrt(np.sum(a_b**2))
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


@nb.jit(forceobj=True)
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


def check_closed(x, y, z):
    """
    Check that the coordinates are closed e.g. first element == last element for all
    dimensions.

    Parameters
    ----------
    x: np.array
        The x coordinates
    y: np.array
        The y coorindates
    z: np.array
        The z coordinates

    Returns
    -------
    closed: bool
        True if the coordinates are closed
    """
    if x[0] == x[-1] and y[0] == y[-1] and z[0] == z[-1]:
        return True
    return False


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


def get_angle_between_points(p0, p1, p2):
    """
    Angle between points. P1 is vertex of angle. ONly tested in 2d
    """
    if not all(isinstance(p, np.ndarray) for p in [p0, p1, p2]):
        p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
    ba = p0 - p1
    bc = p2 - p1
    return get_angle_between_vectors(ba, bc)


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
        The angle between the vectors [radians]
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


def segment_lengths(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Returns the length of each individual segment in a set of coordinates

    Parameters
    ----------
    x: array_like
        x coordinates of the loop [m]
    y: array_like
        y coordinates of the loop [m]
    z: array_like
        z coordinates of the loop [m]

    Returns
    -------
    dL: np.array(N)
        The length of each individual segment in the loop
    """
    return np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)


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
    return BoundingBox.from_xyz(x, y, z).get_box_arrays()


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


def vector_lengthnorm_2d(x, z):
    """
    Get a normalised 1-D parameterisation of an x, z loop.

    Parameters
    ----------
    x: array_like
        x coordinates of the loop [m]
    z: array_like
        z coordinates of the loop [m]

    Returns
    -------
    total_length: np.array(N)
        The cumulative normalised length of each individual segment in the loop
    """
    total_length = np.append(0, np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)))
    return total_length / total_length[-1]


def innocent_smoothie(x, z, n=500, s=0):
    """
    Get a smoothed interpolated set of coordinates.

    Parameters
    ----------
    x: array_like
        x coordinates of the loop [m]
    z: array_like
        z coordinates of the loop [m]
    n: int
        The number of interpolation points
    s: Union[int, float]
        The smoothing parameter to use. 0 results in no smoothing (default)

    Returns
    -------
    x: array_like
        Smoothed, interpolated x coordinates of the loop [m]
    z: array_like
        Smoothed, interpolated z coordinates of the loop [m]
    """
    length_norm = vector_lengthnorm_2d(x, z)
    n = int(n)
    l_interp = np.linspace(0, 1, n)
    if s == 0:
        x = interp1d(length_norm, x)(l_interp)
        z = interp1d(length_norm, z)(l_interp)
    else:
        x = UnivariateSpline(length_norm, x, s=s)(l_interp)
        z = UnivariateSpline(length_norm, z, s=s)(l_interp)
    return x, z


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


# =============================================================================
# Intersection tools
# =============================================================================


def vector_intersect(p1, p2, p3, p4):
    """
    Get the intersection point between two 2-D vectors.

    Parameters
    ----------
    p1: np.ndarray(2)
        The first point on the first vector
    p2: np.ndarray(2)
        The second point on the first vector
    p3: np.ndarray(2)
        The first point on the second vector
    p4: np.ndarray(2)
        The second point on the second vector

    Returns
    -------
    p_inter: np.ndarray(2)
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


def vector_intersect_3d(p_1, p_2, p_3, p_4):
    """
    Get the intersection point between two 3-D vectors.

    Parameters
    ----------
    p1: np.ndarray(3)
        The first point on the first vector
    p2: np.ndarray(3)
        The second point on the first vector
    p3: np.ndarray(3)
        The first point on the second vector
    p4: np.ndarray(3)
        The second point on the second vector

    Returns
    -------
    p_inter: np.ndarray(3)
        The point of the intersection between the two vectors

    Notes
    -----
    Credit: Paul Bourke at
    http://paulbourke.net/geometry/pointlineplane/#:~:text=The%20shortest%20line%20between%20two%20lines%20in%203D
    """
    p_13 = p_1 - p_3
    p_43 = p_4 - p_3

    if np.linalg.norm(p_13) < EPS:
        raise GeometryError("No intersection between 3-D lines.")
    p_21 = p_2 - p_1
    if np.linalg.norm(p_21) < EPS:
        raise GeometryError("No intersection between 3-D lines.")

    d1343 = np.dot(p_13, p_43)
    d4321 = np.dot(p_43, p_21)
    d1321 = np.dot(p_13, p_21)
    d4343 = np.dot(p_43, p_43)
    d2121 = np.dot(p_21, p_21)

    denom = d2121 * d4343 - d4321 * d4321

    if np.abs(denom) < EPS:
        raise GeometryError("No intersection between 3-D lines.")

    numer = d1343 * d4321 - d1321 * d4343

    mua = numer / denom
    intersection = p_1 + mua * p_21
    return intersection


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
    """  # noqa :W505
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
        if loop.point_inside(cp):
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
        if loop.point_inside([x, y]):
            return [x, y]
    raise GeometryError(
        "Unable to find a control point for this Loop using brute force."
    )


# =============================================================================
# Coordinates conversion
# =============================================================================


def convert_coordinates_to_wire(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    label="",
    method="mixed",
    **kwargs,
):
    """
    Converts the provided coordinates into a BluemiraWire using the specified method.

    Parameters
    ----------
    x: np.ndarray
        The x coordinates of points to be converted to a BluemiraWire object
    y: np.ndarray
        The y coordinates of points to be converted to a BluemiraWire object
    z: np.ndarray
        The z coordinates of points to be converted to a BluemiraWire object
    method: str
        The conversion method to be used:

            - mixed (default): results in a mix of splines and polygons
            - polygon: pure polygon representation
            - spline: pure spline representation

    label: str
        The label for the resulting BluemiraWire object
    kwargs: Dict[str, Any]
        Any other arguments for the conversion method, see e.g. make_mixed_face

    Returns
    -------
    face: BluemiraWire
        The resulting BluemiraWire from the conversion
    """
    method_map = {
        "mixed": make_mixed_wire,
        "polygon": partial(make_wire, spline=False),
        "spline": partial(make_wire, spline=True),
    }
    wire = method_map[method](x, y, z, label=label, **kwargs)
    return wire


def convert_coordinates_to_face(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    method="mixed",
    label="",
    **kwargs,
):
    """
    Converts the provided coordinates into a BluemiraFace using the specified method.

    Parameters
    ----------
    x: np.ndarray
        The x coordinates of points to be converted to a BluemiraFace object
    y: np.ndarray
        The y coordinates of points to be converted to a BluemiraFace object
    z: np.ndarray
        The z coordinates of points to be converted to a BluemiraFace object
    method: str
        The conversion method to be used:

            - mixed (default): results in a mix of splines and polygons
            - polygon: pure polygon representation
            - spline: pure spline representation

    label: str
        The label for the resulting BluemiraFace object
    kwargs: Dict[str, Any]
        Any other arguments for the conversion method, see e.g. make_mixed_face

    Returns
    -------
    face: BluemiraFace
        The resulting BluemiraFace from the conversion
    """
    method_map = {
        "mixed": make_mixed_face,
        "polygon": partial(make_face, spline=False),
        "spline": partial(make_face, spline=True),
    }
    face = method_map[method](x, y, z, label=label, **kwargs)
    return face


def make_mixed_wire(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    label="",
    *,
    median_factor=2.0,
    n_segments=4,
    a_acute=150,
    cleaning_atol=1e-6,
    allow_fallback=True,
    debug=False,
):
    """
    Construct a BluemiraWire object from the provided coordinates using a combination of
    polygon and spline wires. Polygons are determined by having a median length larger
    than the threshold or an angle that is more acute than the threshold.

    Parameters
    ----------
    x: np.ndarray
        The x coordinates of points to be converted to a BluemiraWire object
    y: np.ndarray
        The y coordinates of points to be converted to a BluemiraWire object
    z: np.ndarray
        The z coordinates of points to be converted to a BluemiraWire object
    label: str
        The label for the resulting BluemiraWire object

    Other Parameters
    ----------------
    median_factor: float
        The factor of the median for which to filter segment lengths
        (below median_factor*median_length --> spline)
    n_segments: int
        The minimum number of segments for a spline
    a_acute: float
        The angle [degrees] between two consecutive segments deemed to be too
        acute to be fit with a spline.
    cleaning_atol: float
        If a point lies within this distance [m] of the previous point then it will be
        treated as a duplicate and removed. This can stabilise the conversion in cases
        where the point density is too high for a wire to be constructed as a spline.
        By default this is set to 1e-6.
    allow_fallback: bool
        If True then a failed attempt to make a mixed wire will fall back to a polygon
        wire, else an exception will be raised. By default True.
    debug: bool
        Whether or not to print debugging information

    Returns
    -------
    wire: BluemiraWire
        The BluemiraWire of the mixed polygon/spline Loop
    """
    mfm = MixedFaceMaker(
        x,
        y,
        z,
        label=label,
        median_factor=median_factor,
        n_segments=n_segments,
        a_acute=a_acute,
        cleaning_atol=cleaning_atol,
        debug=debug,
    )
    try:
        mfm.build()

    except RuntimeError as e:
        if allow_fallback:
            bluemira_warn(
                f"CAD: MixedFaceMaker failed with error {e} "
                "- falling back to a polygon wire."
            )
            return make_wire(x, y, z, label=label)
        else:
            raise

    return mfm.wire


def make_mixed_face(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    label="",
    *,
    median_factor=2.0,
    n_segments=4,
    a_acute=150,
    cleaning_atol=1e-6,
    area_rtol=5e-2,
    allow_fallback=True,
    debug=False,
):
    """
    Construct a BluemiraFace object from the provided coordinates using a combination of
    polygon and spline wires. Polygons are determined by having a median length larger
    than the threshold or an angle that is more acute than the threshold.

    Parameters
    ----------
    x: np.ndarray
        The x coordinates of points to be converted to a BluemiraFace object
    y: np.ndarray
        The y coordinates of points to be converted to a BluemiraFace object
    z: np.ndarray
        The z coordinates of points to be converted to a BluemiraFace object
    label: str
        The label for the resulting BluemiraFace object

    Other Parameters
    ----------------
    median_factor: float
        The factor of the median for which to filter segment lengths
        (below median_factor*median_length --> spline)
    n_segments: int
        The minimum number of segments for a spline
    a_acute: float
        The angle [degrees] between two consecutive segments deemed to be too
        acute to be fit with a spline.
    cleaning_atol: float
        If a point lies within this distance [m] of the previous point then it will be
        treated as a duplicate and removed. This can stabilise the conversion in cases
        where the point density is too high for a wire to be constructed as a spline.
        By default this is set to 1e-6.
    area_rtol: float
        If the area of the resulting face deviates by this relative value from the area
        enclosed by the provided coordinates then the conversion will fail and either
        fall back to a polygon-like face or raise an exception, depending on the setting
        of `allow_fallback`.
    allow_fallback: bool
        If True then a failed attempt to make a mixed face will fall back to a polygon
        wire, else an exception will be raised. By default True.
    debug: bool
        Whether or not to print debugging information

    Returns
    -------
    face: BluemiraFace
        The BluemiraFace of the mixed polygon/spline Loop
    """
    mfm = MixedFaceMaker(
        x,
        y,
        z,
        label=label,
        median_factor=median_factor,
        n_segments=n_segments,
        a_acute=a_acute,
        cleaning_atol=cleaning_atol,
        debug=debug,
    )
    try:
        mfm.build()

    except RuntimeError as e:
        if allow_fallback:
            bluemira_warn(
                f"CAD: MixedFaceMaker failed with error {e} "
                "- falling back to a polygon face."
            )
            return make_face(x, y, z, label=label)
        else:
            raise

    # Sometimes there won't be a RuntimeError, and you get a free SIGSEGV for your
    # troubles.
    face_area = mfm.face.area
    coords_area = get_area(x, y, z)
    if np.isclose(coords_area, face_area, rtol=area_rtol):
        return mfm.face
    else:
        if allow_fallback:
            bluemira_warn(
                f"CAD: MixedFaceMaker resulted in a face with area {face_area} "
                f"but the provided coordinates enclosed an area of {coords_area} "
                "- falling back to a polygon face."
            )
            return make_face(x, y, z, label=label)
        else:
            raise MixedFaceAreaError(
                f"MixedFaceMaker resulted in a face with area {face_area} "
                f"but the provided coordinates enclosed an area of {coords_area}."
            )


def make_wire(x, y, z, label="", spline=False):
    """
    Makes a wire from a set of coordinates.

    Parameters
    ----------
    x: np.ndarray
        The x coordinates of points to be converted to a BluemiraWire object
    y: np.ndarray
        The y coordinates of points to be converted to a BluemiraWire object
    z: np.ndarray
        The z coordinates of points to be converted to a BluemiraWire object
    label: str
        The label for the resulting BluemiraWire object
    spline: bool
        If True then creates the BluemiraWire using a Bezier spline curve, by default
        False

    Returns
    -------
    wire: BluemiraWire
        The BluemiraWire bound by the coordinates
    """
    wire_func = cadapi.interpolate_bspline if spline else cadapi.make_polygon
    return BluemiraWire(wire_func(np.array([x, y, z]).T), label=label)


def make_face(x, y, z, label="", spline=False):
    """
    Makes a face from a set of coordinates.

    Parameters
    ----------
    x: np.ndarray
        The x coordinates of points to be converted to a BluemiraFace object
    y: np.ndarray
        The y coordinates of points to be converted to a BluemiraFace object
    z: np.ndarray
        The z coordinates of points to be converted to a BluemiraFace object
    label: str
        The label for the resulting BluemiraFace object
    spline: bool
        If True then creates the BluemiraFace using a Bezier spline curve, by default
        False

    Returns
    -------
    face: BluemiraFace
        The BluemiraFace bound by the coordinates
    """
    wire = make_wire(x, y, z, label=label, spline=spline)
    return BluemiraFace(wire, label=label)


class MixedFaceMaker:
    """
    Utility class for the creation of Faces that combine splines and polygons.

    Polygons are detected by median length and turning angle.

    Parameters
    ----------
    x: np.ndarray
        The x coordinates of points to be converted to a BluemiraFace object
    y: np.ndarray
        The y coordinates of points to be converted to a BluemiraFace object
    z: np.ndarray
        The z coordinates of points to be converted to a BluemiraFace object
    label: str
        The label for the resulting BluemiraFace object

    Other Parameters
    ----------------
    median_factor: float
        The factor of the median for which to filter segment lengths
        (below median_factor*median_length --> spline)
    n_segments: int
        The minimum number of segments for a spline
    a_acute: float
        The angle [degrees] between two consecutive segments deemed to be too
        acute to be fit with a spline.
    cleaning_atol: float
        If a point lies within this distance [m] of the previous point then it will be
        treated as a duplicate and removed. This can stabilise the conversion in cases
        where the point density is too high for a wire to be constructed as a spline.
        By default this is set to 1e-6.
    debug: bool
        Whether or not to print debugging information
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        label="",
        *,
        median_factor=2.0,
        n_segments=4,
        a_acute=150,
        cleaning_atol=1e-6,
        debug=False,
    ):
        _validate_coordinates(x, y, z)
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)
        self.num_points = len(x)

        self.label = label

        self.median_factor = median_factor
        self.n_segments = n_segments
        self.a_acute = a_acute
        self.cleaning_atol = cleaning_atol
        self.debug = debug

        # Constructors
        self.edges = None
        self.wire = None
        self.face = None
        self.polygon_loops = None
        self.spline_loops = None
        self.flag_spline_first = None
        self._debugger = None

    def build(self):
        """
        Carry out the MixedFaceMaker sequence to make a Face
        """
        # Get the vertices of polygon-like segments
        p_vertices = self._find_polygon_vertices()

        if len(p_vertices) > 0:
            # identify sequences of polygon indices
            p_sequences = self._get_polygon_sequences(p_vertices)

            if (
                len(p_sequences) == 1
                and p_sequences[0][0] == 0
                and p_sequences[0][-1] == len(p_vertices) - 1
            ):
                # All vertices are pure polygon-like so just make the wire
                self.wires = make_wire(self.x, self.y, self.z, spline=False)
            else:
                # Get the (negative) of the polygon sequences to get spline sequences
                s_sequences = self._get_spline_sequences(p_sequences)

                if self.debug:
                    print("p_sequences :", p_sequences)
                    print("s_sequences :", s_sequences)

                # Make coordinates for all the segments
                self._make_subcoordinates(p_sequences, s_sequences)

                # Make the wires for each of the sub-coordinates, and daisychain them
                self._make_subwires()
        else:
            # All vertices are pure spline-like so just make the wire
            self.wires = make_wire(self.x, self.y, self.z, spline=True)

        # Finally, make the OCC face from the wire formed from the boundary wires
        self._make_wire()
        self.face = BluemiraFace(self.wire, label=self.label)

    def _find_polygon_vertices(self):
        """
        Finds all vertices in the loop which belong to polygon-like edges

        Returns
        -------
        vertices: np.ndarray(dtype=int)
            The vertices of the loop which are polygon-like
        """
        seg_lengths = segment_lengths(self.x, self.y, self.z)
        median = np.median(seg_lengths)

        long_indices = np.where(seg_lengths > self.median_factor * median)[0]

        # find sharp angle indices
        angles = np.zeros(len(self.x) - 2)
        for i in range(len(self.x) - 2):
            angles[i] = get_angle_between_points(
                [self.x[i], self.y[i], self.z[i]],
                [self.x[i + 1], self.y[i + 1], self.z[i + 1]],
                [self.x[i + 2], self.y[i + 2], self.z[i + 2]],
            )
        if (
            self.x[0] == self.x[-1]
            and self.y[0] == self.y[-1]
            and self.z[0] == self.z[-1]
        ):
            # Get the angle over the closed joint
            join_angle = get_angle_between_points(
                [self.x[-2], self.y[-2], self.z[-2]],
                [self.x[0], self.y[0], self.z[0]],
                [self.x[1], self.y[1], self.z[1]],
            )
            angles = np.append(angles, join_angle)

        angles = np.rad2deg(angles)
        sharp_indices = np.where((angles <= self.a_acute) & (angles != 0))[0]
        # Convert angle numbering to segment numbering (both segments of angle)
        sharp_edge_indices = []
        for index in sharp_indices:
            sharp_edges = [index + 1, index + 2]
            sharp_edge_indices.extend(sharp_edges)
        sharp_edge_indices = np.array(sharp_edge_indices)

        # build ordered set of polygon edge indices
        indices = np.unique(np.append(long_indices, sharp_edge_indices))

        # build ordered set of polygon vertex indices
        vertices = []
        for index in indices:
            if index == self.num_points:
                # If it is the last index, do not overshoot
                vertices.extend([index])
            else:
                vertices.extend([index, index + 1])
        vertices = np.unique(np.array(vertices, dtype=int))
        return vertices

    def _get_polygon_sequences(self, vertices: np.ndarray):
        """
        Gets the sequences of polygon segments

        Parameters
        ----------
        vertices: np.ndarray(dtype=int)
            The vertices of the loop which are polygon-like

        Returns
        -------
        p_sequences: list([start, end], [start, end])
            The list of start and end tuples of the polygon segments
        """
        sequences = []

        if len(vertices) == 0:
            return sequences

        start = vertices[0]
        for i, vertex in enumerate(vertices[:-1]):

            delta = vertices[i + 1] - vertex

            if i == len(vertices) - 2:
                # end of loop clean-up
                end = vertices[i + 1]
                sequences.append([start, end])
                break

            if delta <= self.n_segments:
                # Spline would be too short, so stitch polygons together
                continue
            else:
                end = vertex
                sequences.append([start, end])
                start = vertices[i + 1]  # reset start index

        if not sequences:
            raise GeometryError("Not a good candidate for a mixed face ==> spline")

        if (
            len(sequences) == 1
            and sequences[0][0] == 0
            and sequences[0][1] == len(vertices) - 1
        ):
            # Shape is a pure polygon
            return sequences

        # Now check the start and end of the loop, to see if a polygon segment
        # bridges the join
        first_p_vertex = sequences[0][0]
        last_p_vertex = sequences[-1][1]

        if first_p_vertex <= self.n_segments:
            if self.num_points - last_p_vertex <= self.n_segments:
                start_offset = self.n_segments - first_p_vertex
                end_offset = (self.num_points - last_p_vertex) + self.n_segments
                total = start_offset + end_offset
                if total <= self.n_segments:
                    start = sequences[-1][0]
                    end = sequences[0][1]
                    # Remove first sequence
                    sequences = sequences[1:]
                    # Replace last sequence with bridged sequence
                    sequences[-1] = [start, end]

        last_p_vertex = sequences[-1][1]
        if self.num_points - last_p_vertex <= self.n_segments:
            # There is a small spline section at the end of the loop, that
            # needs to be bridged
            if sequences[0][0] == 0:
                # There is no bridge -> take action
                start = sequences[-1][0]
                end = sequences[0][1]
                sequences = sequences[1:]
                sequences[-1] = [start, end]

        return sequences

    def _get_spline_sequences(self, polygon_sequences):
        """
        Gets the sequences of spline segments

        Parameters
        ----------
        polygon_sequences: list([start, end], [start, end])
            The list of start and end tuples of the polygon segments

        Returns
        -------
        spline_sequences: list([start, end], [start, end])
            The list of start and end tuples of the spline segments
        """
        spline_sequences = []

        # Catch the start, if polygon doesn't start at zero, and there is no
        # bridge
        last = polygon_sequences[-1]
        if last[0] > last[1]:  # there is a polygon bridge
            pass  # Don't add a spline at the start
        else:
            # Check that the first polygon segment doesn't start at zero
            first = polygon_sequences[0]
            if first[0] == 0:
                pass
            else:  # It doesn't start at zero and there is no bridge: catch
                spline_sequences.append([0, first[0]])

        for i, seq in enumerate(polygon_sequences[:-1]):
            start = seq[1]
            end = polygon_sequences[i + 1][0]
            spline_sequences.append([start, end])

        # Catch the end, if polygon doesn't end at end
        if last[1] == self.num_points:
            # NOTE: if this is true, there can't be a polygon bridge
            pass
        else:
            if last[0] > last[1]:  # there is a polygon bridge
                spline_sequences.append([last[1], polygon_sequences[0][0]])
            else:
                spline_sequences.append([last[1], self.num_points])

        # Check if we need to make a spline bridge
        spline_first = spline_sequences[0][0]
        spline_last = spline_sequences[-1][1]
        if (spline_first == 0) and (spline_last == self.num_points):
            # Make a spline bridge
            start = spline_sequences[-1][0]
            end = spline_sequences[0][1]
            spline_sequences = spline_sequences[1:]
            spline_sequences[-1] = [start, end]

        if spline_sequences[0][0] == 0:
            self.flag_spline_first = True
        else:
            self.flag_spline_first = False

        return spline_sequences

    def _clean_coordinates(self, coords: np.ndarray):
        """
        Clean the provided coordinates by removing any values that are closer than the
        instance's cleaning_atol value.

        Parameters
        ----------
        coords: np.ndarray
            3D array of coordinates to be cleaned.

        Returns
        -------
        clean_coords: np.ndarray
            3D array of cleaned coordinates.
        """
        mask = ~np.isclose(segment_lengths(*coords), 0, atol=self.cleaning_atol)
        mask = np.insert(mask, 0, True)
        return coords[:, mask]

    def _make_subcoordinates(
        self, polygon_sequences: np.ndarray, spline_sequences: np.ndarray
    ):
        polygon_coords = []
        spline_coords = []

        for seg in polygon_sequences:
            if seg[0] > seg[1]:
                # There is a bridge
                coords = np.hstack(
                    (
                        np.array([self.x[seg[0] :], self.y[seg[0] :], self.z[seg[0] :]]),
                        np.array(
                            [
                                self.x[0 : seg[1] + 1],
                                self.y[0 : seg[1] + 1],
                                self.z[0 : seg[1] + 1],
                            ]
                        ),
                    )
                )
            else:
                coords = np.array(
                    [
                        self.x[seg[0] : seg[1] + 1],
                        self.y[seg[0] : seg[1] + 1],
                        self.z[seg[0] : seg[1] + 1],
                    ]
                )
            clean_coords = self._clean_coordinates(coords)
            if all(shape >= 2 for shape in clean_coords.shape):
                polygon_coords.append(clean_coords)

        for seg in spline_sequences:
            if seg[0] > seg[1]:
                # There is a bridge
                coords = np.hstack(
                    (
                        np.array([self.x[seg[0] :], self.y[seg[0] :], self.z[seg[0] :]]),
                        np.array(
                            [
                                self.x[0 : seg[1] + 1],
                                self.y[0 : seg[1] + 1],
                                self.z[0 : seg[1] + 1],
                            ]
                        ),
                    )
                )
            else:
                coords = np.array(
                    [
                        self.x[seg[0] : seg[1] + 1],
                        self.y[seg[0] : seg[1] + 1],
                        self.z[seg[0] : seg[1] + 1],
                    ]
                )
            clean_coords = self._clean_coordinates(coords)
            if all(shape >= 2 for shape in clean_coords.shape):
                spline_coords.append(clean_coords)

        self.spline_coords = spline_coords
        self.polygon_coords = polygon_coords

    def _make_subwires(self):
        # First daisy-chain correctly...
        coords_order = []
        if self.flag_spline_first:
            set1, set2 = self.spline_coords, self.polygon_coords
        else:
            set2, set1 = self.spline_coords, self.polygon_coords
        for i, (a, b) in enumerate(zip_longest(set1, set2)):
            if a is not None:
                coords_order.append(set1[i])
            if b is not None:
                coords_order.append(set2[i])

        for i, coords in enumerate(coords_order[:-1]):
            if not (coords[:, -1] == coords_order[i + 1][:, 0]).all():
                coords_order[i + 1] = coords_order[i + 1][:, ::-1]
                if i == 0:
                    if not (coords[:, -1] == coords_order[i + 1][:, 0]).all():
                        coords = coords[:, ::-1]
                        if not (coords[:, -1] == coords_order[i + 1][:, 0]).all():
                            coords_order[i + 1] = coords_order[i + 1][:, ::-1]

        if self.flag_spline_first:
            set1 = [
                make_wire(*spline_coord, spline=True)
                for spline_coord in self.spline_coords
            ]
            set2 = [
                make_wire(*polygon_coord, spline=False)
                for polygon_coord in self.polygon_coords
            ]
        else:
            set2 = [
                make_wire(*spline_coord, spline=True)
                for spline_coord in self.spline_coords
            ]
            set1 = [
                make_wire(*polygon_coord, spline=False)
                for polygon_coord in self.polygon_coords
            ]

        wires = []
        for i, (a, b) in enumerate(zip_longest(set1, set2)):
            if a is not None:
                wires.append(a)

            if b is not None:
                wires.append(b)

        self.wires = list(flatten_iterable(wires))
        self._debugger = coords_order

    def _make_wire(self):
        self.wire = BluemiraWire(self.wires)

    def _make_face(self):
        self.face = BluemiraFace(self.wire)

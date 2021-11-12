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
A collection of geometry utility functions
"""
import numba as nb
import numpy as np
from numpy.linalg import LinAlgError
from collections.abc import Iterable
from pyquaternion import Quaternion
from scipy.interpolate import interp1d
from shapely.geometry import MultiLineString, MultiPolygon
from shapely.ops import unary_union
from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn, bluemira_print
from BLUEPRINT.base.error import GeometryError
from BLUEPRINT.geometry.constants import CROSS_P_TOL, DOT_P_TOL


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


def project_point_axis(point, axis):
    """
    Project a 3-D point onto a 3-D axis.

    \t:math:`\\mathbf{p_{proj}} = \\dfrac{\\mathbf{p}\\cdot\\mathbf{a}}{\\mathbf{a}\\cdot\\mathbf{a}}\\mathbf{a}`

    Parameters
    ----------
    point: np.array(3)
        The point to project onto the axis
    axis: np.array(3)
        The axis onto which to project the point

    Returns
    -------
    projection: np.array(3)
        The coordinates of the projected point
    """  # noqa (W505)
    point = np.array(point)
    axis = np.array(axis)
    return axis * np.dot(point, axis) / np.dot(axis, axis)


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


def make_box_xz(x_min, x_max, z_min, z_max):
    """
    Create a box in the xz plane given the min and max x,z params

    Returns
    -------
    box : Loop
    """
    # Import here to avoid circular import
    from BLUEPRINT.geometry.loop import Loop

    if x_max < x_min:
        raise GeometryError("Require x_max > x_min")
    if z_max < z_min:
        raise GeometryError("Require z_max > z_min")

    x_box = [x_min, x_max, x_max, x_min]
    z_box = [z_max, z_max, z_min, z_min]
    box = Loop(x=x_box, z=z_box)
    box.close()
    return box


def grid_2d_contour(loop):
    """
    Grid a smooth contour and get the outline of the cells it encompasses.

    Parameters
    ----------
    loop: Loop
        The closed ccw Loop of the contour to grid

    Returns
    -------
    x: np.array
        The x coordinates of the grid-loop
    z: np.array
        The z coordinates of the grid-loop
    """
    x, z = loop.d2
    x_new, z_new = [], []
    for i, (xi, zi) in enumerate(zip(x[:-1], z[:-1])):
        x_new.append(xi)
        z_new.append(zi)
        if not np.isclose(xi, x[i + 1]) and not np.isclose(zi, z[i + 1]):
            # Add an intermediate point (ccw)
            if x[i + 1] > xi and z[i + 1] < zi:
                x_new.append(x[i + 1])
                z_new.append(zi)
            elif x[i + 1] > xi and z[i + 1] > zi:
                x_new.append(xi)
                z_new.append(z[i + 1])
            elif x[i + 1] < xi and z[i + 1] > zi:
                x_new.append(x[i + 1])
                z_new.append(zi)
            else:
                x_new.append(xi)
                z_new.append(z[i + 1])

    x_new.append(x[0])
    z_new.append(z[0])
    return np.array(x_new), np.array(z_new)


def loop_plane_intersect(loop, plane):
    """
    Calculates the intersection of a loop with a plane

    Parameters
    ----------
    loop: BLUEPRINT Loop object
        The loop to calculate the intersection on
    plane: BLUEPRINT Plane object
        The plan to calculate the intersection with

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


def circle_line_intersect(x_c, z_c, r, x1, y1, x2, y2):
    """
    Fast and avoids Loops (pure circle)

    Parameters
    ----------
    x_c: float
        The x coordinate of the centre of the circle
    z_c: float
        The z coordinate of the centre of the circle
    r: float
        The radius of the circle
    x1, y1: float, float
        The coordinates of the first point of the line
    x2, y2: float, float
        The coordinates of the second point of the line

    Returns
    -------
    x: list or None
        The x coordinates of the intersections of the line with the circle
    z: list or None
        The z coordinates of the intersections of the line with the circle
        Will return None if no intersections are found
    """
    dx = x2 - x1
    if dx == 0:
        x = np.array([x1, x1])
        t2 = r ** 2 - (x1 - x_c) ** 2
        if t2 < 0:
            bluemira_warn("No intersection between line and circle!")
            return None
        t = np.sqrt(t2)
        if t == 0:  # tangency
            return [x1], [z_c]
        z = np.array([z_c - t, z_c + t])
        return x, z

    dy = y2 - y1
    if dy == 0:
        z = np.array([y1, y1])
        t2 = r ** 2 - (y1 - z_c) ** 2
        if t2 < 0:
            bluemira_warn("No intersection between line and circle!")
            return None
        t = np.sqrt(t2)
        if t == 0:  # tangency
            return [x_c], [y1]
        x = np.array([x_c - t, x_c + t])
        return x, z

    dr2 = dx ** 2 + dy ** 2
    det = x1 * y2 - x2 * y1
    delta = r ** 2 * dr2 - det ** 2
    if delta < 0:
        bluemira_warn("No intersection between line and circle!")
        return None

    t = np.sqrt(r ** 2 * dr2 - det ** 2)
    t1 = np.sign(dy) * dx * t
    x = np.array([det * dy + t1, det * dy - t1]) / dr2
    z = np.array([-det * dx + np.abs(dy) * t, -det * dx - np.abs(dy) * t]) / dr2
    if delta < 1e-10:  # tangency
        return [x[0]], [z[0]]
    return x, z


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
        except LinAlgError:
            # Parallel segments. Will raise numpy RuntimeWarnings
            xz[0, i] = np.nan
    in_range = (xz[0, :] >= 0) & (xz[1, :] >= 0) & (xz[0, :] <= 1) & (xz[1, :] <= 1)
    xz = xz[2:, in_range].T
    x, z = xz[:, 0], xz[:, 1]
    if len(x) > 0:
        x, z = np.unique([x, z], axis=1)
    return x, z


@nb.jit(cache=True, forceobj=True)
def check_linesegment(point_a, point_b, point_c):
    """
    Check that point C is on the line between points A and B

    Parameters
    ----------
    point_a, point_b: [float, float]*2
        The line segment points
    point_c: [float, float]
        The point which to check is on A--B

    Returns
    -------
    check: bool
        True: if C on A--B, else False
    """
    point_a, point_b, point_c = np.array(point_a), np.array(point_b), np.array(point_c)
    a_c, a_b = point_c - point_a, point_b - point_a
    distance = distance_between_points(point_a, point_b)
    if np.abs(np.cross(a_b, a_c)) > CROSS_P_TOL * distance:
        return False
    k_ac = np.dot(a_b, a_c)
    k_ab = np.dot(a_b, a_b)
    if k_ac < 0:
        return False
    elif k_ac > k_ab:
        return False
    else:
        return True


def get_points_of_loop(loop):
    """
    Get the [x, z] points corresponding to this loop. If the loop is closed then skips
    the last (closing) point.

    Parameters
    ----------
    loop: Loop
        Loop to get the points of

    Returns
    -------
    points : List[float, float]
        The [x, z] points corresponding to this loop.

    Notes
    -----
        Deprecation / portover utility
    """
    if loop.closed:
        return loop.d2.T[:-1].tolist()
    else:
        return loop.d2.T.tolist()


def index_of_point_on_loop(loop, point_on_loop, before=True):
    """
    Return the index of the point on the given loop belonging to a
    pair which form a linesegment that intersects the given point.
    Raises a GeometryError if given point does not intersect the loop.

    Parameters
    ----------
    loop : Loop
        Loop on with which the point should intersect
    point_on_loop: [ float, float]
        List of x,z coords of point
    before: bool
        If :code:`True`, return the index of the first point in the intersecting
        linesegment on the loop which intersects our point. If :code:`False`,
        return the index of the second in the pair.

    Returns
    -------
    index_of_point: int
        Index of the nearest point on the loop to the given point.
        Either before or after depending on the value of :code:`before` arg.
    """
    # Combine coords into single array, skipping the last if it's a closed loop
    coords = np.array(get_points_of_loop(loop))

    # Get the number of points in the loop
    n_points = coords.shape[0]

    # By creating linesegments from pairs of points along the loop,
    # check which intersect the outer strike point, and create an array of
    # indices corresponding to those which do
    index_of_point = None

    for i in range(n_points):
        i_start = i
        i_end = (i + 1) % n_points
        if check_linesegment(coords[i_start], coords[i_end], point_on_loop):
            if before:
                index_of_point = i_start
            else:
                index_of_point = i_end
            break

    if not index_of_point:
        raise GeometryError("Point is not on loop")

    return index_of_point


def join_intersect(loop1, loop2, get_arg=False):
    """
    Adds the intersection points between Loop1 and Loop2 to Loop1

    Parameters
    ----------
    loop1: BLUEPRINT Loop object
        The Loop to which the intersection points should be added
    loop2: BLUEPRINT Loop object
        The intersecting Loop
    get_arg: bool (default = False)

    Returns
    -------
    (if get_arg is True)
    args: list(int, int, ..) of len(N_intersections)
        The arguments of Loop1 in which the intersections were added.

    Note
    ----
    Modifies loop1
    """
    x_inter, z_inter = get_intersect(loop1, loop2)
    xz = loop1.d2.T
    count = 0
    args = []
    for i, (x, z) in enumerate(zip(x_inter, z_inter)):
        for j, (xx, zz) in enumerate(xz[:-1]):
            if check_linesegment((xx, zz), xz[j + 1], (x, z)):
                args.append(j)
                break

    orderr = np.array(args).argsort()
    x_int = x_inter[orderr]
    z_int = z_inter[orderr]
    for i, (x, z) in enumerate(zip(x_int, z_int)):
        for j, (xx, zz) in enumerate(xz[:-1]):
            if check_linesegment((xx, zz), xz[j + 1], (x, z)):
                arg = j + count + 1
                loop1.insert((x, z), pos=arg)
                count += 1
                # Stop looking (can only be one)
                # Occasional duplicates will happen when the intersection lies
                # on a point of the Loop
                break

    if get_arg:
        args = []
        for x, z in zip(x_inter, z_inter):
            args.append(loop1.argmin([x, z]))
        return list(set(args))


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def line_crossing(a, b, c, d):
    """
    Determines whether there is a line segment crossing between A-B and C-D
    """
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def lineq(point_a, point_b, show=False):
    """
    Find the equation of a line based on two points

    Parameters
    ----------
    point_a: (float, float)
        The coordinates of the firct point on the line
    point_b: (float, float)
        The coordinates of the second point on the line
    show: bool (default = False)
        Prints output to console

    Returns
    -------
    m: float
        The slope of the line
    c: float
        The intercept of the line
    """
    try:
        m = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
    except ZeroDivisionError:
        m = 0
    c = point_b[1] - m * point_b[0]
    if show is True:
        bluemira_print("Equation of line: y = {0}x + {1}".format(m, c))
    return m, c


def check_ccw(x, z):
    """
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


@nb.jit(cache=True, forceobj=True)
def on_polygon(x, z, poly):
    """
    Determines if a point (x, z) is on the perimeter of a closed 2-D polygon

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
        c = check_linesegment(point_a, point_b, [x, z])

        if c is True:
            return True
    return on_edge


@nb.jit(cache=True, nopython=True)
def in_polygon(x, z, poly, include_edges=False):
    """
    Determines if a point (x, z) is inside a 2-D polygon

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
    Determines what points of a polygon are inside another polygon

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


# # TODO: test/compare speed with ray-tracer algos also found here
def inloop(x_loop, z_loop, x, z, side="in"):
    """
    Determine whether a set of points are within a loop.
    """
    if side == "in":
        sign = 1
    elif side == "out":
        sign = -1
    else:
        raise ValueError("define side, 'in' or 'out'")
    x_loop, z_loop = clock(x_loop, z_loop)
    n_rloop, n_zloop = normal(x_loop, z_loop)
    x_in, z_in = np.array([]), np.array([])
    if isinstance(x, Iterable):
        for x_i, z_i in zip(x, z):
            i = np.argmin((x_i - x_loop) ** 2 + (z_i - z_loop) ** 2)
            dx = [x_loop[i] - x_i, z_loop[i] - z_i]
            dn = [n_rloop[i], n_zloop[i]]
            if sign * np.dot(dx, dn) > 0:
                x_in, z_in = np.append(x_in, x_i), np.append(z_in, z_i)
        return x_in, z_in
    else:
        i = np.argmin((x - x_loop) ** 2 + (z - z_loop) ** 2)
        dx = [x_loop[i] - x, z_loop[i] - z]
        dn = [n_rloop[i], n_zloop[i]]
        return sign * np.dot(dx, dn) > 0


def polyarea(x, y, d3=None):
    """
    Returns the area inside a closed polygon with x, y coordinate vectors.
    `Link Shoelace method <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    Parameters
    ----------
    x, y: list or np.array
        The sets of coordinates [m]
    d3: list or np.array or None
        The third set of coordinates or None (for a 2-D polygon)

    Returns
    -------
    area: float
        The area of the polygon [m^2]
    """
    # TODO: catch edge case of n_hat with 3 points in a line
    if d3 is not None:
        p1 = np.array([x[0], y[0], d3[0]])
        p2 = np.array([x[1], y[1], d3[1]])
        p3 = np.array([x[2], y[2], d3[2]])
        v1, v2 = p3 - p1, p2 - p1
        v3 = np.cross(v1, v2)
        v3 = v3 / np.linalg.norm(v3)
        a = np.zeros(3)
        m = np.array([x, y, d3])
        for i in range(len(d3)):
            a += np.cross(m[:, i], m[:, (i + 1) % len(d3)])
        a /= 2
        area = abs(np.dot(a, v3))
    else:
        if len(x) != len(y):
            raise GeometryError("Coordinate vectors must have same length.")
        area = np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
    return area


def get_centroid(x, z, output_area=False):
    """
    Calculates the centroid of a non-self-intersecting counterclockwise polygon

    Parameters
    ----------
    x: np.array
        x coordinates of the loop to calculate on
    z: np.array
        z coordinates of the loop to calculate on
    output_area: bool (default = False)
        Whether or not to also return the area of the loop

    Returns
    -------
    cx: float
        The x coordinate of the centroid [m]
    cz: float
        The z coordinate of the centroid [m]
    area: float (optional output)
        The area of the loop [m^2]
    """
    if not check_ccw(x, z):
        x = x[::-1]
        z = z[::-1]
    area = polyarea(x, z)

    cx, cz = 0, 0
    for i in range(len(x) - 1):
        a = x[i] * z[i + 1] - x[i + 1] * z[i]
        cx += (x[i] + x[i + 1]) * a
        cz += (z[i] + z[i + 1]) * a

    if area != 0:
        # Zero division protection
        cx /= 6 * area
        cz /= 6 * area

    if output_area:
        return cx, cz, area
    else:
        return cx, cz


def get_centroid_3d(x, y, z):
    """
    Calculates the centroid of a non-self-intersecting counterclockwise polygon
    in 3-D.

    Parameters
    ----------
    x, y, z: np.array(N), np.array(N), np.array(N)
        Coordinates of the loop to calculate on

    Returns
    -------
    cx, cy, cz: float, float
        The x, y, z coordinates of the centroid [m]
    """
    cx, cy = get_centroid(x, y)
    cx2, cz = get_centroid(x, z)
    cy2, cz2 = get_centroid(y, z)

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


def loop_volume(x, z):
    """
    Calculates the volume of a loop about axis ([0, 0, 0], [0, 0, 1])

    Parameters
    ----------
    x, z: np.array(N), np.array(N)
        Coordinates of the loop [m]

    Returns
    -------
    volume: float
        The volume of the loop rotated about the z axis [m^3]

    \t:math:`V=2\\pi\\overline{x}A`
    """
    x, z, area = get_centroid(x, z, output_area=True)
    return 2 * np.pi * x * area


def loop_surface(x, z):
    """
    Calculates the surface area of a loop rotated about axis
    ([0, 0, 0], [0, 0, 1])

    Parameters
    ----------
    x: array_like
        x coordinates of the loop [m]
    z: array_like
        z coordinates of the loop [m]

    Returns
    -------
    surface: float
        The surface of the loop rotated about the z axis [m^2]

    \t:math:`S=2\\pi\\overline{x}P`
    """
    x_c, _ = get_centroid(x, z)
    return 2 * np.pi * x_c * perimeter(x, z)


def get_dl(x, z):
    """
    Returns the length of each individual segment in a set of coordinates

    Parameters
    ----------
    x: array_like
        x coordinates of the loop [m]
    z: array_like
        z coordinates of the loop [m]

    Returns
    -------
    dL: np.array(N)
        The length of each individual segment in the loop
    """
    return np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)


def perimeter(x, z):
    """
    Returns the perimeter of an X, Z loop

    Parameters
    ----------
    x: array_like
        x coordinates of the loop [m]
    z: array_like
        z coordinates of the loop [m]

    Returns
    -------
    P: float
        The total perimeter of the loop
    """
    return np.sum(get_dl(x, z))


def length(x, z):
    """
    Return a 1-D parameterisation of an X, Z loop.

    Parameters
    ----------
    x: array_like
        x coordinates of the loop [m]
    z: array_like
        z coordinates of the loop [m]

    Returns
    -------
    lengt: np.array(N)
        The cumulative length of each individual segment in the loop
    """
    lengt = np.append(0, np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)))
    return lengt


def lengthnorm(x, z):
    """
    Return a normalised 1-D parameterisation of an X, Z loop.

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
    total_length = length(x, z)
    return total_length / total_length[-1]


def vector_lengthnorm(xyz):
    """
    Return a normalised 1-D parameterisation of an X, Y, Z loop.

    Parameters
    ----------
    xyz: np.array(n, 3)
        The 3-D coordinate matrix

    Returns
    -------
    length_: np.array(n)
        The normalised length vector
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    length_ = np.append(
        0,
        np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)),
    )
    return length_ / length_[-1]


def tangent(x, z):
    """
    Returns tangent vectors along an anticlockwise X, Z loop
    """
    d_x, d_z = np.gradient(x), np.gradient(z)
    mag = np.sqrt(d_x ** 2 + d_z ** 2)
    index = mag > 0
    d_x, d_z, mag = d_x[index], d_z[index], mag[index]  # clear duplicates
    t_x, t_z = d_x / mag, d_z / mag
    return t_x, t_z


def normal(x, z):
    """
    Returns normal vectors along a set of points (anti-clockwise)
    """
    x, z = np.array(x), np.array(z)
    t_x, t_z = tangent(x, z)
    t = np.zeros((len(t_x), 3))
    t[:, 0], t[:, 1] = t_x, t_z
    n = np.cross(t, [0, 0, 1])
    n_r, n_z = n[:, 0], n[:, 1]
    return n_r, n_z


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


def vector_intersect(p1, p2, p3, p4):
    """

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


def unique(x, z):
    """
    Removes duplicates
    """
    length_n = lengthnorm(x, z)
    io = np.append(np.diff(length_n) > 0, True)
    return x[io], z[io], length_n[io]


# =============================================================================
# Rotation algorithms
# =============================================================================


def rotate_vector_2d(vector, theta):
    """
    Rotates a 2-D vector in 2 dimensions by an angle theta

    Parameters
    ----------
    vector: np.array(2)
        vector to be rotated
    theta: float
        Rotation angle [rad]

    Returns
    -------
    Vhat: np.array(2)
        Rotated vector
    """
    v_mag = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
    r_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    r_matrix = r_matrix[:2, :2]
    v_hat = np.dot(r_matrix, vector) / v_mag
    return v_hat


def qrotate(point, **kwargs):
    """
    Rotate a point cloud by angle theta around vector (right-hand coordinates).
    Uses black quarternion magic.

    Parameters
    ----------
    point: np.array(N, 3) or dict('x': [], 'y': [], 'z': [])
        The coordinates of the points to be rotated
    kwargs:
        theta: float
            Rotation angle [radians]
        p1: [float, float, float]
            Origin of rotation vector
        p2: [float, float, float]
            Second point defining rotation axis
    OR kwargs:
        theta: float
            Rotation angle [radians]
        xo: [float, float, float]
            Origin of rotation vector
        dx: [float, float, float] or one of 'x', 'y', 'z'
            Direction vector definition rotation axis from origin. If a string
            is specified the dx vector is automatically calculated, e.g.
            'z': (0, 0, 1)
    OR kwargs:
        quart: Quarternion object
            The rotation quarternion
        xo: [float, float, float]
            Origin of rotation vector

    Returns
    -------
    rpoint: np.array(N, 3) or dict('x': [], 'y': [], 'z': [])
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


def rotate_matrix(theta, axis="z"):
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
        # All das hier lass ich rein, damit alle verstehen koennen, dass es
        # sich um normalen Rotation Matrices handelt.
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
        # Mignon, mais difficile de savoir ce qui se passe
        axis = np.array(axis) / np.linalg.norm(axis)  # Unit vector
        cos = np.cos(theta)
        sin = np.sin(theta)
        x, y, z = axis
        u_x = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        u_o_u = np.outer(axis, axis)
        r_matrix = cos * np.eye(3) + sin * u_x + (1 - cos) * u_o_u
    return r_matrix


# =============================================================================
# Ordering algorithms
#     TODO: sort these out... subtleties important
# =============================================================================
def anticlock(x, z):
    """
    Orders an x, z list of coordinates anticlockwise about centre
    """
    xm, zm = (np.mean(x), np.mean(z))
    radius = ((x - xm) ** 2 + (z - zm) ** 2) ** 0.5
    theta = np.arctan2(z - zm, x - xm)
    index = theta.argsort()[::-1]
    radius, theta = radius[index], theta[index]
    x, z = xm + radius * np.cos(theta), zm + radius * np.sin(theta)
    return x, z


def clock(x, z, reverse=True):
    """
    Order loop anti-clockwise with spline smoothing.
    """
    # Circular import
    from bluemira.geometry._deprecated_tools import innocent_smoothie

    rc, zc = (np.mean(x), np.mean(z))
    radius = ((x - rc) ** 2 + (z - zc) ** 2) ** 0.5
    theta = np.arctan2(z - zc, x - rc)
    index = theta.argsort()[::-1]
    radius, theta = radius[index], theta[index]
    x, z = rc + radius * np.cos(theta), zc + radius * np.sin(theta)
    x, z = np.append(x, x[0]), np.append(z, z[0])
    x, z = innocent_smoothie(x, z, n=len(x) - 1)
    if reverse:
        x, z = x[::-1], z[::-1]
    return x, z


def order(x, z, anti=True):
    """
    Order points anti-clockwise.
    """
    rc, zc = (np.mean(x), np.mean(z))
    theta = np.unwrap(np.arctan2(z - zc, x - rc))
    if theta[-1] < theta[0]:  # NOTE: theta[-2] and theta[-1] causes differences!
        x, z = x[::-1], z[::-1]
    if not anti:
        x, z = x[::-1], z[::-1]
    return x, z


def theta_sort(x, z, origin="lfs", **kwargs):
    """
    Sort x, z based on angle.
    """
    xo = kwargs.get("xo", (np.mean(x), np.mean(z)))
    anti = kwargs.get("anti", True)  # changed from False
    if origin == "lfs":
        theta = np.arctan2(z - xo[1], x - xo[0])
    elif origin == "top":
        theta = np.arctan2(xo[0] - x, z - xo[1])
    elif origin == "bottom":
        theta = np.arctan2(x - xo[0], xo[1] - z)
    if kwargs.get("unwrap", False):
        theta = np.unwrap(theta)
    index = np.argsort(theta)
    x, z = x[index], z[index]
    if not anti:
        x, z = x[::-1], z[::-1]
    return x, z


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
        The angle between the vector [degrees]
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

    return sign * np.degrees(angle)


def circle_arc(p, h=(0, 0), angle=90, npoints=200):
    """
    Crée un arc de cercle a partir du point p, au centre (h, k) et d'arc alpha.

    Parameters
    ----------
    p: (float, float)
        The starting point of the circle arc to draw
    h: (float, float)
        The coordinates of the centre of the circle
    angle: float
        The revolution angle of the circle segment to draw
    npoints: int
        The number of points to draw in the circle segment

    Returns
    -------
    x, y: np.array(npoints), np.array(npoints)
        The coordinates of the circle segment
    """
    dx = p[0] - h[0]
    dz = p[1] - h[1]
    start_angle = np.arctan2(dz, dx)
    r = distance_between_points(p, h)
    start_alpha = np.rad2deg(start_angle)
    return circle_seg(r, h=h, angle=angle, npoints=npoints, start=start_alpha)


def circle_seg(r, h=(0, 0), angle=360, npoints=500, **kwargs):
    """
    Crée un arc de cercle de radius r au centre (h, k) et d'arc alpha.
    Résolution de 500 points

    Parameters
    ----------
    r: float
        The radius of the circle segment to draw
    h: (float, float)
        The coordinates of the centre of the circle
    angle: float
        The revolution angle of the circle segment to draw
    npoints: int
        The number of points to draw in the circle segment
    kwargs:
        'start': float
            The starting angle of the circle segment. Will default to equi-
            spacing around 0 degrees

    Returns
    -------
    x: np.array(npoints)
        The x coordinates of the circle segment
    y: np.array(npoints)
        The y coordinates of the circle segment
    """
    alpha_start = kwargs.get("start", -angle / 2)
    alpha_start = np.deg2rad(alpha_start)
    alpha = np.deg2rad(angle)
    n = np.linspace(alpha_start, alpha_start + alpha, npoints)
    x = r * np.cos(n) + h[0]
    y = r * np.sin(n) + h[1]
    if angle == 360:
        # Small number error correction (close circle exactly)
        x[-1] = x[0]
        y[-1] = y[0]
    return x, y


def rainbow_arc(p_i, p_o, h=(0, 0), angle=360, npoints=200):
    """
    Arc-en-ciel

    Parameters
    ----------
    p_i: (float, float)
        The inner arc start point
    p_o: (float, float)
        The outer arc start point
    h: (float, float)
        The centre of the arc
    angle: float
        The rotation angle [degrees]
    npoints: int
        The number of points in the rainbow arc

    Returns
    -------
    x: np.array(npoints)
        The x coordinates of the rainbow arc
    y: np.array(npoints)
        The y coordinates of the rainbow arc
    """
    xi, yi = circle_arc(p_i, h=h, angle=angle, npoints=npoints // 2)
    xo, yo = circle_arc(p_o, h=h, angle=angle, npoints=npoints // 2)
    x = np.append(xi, xo[::-1])
    x = np.append(x, xi[0])
    y = np.append(yi, yo[::-1])
    y = np.append(y, yi[0])
    return x, y


def rainbow_seg(ri, ro, h=(0, 0), angle=360, npoints=200):
    """
    Cree un arco iris con ri y ro, de centro h(0, 0)

    Parameters
    ----------
    ri: float
        Inner radius of the rainbow segment
    ro: float
        Outer radius of the rainbow segment
    h: (float, float)
        Centre of the rainbow
    angle: float
        The angle of rotation [degrees]
    npoints: int
        The number of points in the rainbow segment

    Returns
    -------
    x: np.array(npoints)
        The x coordinates of the rainbow segment
    y: np.array(npoints)
        The y coordinates of the rainbow segment
    """
    xi, yi = circle_seg(ri, h=h, angle=angle, npoints=npoints // 2)
    xo, yo = circle_seg(ro, h=h, angle=angle, npoints=npoints // 2)
    x = np.append(xi, xo[::-1])
    x = np.append(x, xi[0])
    y = np.append(yi, yo[::-1])
    y = np.append(y, yi[0])
    # La flemme de faire qqchose qui marche partout
    return x, y


def xz_interp(x, z, npoints=500, ends=True):
    """
    Returns npoints interpolation of X, Z coordinates
    """
    l_vector = lengthnorm(x, z)
    l_interp = np.linspace(0, 1, npoints, endpoint=ends)
    x = interp1d(l_vector, x)(l_interp)
    z = interp1d(l_vector, z)(l_interp)
    return x, z


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
    elif loop.__class__.__name__ == "Shell":
        return _montecarloshellcontrol(loop)
    else:
        raise ValueError("Was zur Hoelle ist hier los?!")


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
    raise ValueError("Da musst du was Besseres scheiben...")


def _montecarloshellcontrol(shell):
    """
    Find an arbitrary point inside a Shell

    Uses brute force (for now)...

    Parameters
    ----------
    shell: Shell
        The geometry to find a control point for. Must be 2-D.

    Returns
    -------
    float, float
        An arbitrary control point for the Shell.
    """
    outer = shell.outer
    xmin, xmax = np.min(outer.d2[0]), np.max(outer.d2[0])
    dx = xmax - xmin
    ymin, ymax = np.min(outer.d2[1]), np.max(outer.d2[1])
    dy = ymax - ymin
    i = 0
    while i < 1000:
        i += 1
        n, m = np.random.rand(2)
        x = xmin + n * dx
        y = ymin + m * dy
        if shell.point_inside([x, y]):
            return [x, y]
    raise ValueError("Da musst du was Besseres schreiben...")


def _points_beyond_length(point1, point2, min_length):
    """
    Determine whether the two points are separated by at least `min_length`

    Parameters
    ----------
    point1 : (float, float)
        The first point to be used in the comparison.
    point2 : (float, float)
        The second point to be used in the comparison.
    min_length : float
        The minimum length [m] by which the two points should be separated.

    Returns
    -------
    bool
        True if the `point1` and `point2` are separated by at least `min_length`,
        else False.
    """
    distance = distance_between_points(point1, point2)
    return distance > min_length or np.isclose(distance, min_length)


def _points_beyond_angle(point1, point2, point3, min_angle):
    """
    Determine whether three points sit on a curve subtending at least `min_angle`

    A straight line is defined between `point1` and `point2` and the angle is measured
    to `point3` from that straight line. As such, if the three points lie on a straight
    line then the angle will be zero.

    Parameters
    ----------
    point1 : (float, float)
        The first point to be used in the comparison.
    point2 : (float, float)
        The second point to be used in the comparison, sitting between `point1` and
        `point3`.
    point3 : (float, float)
        The third point to be used in the comparison.
    min_angle : float
        The angle [°] from which the `point3` should at least lie off of the line
        defined by `point1` and `point2`.

    Returns
    -------
    bool
        True if `point3` lies at least `min_angle` from the line defined by `point1` and
        `point2`, else False.
    """
    angle = get_angle_between_points(point1, point2, point3)
    target_angle = 180.0 - min_angle
    return angle < target_angle or np.isclose(angle, target_angle)


def clean_loop_points(loop, min_length=None, min_angle=None):
    """
    Remove points from loop that are closer than `min_length` and within `min_angle`

    The points on the 2D loop are scanned for any cases where points are within the
    defined `min_length` and do not subtend an angle larger than `min_angle`. Any such
    cases are removed from the points. As the points on the loop are scanned
    sequentially, it is assumed that the points form a continous clockwise or
    counter-clockwise loop.

    If either `min_length` or `min_angle` are not defined then the corresponding cleaning
    logic will not be used.

    Any duplicate points are always removed from the loop.

    Parameters
    ----------
    loop : Loop
        The loop from which to extract the cleaned points.
    min_length : float, optional
        The minimum length [m] by which any two points should be separated,
        by default None.
    min_angle : float, optional
        The angle [°] from which the `point3` should lie off of the line defined by
        `point1` and `point2`, by default None.

    Returns
    -------
    List[float, float]
        The remaining points after the cleaning algorithm has been applied.
    """

    def use_point(point1, point2, point3, min_length, min_angle):
        if np.allclose(point1, point2):
            return False

        if min_angle is None and min_length is None:
            return True
        elif min_angle is None:
            return _points_beyond_length(point1, point2, min_length)
        elif min_length is None:
            return _points_beyond_angle(point1, point2, point3, min_angle)
        else:
            return _points_beyond_length(
                point1, point2, min_length
            ) or _points_beyond_angle(point1, point2, point3, min_angle)

    # Always remove duplicate points in case the loop is closed
    _, unique_idx = np.unique(loop.d2.T, axis=0, return_index=True)
    unique_points = loop.d2.T[np.sort(unique_idx)].tolist()

    if min_length is None and min_angle is None:
        # Avoid duplicate points even if whole loop
        points = unique_points
    else:
        # Scan the points to find those that should be used
        points = []
        current_point = None
        for i, point in enumerate(unique_points):
            if current_point is None:
                points += [point]
                current_point = point
            else:
                next_point = unique_points[(i + 1) % len(unique_points)]
                if use_point(current_point, point, next_point, min_length, min_angle):
                    points += [point]
                    current_point = point
    return points


def get_boundary(polygons):
    """
    Get the boundary of the union of polygons

    Constructs a union on the polygons and generates the points and facets corresponding
    to that union.

    Parameters
    ----------
    polygons : List[shapely.geometry.Polygon]
        The polygons to find the boundary of.

    Returns
    -------
    bounary_points : List[float, float]
        The points defining the boundary of the union of polygons.
    boundary_facets : List[int, int]
        The facets defining the boundary of the union of polygons.
    """

    def get_points(coords):
        return [list(coord) for coord in coords[:-1]]

    def get_facets(coords, start):
        num_points = len(get_points(coords))
        facets = [[i, i + 1] for i in range(start, start + num_points - 1)]
        facets += [[start + num_points - 1, start]]
        return facets

    boundary_points = []
    boundary_facets = []

    union = unary_union(polygons)
    if isinstance(union, MultiPolygon):
        for polygon in MultiPolygon(union):
            if isinstance(polygon.boundary, MultiLineString):
                for boundary in polygon.boundary:
                    boundary_facets += get_facets(boundary.coords, len(boundary_points))
                    boundary_points += get_points(boundary.coords)
            else:
                boundary_facets += get_facets(
                    polygon.boundary.coords, len(boundary_points)
                )
                boundary_points += get_points(polygon.boundary.coords)
    else:
        if isinstance(union.boundary, MultiLineString):
            for boundary in union.boundary:
                boundary_facets += get_facets(boundary.coords, len(boundary_points))
                boundary_points += get_points(boundary.coords)
        else:
            boundary_facets += get_facets(union.boundary.coords, len(boundary_points))
            boundary_points += get_points(union.boundary.coords)
    return boundary_points, boundary_facets


if __name__ == "__main__":
    from BLUEPRINT import test

    test()

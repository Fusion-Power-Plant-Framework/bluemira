# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Utility for sets of coordinates
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import Enum, auto
from itertools import count, starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numba as nb
import numpy as np
from numba.np.extensions import cross2d
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.constants import CROSS_P_TOL, DOT_P_TOL
from bluemira.geometry.error import CoordinatesError
from bluemira.utilities.tools import json_writer

if TYPE_CHECKING:
    import numpy.typing as npt

    from bluemira.geometry.plane import BluemiraPlane


DIM = 3


# =============================================================================
# Pre-processing utilities
# =============================================================================
class RotationAxis(Enum):
    """Enumeration of rotation axes."""

    X = auto()
    Y = auto()
    Z = auto()

    @classmethod
    def _missing_(cls, value: str | RotationAxis) -> RotationAxis:
        try:
            return cls[value.upper()]
        except KeyError:
            raise CoordinatesError(
                f"Invalid rotation axis: {value}. Choose from: {(*cls._member_names_,)}"
            ) from None


def xyz_process(func):
    """
    Decorator for parsing x, y, z coordinates to numpy float arrays and dimension
    checking.

    Returns
    -------
    :
        Decorator for parsing x, y, z coordinates to numpy float arrays and dimension
        checking.
    """

    def wrapper(x, y, z=None):
        _validate_coordinates(x, y, z)
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if z is not None:
            z = np.ascontiguousarray(z, dtype=np.float64)

        return func(x, y, z)

    return wrapper


def _validate_coordinates(x, y, z=None):
    if z is None:
        if len(x) != len(y):
            raise CoordinatesError(
                "All coordinates must have the same length but "
                f"got len(x) = {len(x)}, len(y) = {len(y)}"
            )
    elif not len(x) == len(y) == len(z):
        raise CoordinatesError(
            "All coordinates must have the same length but "
            f"got len(x) = {len(x)}, len(y) = {len(y)}, len(z) = {len(z)}"
        )


# =============================================================================
# Tools and calculations for sets of coordinates
# =============================================================================


def vector_lengthnorm(
    x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None
) -> np.ndarray:
    """
    Get a normalised 1-D parameterisation of a set of x-y(-z) coordinates.

    Parameters
    ----------
    x:
        The x coordinates
    y:
        The y coordinates
    z:
        The z coordinates. If None, carries out the operation in 2-D

    Returns
    -------
    The normalised length vector

    Notes
    -----
    The normalized length vector is:

    .. math::
        \\text{Normalized Length} = \\frac{L}{L[-1]}

    where

    .. math::
        L = \\sum_{i=0}^{n-1} \\sqrt{(\\Delta x_i)^2 + (\\Delta y_i)^2 + (\\Delta z_i)^2}

    Where :math:`\\Delta x_i`, :math:`\\Delta y_i`, and :math:`\\Delta z_i` are the
    finite differences along the x, y, and z dimensions, respectively for the i-th
    index. n is the length of the array of coordinates.
    """
    coords = [x, y] if z is None else [x, y, z]
    dl_vectors = np.sqrt(np.sum([np.diff(ci) ** 2 for ci in coords], axis=0))
    length_ = np.append(0, np.cumsum(dl_vectors))
    return length_ / length_[-1]


def interpolate_points(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, n_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate points.

    Parameters
    ----------
    x:
        The x coordinates
    y:
        The y coordinates
    z:
        The z coordinates
    n_points:
        number of points

    Returns
    -------
    x:
        The interpolated x coordinates
    y:
        The interpolated y coordinates
    z:
        The interpolated z coordinates
    """
    ll = vector_lengthnorm(x, y, z)
    linterp = np.linspace(0, 1, int(n_points))
    x = interp1d(ll, x)(linterp)
    y = interp1d(ll, y)(linterp)
    z = interp1d(ll, z)(linterp)
    return x, y, z


def interpolate_midpoints(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate the points adding the midpoint of each segment to the points.

    Parameters
    ----------
    x:
        The x coordinates
    y:
        The y coordinates
    z:
        The z coordinates

    Returns
    -------
    x:
        The interpolated x coordinates
    y:
        The interpolated y coordinates
    z:
        The interpolated z coordinates
    """
    xyz = np.c_[x, y, z]
    xyz_new = xyz[:, :-1] + np.diff(xyz) / 2
    xyz_new = np.insert(xyz_new, np.arange(len(x) - 1), xyz[:, :-1], axis=1)
    xyz_new = np.append(xyz_new, xyz[:, -1].reshape(3, 1), axis=1)
    return xyz_new[0], xyz_new[1], xyz_new[2]


@nb.jit(cache=True, nopython=True)
def get_normal_vector(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Calculate the normal vector from a series of planar points.

    Parameters
    ----------
    x:
        The x coordinates
    y:
        The y coordinates
    z:
        The z coordinates

    Returns
    -------
    The normalised normal vector

    Raises
    ------
    CoordinatesError
        Cannot find normal vector or arrays not of equal length >= 3
    """
    if len(x) < DIM:
        raise CoordinatesError(
            "Cannot get a normal vector for a set of points with length less than 3."
        )

    if not (len(x) == len(y) == len(z)):
        raise CoordinatesError("Point coordinate vectors must be of equal length.")

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
        raise CoordinatesError("Unable to find a normal vector from set of points.")

    return n_hat / np.linalg.norm(n_hat)


@xyz_process
def get_perimeter(x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None) -> float:
    """
    Calculate the perimeter of a set of coordinates.

    Parameters
    ----------
    x:
        The x coordinates
    y:
        The y coordinates
    z:
        The z coordinates

    Returns
    -------
    The perimeter of the coordinates
    """
    if z is None:
        return get_perimeter_2d(x, y)
    return get_perimeter_3d(x, y, z)


@nb.jit(cache=True, nopython=True)
def get_perimeter_2d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the perimeter of a 2-D set of coordinates.

    Parameters
    ----------
    x:
        The x coordinates
    y:
        The y coordinates

    Returns
    -------
    The perimeter of the coordinates
    """
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    return np.sum(np.sqrt(dx**2 + dy**2))


@nb.jit(cache=True, nopython=True)
def get_perimeter_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Calculate the perimeter of a set of 3-D coordinates.

    Parameters
    ----------
    x:
        The x coordinates
    y:
        The y coordinates
    z:
        The z coordinates

    Returns
    -------
    The perimeter of the coordinates
    """
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))


@xyz_process
def get_area(x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None) -> float:
    """
    Calculate the area inside a closed polygon with x, y coordinate vectors.
    `Link Shoelace method <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    Parameters
    ----------
    x:
        The first set of coordinates [m]
    y:
        The second set of coordinates [m]
    z:
        The third set of coordinates or None (for a 2-D polygon)

    Returns
    -------
    The area of the polygon [m^2]
    """
    if z is None:
        return get_area_2d(x, y)
    return get_area_3d(x, y, z)


@nb.jit(cache=True, nopython=True)
def get_area_2d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the area inside a closed polygon with x, y coordinate vectors.
    `Link Shoelace method <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    Parameters
    ----------
    x:
        The first set of coordinates [m]
    y:
        The second set of coordinates [m]

    Returns
    -------
    The area of the polygon [m^2]
    """
    # No np.roll in numba
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    x1 = np.append(x[-1], x[:-1])
    y1 = np.append(y[-1], y[:-1])
    return 0.5 * np.abs(np.dot(x, y1) - np.dot(y, x1))


@nb.jit(cache=True, nopython=True)
def get_area_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Calculate the area inside a closed polygon.
    `Link Shoelace method <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    Parameters
    ----------
    x:
        The first set of coordinates [m]
    y:
        The second set of coordinates [m]
    z:
        The third set of coordinates [m]

    Returns
    -------
    The area of the polygon [m^2]
    """
    if np.all(x == x[0]) and np.all(y == y[0]) and np.all(z == z[0]):
        # Basically a point, but avoid getting the normal vector..
        return 0

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


@nb.jit(cache=True, nopython=True)
def check_ccw(x: np.ndarray, z: np.ndarray) -> bool:
    """
    Check that a set of x, z coordinates are counter-clockwise.

    Parameters
    ----------
    x:
        The x coordinates of the polygon
    z:
        The z coordinates of the polygon

    Returns
    -------
    True if polygon counterclockwise
    """
    a = 0
    for n in range(len(x) - 1):
        a += (x[n + 1] - x[n]) * (z[n + 1] + z[n])
    return a < 0


def check_ccw_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, normal: np.ndarray
) -> bool:
    """
    Check if a set of coordinates is counter-clockwise w.r.t a normal vector.

    Parameters
    ----------
    x:
        The first set of coordinates [m]
    y:
        The second set of coordinates [m]
    z:
        The third set of coordinates [m]
    normal:
        The normal vector about which to check for CCW

    Returns
    -------
    Whether or not the set is CCW about the normal vector
    """
    # Translate to centroid
    dx, dy, dz = get_centroid_3d(x, y, z)
    x, y, z = x - dx, y - dy, z - dz
    # Rotate to x-y plane
    r = rotation_matrix_v1v2([0, 0, 1], normal)
    x, y, z = r.T @ np.array([x, y, z])
    # Check projected x-y is CCW
    return check_ccw(x, y)


@xyz_process
def get_centroid(
    x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None
) -> np.ndarray:
    """
    Calculate the centroid of a non-self-intersecting 2-D counter-clockwise polygon.

    Parameters
    ----------
    x:
        x coordinates of the coordinates to calculate on
    y:
        y coordinates of the coordinates to calculate on
    z:
        z coordinates of the coordinates to calculate on

    Returns
    -------
    The x, y, [z] coordinates of the centroid [m]
    """
    if z is None:
        return get_centroid_2d(x, y)
    return get_centroid_3d(x, y, z)


@nb.jit(cache=True, nopython=True)
def get_centroid_2d(x: np.ndarray, z: np.ndarray) -> list[float]:
    """
    Calculate the centroid of a non-self-intersecting 2-D counter-clockwise polygon.

    Parameters
    ----------
    x:
        x coordinates of the coordinates to calculate on
    z:
        z coordinates of the coordinates to calculate on

    Returns
    -------
    The x, z coordinates of the centroid [m]
    """
    if not check_ccw(x, z):
        x = np.ascontiguousarray(x[::-1])
        z = np.ascontiguousarray(z[::-1])
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


def get_centroid_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float]:
    """
    Calculate the centroid of a non-self-intersecting counterclockwise polygon
    in 3-D.

    Parameters
    ----------
    x:
        The x coordinates
    y:
        The y coordinates
    z:
        The z coordinates

    Returns
    -------
    The x, y, z coordinates of the centroid [m]
    """
    cx, cy = get_centroid_2d(x, y)
    if np.allclose(z, z[0]):
        return [cx, cy, z[0]]
    cx2, cz = get_centroid_2d(x, z)
    if np.allclose(y, y[0]):
        return [cx2, y[0], cz]
    cy2, cz2 = get_centroid_2d(y, z)
    if np.allclose(x, x[0]):
        return [x[0], cy2, cz2]

    # The following is an "elegant" but computationally more expensive way of
    # dealing with the 0-area edge cases
    # (of which there are more than you think)
    cx = np.array([cx, cx2])
    cy = np.array([cy, cy2])
    cz = np.array([cz, cz2])

    def get_rational(i, array):
        """
        Gets rid of infinity and nan coordinates

        Returns
        -------
        :
            Array without infinity and nan coordinates.
        """
        args = np.argwhere(np.isfinite(array))
        if len(args) == 0:
            # 2-D shape with a simple axis offset
            # Get the first value of the coordinate set which is equal to the
            # offset
            return [x, y, z][i][0]
        if len(args) == 1:
            return array[args[0][0]]
        if all(np.isclose(array, 0)):
            return 0
        if any(np.isclose(array, 0)):
            # Occasionally the two c values are not the same, and one is 0
            return array[np.argmax(np.abs(array))]
        return array[0]

    return list(starmap(get_rational, enumerate([cx, cy, cz])))


def get_angle_between_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Angle between points. P1 is vertex of angle. ONly tested in 2d

    Returns
    -------
    :
        The angle between points.
    """
    if not all(isinstance(p, np.ndarray) for p in [p0, p1, p2]):
        p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
    ba = p0 - p1
    bc = p2 - p1
    return get_angle_between_vectors(ba, bc)


def get_angle_between_vectors(
    v1: np.ndarray, v2: np.ndarray, *, signed: bool = False
) -> float:
    """
    Angle between vectors. Will return the signed angle if specified.

    Parameters
    ----------
    v1:
        The first vector
    v2:
        The second vector
    signed:
        Whether or not to calculate the signed angle

    Returns
    -------
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
        # Vectors parallel
        sign = 1 if det == 0 else np.sign(det)

    return sign * angle


# =============================================================================
# Rotations
# =============================================================================


def rotation_matrix(
    theta: float, axis: str | RotationAxis | np.ndarray = RotationAxis.Z
) -> np.ndarray:
    """
    Old-fashioned rotation matrix: :math:`\\mathbf{R_{u}}(\\theta)`
    \t:math:`\\mathbf{x^{'}}=\\mathbf{R_{u}}(\\theta)\\mathbf{x}`

    \t:math:`\\mathbf{R_{u}}(\\theta)=cos(\\theta)\\mathbf{I}+sin(\\theta)[\\mathbf{u}]_{\\times}(1-cos(\\theta))(\\mathbf{u}\\otimes\\mathbf{u})`

    Parameters
    ----------
    theta:
        The rotation angle [radians] (counter-clockwise about axis!)
    axis:
        The rotation axis (specified by axis label or vector)

    Returns
    -------
    The (active) rotation matrix about the axis for an angle theta
    """
    if isinstance(axis, str | RotationAxis):
        axis = RotationAxis(axis)
        # I'm leaving all this in here, because it is easier to understand
        # what is going on, and that these are just "normal" rotation matrices
        if axis is RotationAxis.Z:
            r_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ])
        elif axis is RotationAxis.Y:
            r_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ])
        elif axis is RotationAxis.X:
            r_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ])
        else:
            raise NotImplementedError
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


def rotation_matrix_v1v2(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Get a rotation matrix based off two vectors.

    Returns
    -------
    :
        A roational matrix based off two vectors.
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


def project_point_axis(point: npt.ArrayLike, axis: npt.ArrayLike) -> np.ndarray:
    """
    Project a 3-D point onto a 3-D axis.
    \t:math:`\\mathbf{p_{proj}} = \\dfrac{\\mathbf{p}\\cdot\\mathbf{a}}{\\mathbf{a}\\cdot\\mathbf{a}}\\mathbf{a}`

    Parameters
    ----------
    point:
        The point to project onto the axis
    axis:
        The axis onto which to project the point

    Returns
    -------
    The coordinates of the projected point
    """  # noqa: W505, E501
    point = np.array(point)
    axis = np.array(axis)
    return axis * np.dot(point, axis) / np.dot(axis, axis)


def principal_components(xyz_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Principal component analysis.

    Returns
    -------
    :
        Eigenvalues and eigenvectors or an xyz array.
    """
    mean = np.mean(xyz_array, axis=1)
    xyz_shift = xyz_array - mean.reshape((3, 1))

    cov = np.cov(xyz_shift)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors


# =============================================================================
# Boolean checks
# =============================================================================


@nb.jit(cache=True, nopython=True)
def check_linesegment(
    point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray
) -> bool:
    """
    Check that point C is on the line between points A and B.

    Parameters
    ----------
    point_a:
        The first line segment 2-D point
    point_b:
        The second line segment 2-D point
    point_c:
        The 2-D point which to check is on A--B

    Returns
    -------
    True: if C on A--B, else False
    """
    # Do some protection of numba against integers and lists
    a_c = np.array([point_c[0] - point_a[0], point_c[1] - point_a[1]], dtype=np.float64)
    a_b = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]], dtype=np.float64)

    distance = np.sqrt(np.sum(a_b**2))
    # Numba doesn't like doing cross-products of things with size 2
    cross = cross2d(a_b, a_c)
    if np.abs(cross) > CROSS_P_TOL * distance:
        return False
    k_ac = np.dot(a_b, a_c)
    k_ab = np.dot(a_b, a_b)
    if k_ac < 0:
        return False
    return k_ac <= k_ab


@nb.jit(cache=True, nopython=True)
def in_polygon(
    x: float,
    z: float,
    poly: np.ndarray,
    include_edges: bool = False,  # noqa: FBT001, FBT002
) -> bool:
    """
    Determine if a point (x, z) is inside a 2-D polygon.

    Parameters
    ----------
    x:
        Point x coordinate
    z:
        Point z coordinate
    poly:
        The 2-D array of polygon point coordinates
    include_edges:
        Whether or not to return True if a point is on the perimeter of the
        polygon

    Returns
    -------
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

        if z > min(z0, z1) and z <= max(z0, z1) and x <= max(x0, x1):
            if z0 != z1:
                x_inter = (z - z0) * (x1 - x0) / (z1 - z0) + x0
                if x == x_inter:
                    return include_edges
            if x0 == x1 or x <= x_inter:
                inside = not inside  # Beautiful
        elif z == min(z0, z1) and z0 == z1 and (x <= max(x0, x1)) and (x >= min(x0, x1)):
            return include_edges

        x0, z0 = x1, z1
    return inside


@nb.jit(cache=True, nopython=True)
def polygon_in_polygon(
    poly1: np.ndarray,
    poly2: np.ndarray,
    include_edges: bool = False,  # noqa: FBT001, FBT002
) -> np.ndarray:
    """
    Determine what points of a 2-D polygon are inside another 2-D polygon.

    Parameters
    ----------
    poly1:
        The array of 2-D polygon1 point coordinates
    poly2:
        The array of 2-D polygon2 point coordinates
    include_edges:
        Whether or not to return True if a point is on the perimeter of the
        polygon

    Returns
    -------
    The array of boolean values per index of polygon1
    """
    inside_array = np.empty(len(poly1), dtype=np.bool_)
    for i in range(len(poly1)):
        inside_array[i] = in_polygon(
            poly1[i][0], poly1[i][1], poly2, include_edges=include_edges
        )
    return inside_array


@nb.jit(nopython=True)
def on_polygon(x: float, z: float, poly: npt.NDArray[np.float64]) -> bool:
    """
    Determine if a point (x, z) is on the perimeter of a closed 2-D polygon.

    Parameters
    ----------
    x:
        Point x coordinate
    z:
        Point z coordinate
    poly:
        The array of 2-D polygon point coordinates

    Returns
    -------
    Whether or not the point is on the perimeter of the polygon
    """
    xz = np.array([x, z], dtype=nb.float64)
    for ind in range(poly.shape[0] - 1):
        c = check_linesegment(poly[ind], poly[ind + 1], xz)

        if c is True:
            return True
    return False


def normal_vector(side_vectors: np.ndarray) -> np.ndarray:
    """
    Find the anti-clockwise normal vector to the given side vectors.

    Parameters
    ----------
    side_vectors:
        The side vectors of a polygon (shape: (N, 2)).

    Returns
    -------
    The array of 2-D normal vectors of each side of a polygon
    (shape: (2, N)).

    Notes
    -----
    The normal vector `a` is calculated using the formula:

    .. math::

            \\mathbf{a} = -\\frac{-[\\mathbf{v}[1],~
            \\mathbf{v}[0]]}{\\sqrt{\\mathbf{v}[0]^2~
            + \\mathbf{v}[1]^2}}

    where :math:`\\mathbf{v}` are the side vectors.
    """
    a = -np.array([-side_vectors[1], side_vectors[0]]) / np.sqrt(
        side_vectors[0] ** 2 + side_vectors[1] ** 2
    )
    nan = np.isnan(a)
    a[nan] = 0
    return a


def vector_intersect(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> np.ndarray:
    """
    Get the intersection point between two 2-D vectors.

    Parameters
    ----------
    p1:
        The first point on the first vector (shape: (2,)).
    p2:
        The second point on the first vector (shape: (2,)).
    p3:
        The first point on the second vector (shape: (2,)).
    p4:
        The second point on the second vector (shape: (2,)).

    Returns
    -------
    The point of the intersection between the two vectors (shape: (2,)).

    Notes
    -----
    If the vectors are parallel:
        - The vectors do not intersect.
        - The function returns p2

    Otherwise:
        - Calculates the intersection point using vector algebra:

        .. math::

            \\text{point} = \\frac{
                \\lVert \\mathbf{p2} - \\mathbf{p1} \\rVert~
                \\cdot (\\mathbf{p1} - \\mathbf{p3})
            }{
            \\lVert \\mathbf{p2} - \\mathbf{p1} \\rVert~
            \\cdot (\\mathbf{p4} - \\mathbf{p3})
            } (\\mathbf{p4} - \\mathbf{p3}) + \\mathbf{p3}
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


def get_bisection_line(
    p1: npt.NDArray[float],
    p2: npt.NDArray[float],
    p3: npt.NDArray[float],
    p4: npt.NDArray[float],
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Find the bisection line between two lines.

    Parameters
    ----------
    p1:
        The first point on the first vector (shape: (2,)).
    p2:
        The second point on the first vector (shape: (2,)).
    p3:
        The first point on the second vector (shape: (2,)).
    p4:
        The second point on the second vector (shape: (2,)).

    Returns
    -------
    origin:
        A point on that bisection line. (shape: (2,))
    direction:
        A normal vector that the bisection line points in (shape: (2,))

    Notes
    -----
    The intersection point is calculated using vector algebra, and the
    direction is the normalized sum of the normalized vectors of da and db.

    """
    origin = vector_intersect(p1, p2, p3, p4)
    da = p2 - p1
    db = p4 - p3
    normed_da = da / np.linalg.norm(da)
    normed_db = db / np.linalg.norm(db)
    dc = normed_da + normed_db
    direction = dc / np.linalg.norm(dc)
    return origin, direction


# =============================================================================
# Coordinate class and parsers
# =============================================================================


def _parse_to_xyz_array(
    xyz_array: npt.ArrayLike | dict[str, npt.ArrayLike],
) -> npt.NDArray:
    """
    Make a 3, N xyz array out of just about anything.

    Raises
    ------
    CoordinatesError
        Cannot instantiate coordinates

    Returns
    -------
    :
        A 3, N xyz array.
    """
    if isinstance(xyz_array, np.ndarray):
        xyz_array = _parse_array(xyz_array)
    elif isinstance(xyz_array, dict):
        xyz_array = _parse_dict(xyz_array)
    elif isinstance(xyz_array, Iterable):
        # We temporarily set the dtype to object to avoid a VisibleDeprecationWarning
        xyz_array = _parse_array(np.array(xyz_array, dtype=object))
    else:
        raise CoordinatesError(f"Cannot instantiate Coordinates with: {type(xyz_array)}")
    return xyz_array


def _parse_array(xyz_array: npt.ArrayLike):
    try:
        xyz_array = np.atleast_2d(np.squeeze(np.array(xyz_array, dtype=np.float64)))
    except ValueError as ve:
        raise CoordinatesError(
            "Cannot instantiate Coordinates with a ragged (3, N | M) array."
        ) from ve

    shape = xyz_array.shape
    if len(shape) > 2:  # noqa: PLR2004
        raise NotImplementedError

    n, m = shape
    if n == DIM:
        if m == DIM:
            bluemira_warn(
                "You are creating Coordinates with a (3, 3) array, defaulting to (3, N)."
            )

    elif m == DIM:
        xyz_array = xyz_array.T

    else:
        raise CoordinatesError(
            "Cannot instantiate Coordinates where either n or m != 3."
        )

    if not np.allclose([len(xyz_array[i]) for i in range(3)], len(xyz_array[0])):
        raise CoordinatesError(
            "Cannot instantiate Coordinates with a ragged (3, N | M) array."
        )

    return xyz_array


def _parse_dict(xyz_dict):
    x = np.atleast_1d(xyz_dict.get("x", 0))
    y = np.atleast_1d(xyz_dict.get("y", 0))
    z = np.atleast_1d(xyz_dict.get("z", 0))

    shape_lengths = np.array([len(c.shape) for c in [x, y, z]])

    if np.any(shape_lengths > 1):
        raise CoordinatesError(
            "Cannot instantiate Coordinates from dict with coordinate vectors that are"
            " not 1-D."
        )

    lengths = [len(c) for c in [x, y, z]]
    if np.all(np.array(lengths) <= 1):
        # Vertex detected
        return np.array([x, y, z])

    usable_lengths = [length for length in lengths if length != 1]

    if not np.allclose(usable_lengths, usable_lengths[0]):
        raise CoordinatesError(
            "Cannot instantiate Coordinate from dict with a ragged set of vectors."
        )

    # Backfill single-value coordinates
    actual_length = usable_lengths[0]
    if len(x) == 1:
        x = x[0] * np.ones(actual_length)
    if len(y) == 1:
        y = y[0] * np.ones(actual_length)
    if len(z) == 1:
        z = z[0] * np.ones(actual_length)

    return np.array([x, y, z])


class Coordinates:
    """
    Coordinates object for storing ordered sets of coordinates.

    An array shape of (3, N) is enforced.

    Counter-clockwise direction can be set relative to a normal vector.

    Notes
    -----
    This is a utility class for dealing with sets of coordinates in a number of different
    contexts. It should not be used for the creation of CAD geometries.

    If a 3 x 3 array is provided it is assumed that the first axis is xyz and the second
    is the coordinate, this will output a warning to notify users.
    """

    __slots__ = ("_array", "_is_planar", "_normal_vector")
    # =============================================================================
    # Instantiation
    # =============================================================================

    def __init__(self, xyz_array: npt.ArrayLike | dict[str, npt.ArrayLike]):
        self._array = _parse_to_xyz_array(xyz_array)
        self._is_planar = None
        self._normal_vector = None

    @classmethod
    def from_json(cls, filename: str) -> Coordinates:
        """
        Load a Coordinates object from a JSON file.

        Parameters
        ----------
        filename:
            Full path file name of the data

        Raises
        ------
        CoordinatesError
            Cannot read json file

        Returns
        -------
        :
            Coordinate object.
        """
        try:
            with open(filename) as data:
                xyz_dict = json.load(data)
        except json.JSONDecodeError:
            raise CoordinatesError(
                f"Could not read the file: {filename}\n Please ensure it is a JSON file."
            ) from None

        # NOTE: Stabler than **xyz_dict
        x = xyz_dict.get("x", 0)
        y = xyz_dict.get("y", 0)
        z = xyz_dict.get("z", 0)
        return cls({"x": x, "y": y, "z": z})

    # =============================================================================
    # Checks
    # =============================================================================

    def _set_plane_props(self):
        """
        Set the planar properties of the Coordinates.
        """
        if self._is_planar is None and self._normal_vector is None:
            self._update_plane_props()

    def _update_plane_props(self):
        if len(self) > DIM:
            eigenvalues, eigenvectors = principal_components(self._array)

            self._is_planar = np.isclose(eigenvalues[-1], 0.0)
            self._normal_vector = eigenvectors[:, -1]
        else:
            bluemira_warn("Cannot set planar properties on Coordinates with length < 3.")
            self._is_planar = False
            self._normal_vector = None

    @property
    def is_planar(self) -> bool:
        """
        Whether or not the Coordinates are planar.
        """
        self._set_plane_props()
        return self._is_planar

    @property
    def normal_vector(self) -> np.ndarray:
        """
        The normal vector of the best-fit plane of the Coordinates.
        """
        self._set_plane_props()
        return self._normal_vector

    def check_ccw(self, axis: np.ndarray | None = None) -> bool:
        """
        Whether or not the Coordinates are ordered in the counter-clockwise direction
        about a specified axis. If None is specified, the Coordinates normal vector will
        be used.

        Raises
        ------
        CoordinatesError
            axis must be of size 3

        Returns
        -------
        :
            The check for whether the Coordinates are ordered in counter-clockwise or
            not.
        """
        if len(self) < DIM:
            return False

        if axis is None:
            axis = self.normal_vector
        else:
            axis = np.array(axis, dtype=float)
            if axis.size != DIM:
                raise CoordinatesError("Base vector must be of size 3.")
            axis /= np.linalg.norm(axis)

        return check_ccw_3d(self.x, self.y, self.z, axis)

    def set_ccw(self, axis: np.ndarray | None = None):
        """
        Set the Coordinates to be counter-clockwise about a specified axis. If None is
        specified, the Coordinates normal vector will be used.
        """
        if len(self) < DIM:
            bluemira_warn("Cannot set Coordinates of length < 3 to CCW.")
            return

        if not self.check_ccw(axis=axis):
            self.reverse()

    def distance_to(self, point: np.ndarray) -> np.ndarray:
        """
        Calculates the distances from each point in the Coordinates to the point.

        Parameters
        ----------
        point:
            The point (3-D) to which to calculate the distances

        Returns
        -------
        The vector of distances of the Coordinates to the point

        Notes
        -----
        Euclidean distance:

        .. math::
            d = \\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}

        where indices refer to the Coordinate and the given point.

        """
        point = np.array(point)
        point = point.reshape(3, 1).T
        return cdist(self.xyz.T, point, "euclidean")

    def argmin(self, point: np.ndarray) -> int:
        """
        Parameters
        ----------
        point:
            The 3-D point to which to calculate the distances

        Returns
        -------
        The index of the closest point
        """
        return np.argmin(self.distance_to(point))

    # =============================================================================
    # Property access
    # =============================================================================

    @property
    def x(self) -> np.ndarray:
        """
        The x coordinate vector
        """
        return self._array[0]

    @property
    def y(self) -> np.ndarray:
        """
        The y coordinate vector
        """
        return self._array[1]

    @property
    def z(self) -> np.ndarray:
        """
        The z coordinate vector
        """
        return self._array[2]

    @property
    def xy(self) -> np.ndarray:
        """
        The x-y coordinate array
        """
        return self._array[[0, 1], :]

    @property
    def xz(self) -> np.ndarray:
        """
        The x-z coordinate array
        """
        return self._array[[0, 2], :]

    @property
    def yz(self) -> np.ndarray:
        """
        The y-z coordinate array
        """
        return self._array[[1, 2], :]

    @property
    def xyz(self) -> np.ndarray:
        """
        The x-y-z coordinate array
        """
        return self._array

    @property
    def points(self) -> list[np.ndarray]:
        """
        A list of the individual points of the Coordinates.
        """
        return list(self.T)

    # =========================================================================
    # Conversions
    # =========================================================================

    def as_dict(self) -> dict[str, np.ndarray]:
        """
        Cast the Coordinates as a dictionary.

        Returns
        -------
        d: dict
            Dictionary with {'x': [], 'y': [], 'z':[]}
        """
        return {"x": self.x, "y": self.y, "z": self.z}

    def to_json(self, filename: str, **kwargs: dict[str, Any]) -> str:
        """
        Save the Coordinates as a JSON file.

        Returns
        -------
        :
            The Coordinates as a JSON file.
        """
        return json_writer(
            self.as_dict(), Path(filename).with_suffix("").with_suffix(".json"), **kwargs
        )

    # =============================================================================
    # Useful properties
    # =============================================================================

    @property
    def closed(self) -> bool:
        """
        Whether or not this is a closed set of Coordinates.
        """
        return len(self) > 2 and np.allclose(  # noqa: PLR2004
            self[:, 0], self[:, -1], rtol=EPS, atol=0
        )

    @property
    def length(self) -> float:
        """
        Perimeter length of the coordinates.
        """
        return get_perimeter_3d(*self._array)

    @property
    def center_of_mass(self) -> tuple[float, float, float]:
        """
        Geometrical centroid of the Coordinates.
        """
        # [sic] coordinates do not have a "mass", but named such for consistency with
        # other geometry objects.
        if len(self) == 1:
            return self.xyz.T[0]

        if len(self) == 2:  # noqa: PLR2004
            return np.average(self.xyz.T)

        return tuple(get_centroid_3d(*self._array))

    # =============================================================================
    # Array-like behaviour
    # =============================================================================

    @property
    def T(self) -> np.ndarray:  # noqa: N802
        """
        Transpose of the Coordinates
        """
        return self._array.T

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of the Coordinates
        """
        return self._array.shape

    # =============================================================================
    # Modification
    # =============================================================================
    def reverse(self):
        """
        Reverse the direction of the Coordinates.
        """
        self._array = self._array[:, ::-1]

    def open(self):
        """
        Open the Coordinates (if they are closed)
        """
        if len(self) < DIM:
            bluemira_warn(f"Cannot open Coordinates of length {len(self)}")
            return

        if self.closed:
            self._array = self._array[:, :-1]

    def insert(self, point: np.ndarray, index: int = 0):
        """
        Insert a point to the Coordinates.

        Parameters
        ----------
        point:
            The 3-D point to insert into the Coordinates
        index:
            The position of the point in the Coordinates (order index)
        """
        if index > len(self):
            bluemira_warn(
                "Inserting a point in Coordinates at an index greater than the number of"
                " points."
            )
            index = -1
        if not np.isclose(self.xyz.T, point).all(axis=1).any():
            point = np.array(point).reshape((3, 1))
            if index == -1:
                self._array = np.hstack((self._array, point))
            else:
                self._array = np.hstack((
                    self._array[:, :index],
                    point,
                    self._array[:, index:],
                ))

    def close(self):
        """
        Close the Coordinates (if they are open)
        """
        if len(self) < DIM:
            bluemira_warn(f"Cannot close Coordinates of length {len(self)}")
            return

        if not self.closed:
            self._array = np.vstack((self._array.T, self._array[:, 0])).T

    def rotate(
        self,
        base: tuple[float, float, float] = (0, 0, 0),
        direction: tuple[float, float, float] = (0, 0, 1),
        degree: float = 0.0,
    ):
        """
        Rotate the Coordinates.

        Parameters
        ----------
        base:
            Origin location of the rotation
        direction:
            The direction vector
        degree:
            rotation angle [degrees]

        Raises
        ------
        CoordinatesError
            Base and direction must be of size 3
        """
        if degree == 0.0:
            return

        base = np.array(base, dtype=float)
        if base.size != DIM:
            raise CoordinatesError("Base vector must be of size 3.")

        direction = np.array(direction, dtype=float)
        if direction.size != DIM:
            raise CoordinatesError("Direction vector must be of size 3.")
        direction /= np.linalg.norm(direction)

        points = self._array - base.reshape(DIM, 1)

        r_matrix = Rotation.from_rotvec(np.deg2rad(degree) * direction).as_matrix()
        new_array = points.T @ r_matrix.T + base
        self._array = new_array.T

        self._update_plane_props()

    def translate(self, vector: tuple[float, float, float] = (0, 0, 0)):
        """
        Translate this shape with the vector. This function modifies the self
        object.

        Raises
        ------
        CoordinatesError
            vector must be of size 3
        """
        vector = np.array(vector)
        if vector.size != DIM:
            raise CoordinatesError("Translation vector must be of size 3.")

        self._array += vector.reshape(3, 1)
        self._update_plane_props()

    def simplify(
        self,
        max_angle: float,
        dx_min: float,
        dx_max: float = np.inf,
    ) -> tuple[Coordinates, npt.NDArray[np.int64]]:
        """
        Generate a set of pivot points along the given boundary.

        Given a set of boundary points, some maximum angle, and minimum and
        maximum segment length, this function derives a set of pivot points
        along the boundary, that define a 'string'. You might picture a
        'string' as a thread wrapped around some nails (pivot points) on a
        board.

        Parameters
        ----------
        angle: float
            Maximum turning angle [degree]
        dx_min: float
            Minimum segment length
        dx_max: float
            Maximum segment length

        Returns
        -------
        new_points:
            The pivot points' coordinates.
        index:
            The indices of the pivot points into the input points.

        Raises
        ------
        ValueError
            dx_min > dx_maz
        """
        if dx_min > dx_max:
            raise ValueError(
                f"'dx_min' cannot be greater than 'dx_max': '{dx_min} > {dx_max}'"
            )
        points = self._array.copy().T

        if len(points) < 3:  # noqa: PLR2004
            return Coordinates(points), np.array([], dtype=int)

        t_vector = points[1:] - points[:-1]  # tangent vector
        t_vec_norm = np.linalg.norm(t_vector, axis=1)
        t_vec_norm[t_vec_norm == 0] = 1e-36  # protect zero division
        median_dt = np.median(t_vec_norm)  # average step length
        t_vector /= t_vec_norm.reshape(-1, 1) * np.ones((1, np.shape(t_vector)[1]))

        new_points = np.zeros_like(points)
        index = np.zeros(len(points), dtype=int)

        delta_x, delta_turn = np.zeros((2, len(points)))
        t0, p0 = t_vector[0], points[0]

        new_points[0] = p0
        angle_crit = np.sin(max_angle * np.pi / 180)

        k = count(1)
        for i, (p, t) in enumerate(zip(points[1:], t_vector, strict=False)):
            c_mag = np.linalg.norm(np.cross(t0, t))
            dx = np.linalg.norm(p - p0)  # segment length
            if (c_mag > angle_crit and dx > dx_min) or dx + median_dt > dx_max:
                j = next(k)
                new_points[j] = points[i]  # pivot point
                index[j] = i + 1  # pivot index

                delta_x[j - 1] = dx  # panel length
                delta_turn[j - 1] = np.arcsin(c_mag) * 180 / np.pi
                t0, p0 = t, p  # update

        if dx > dx_min:
            j = next(k)
            delta_x[j - 1] = dx  # last segment length
        else:
            delta_x[j - 1] += dx  # last segment length
        new_points[j] = p  # replace / append last point
        new_points = new_points[: j + 1]  # trim

        index[j] = i + 1  # replace/append last point index
        index = index[: j + 1]  # trim

        return Coordinates(new_points), index

    # =============================================================================
    # Dunders (with different behaviour to array)
    # =============================================================================

    def __eq__(self, other: Coordinates) -> bool:
        """
        Check the Coordinates for equality with other Coordinates.

        Parameters
        ----------
        other:
            The other Coordinates to compare against

        Returns
        -------
        :
            The check of the Coordinates for equality with other Coordinates.

        Notes
        -----
        Coordinates with identical coordinates but different orderings will not be
        counted as identical.
        """
        if isinstance(other, type(self)):
            return np.allclose(self._array, other._array, rtol=EPS, atol=0)
        return False

    def __hash__(self):
        """Hash of Coordinates

        Returns
        -------
        :
            The hash of Coordinates.
        """
        return hash((self._array, self._is_planar, self._normal_vector))

    def __len__(self) -> int:
        """
        The number of points in the Coordinates.

        Returns
        -------
        :
            The number of points in the Coordinates.
        """
        return self.shape[1]

    # =============================================================================
    # Array-like dunders
    # =============================================================================

    def __repr__(self) -> str:
        """
        Representation of the Coordinates.

        Returns
        -------
        :
            Representation of the Coordinates.
        """
        r = repr(self._array)
        return f"{self.__class__.__name__}{r[5:]}"

    def __getitem__(self, *args, **kwargs):
        """
        Array-like indexing and slicing.

        Returns
        -------
        :
            Indexed or sliced array.
        """
        return self._array.__getitem__(*args, **kwargs)

    def __iter__(self):
        """
        Array-like unpacking.

        Returns
        -------
        :
            Unpacked array.
        """
        return iter(self._array)


# =============================================================================
# Intersection tools
# =============================================================================


def vector_intersect_3d(
    p_1: np.ndarray, p_2: np.ndarray, p_3: np.ndarray, p_4: np.ndarray
) -> np.ndarray:
    """
    Get the intersection point between two 3-D vectors.

    Parameters
    ----------
    p1:
        The first point on the first vector
    p2:
        The second point on the first vector
    p3:
        The first point on the second vector
    p4:
        The second point on the second vector

    Returns
    -------
    The point of the intersection between the two vectors

    Raises
    ------
    CoordinatesError
        If there is no intersection between the points

    Notes
    -----
    Credit: Paul Bourke at
    http://paulbourke.net/geometry/pointlineplane/#:~:text=The%20shortest%20line%20between%20two%20lines%20in%203D
    """
    p_13 = p_1 - p_3
    p_43 = p_4 - p_3

    if np.linalg.norm(p_13) < EPS:
        raise CoordinatesError("No intersection between 3-D lines.")
    p_21 = p_2 - p_1
    if np.linalg.norm(p_21) < EPS:
        raise CoordinatesError("No intersection between 3-D lines.")

    d1343 = np.dot(p_13, p_43)
    d4321 = np.dot(p_43, p_21)
    d1321 = np.dot(p_13, p_21)
    d4343 = np.dot(p_43, p_43)
    d2121 = np.dot(p_21, p_21)

    denom = d2121 * d4343 - d4321 * d4321

    if np.abs(denom) < EPS:
        raise CoordinatesError("No intersection between 3-D lines.")

    numer = d1343 * d4321 - d1321 * d4343

    mua = numer / denom
    return p_1 + mua * p_21


def coords_plane_intersect(
    coords: Coordinates, plane: BluemiraPlane
) -> np.ndarray | None:
    """
    Calculate the intersection of Coordinates with a plane.

    Parameters
    ----------
    coords:
        The coordinates to calculate the intersection on
    plane:
        The plane to calculate the intersection with

    Returns
    -------
    The xyz coordinates (3, n_intersections) of the intersections with the Coordinates.
    Returns None if there are no intersections detected
    """
    out = _coords_plane_intersect(coords.xyz.T[:-1], plane.base, plane.axis)
    if not out:
        return None
    return np.unique(out, axis=0)  # Drop occasional duplicates


@nb.jit(cache=True, nopython=True)
def _coords_plane_intersect(
    array: np.ndarray, p1: np.ndarray, vec2: np.ndarray
) -> list[float]:
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


def get_intersect(xy1: np.ndarray, xy2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the intersection points between two sets of 2-D coordinates. Will return
    a unique list of x, z intersections (no duplicates in x-z space).

    Parameters
    ----------
    xy1:
        The 2-D coordinates between which intersection points should be calculated.
        Shape = (2, N)
    xy2:
        The 2-D coordinates between which intersection points should be calculated.
        Shape = (2, N)

    Returns
    -------
    :
        The x, z coordinates of the intersection points. shape = (2, N)

    Notes
    -----
    D. Schwarz, <https://uk.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections>
    """
    x1, y1 = xy1
    x2, y2 = xy2

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
        except np.linalg.LinAlgError:  # noqa: PERF203
            # Parallel segments. Will raise numpy RuntimeWarnings
            xz[0, i] = np.nan
    in_range = (xz[0, :] >= 0) & (xz[1, :] >= 0) & (xz[0, :] <= 1) & (xz[1, :] <= 1)
    xz = xz[2:, in_range].T
    return np.unique(xz, axis=0).T


@nb.jit(cache=True, nopython=True)
def _intersect_count(
    x_inter: np.ndarray, z_inter: np.ndarray, x2: np.ndarray, z2: np.ndarray
) -> np.ndarray:
    """Get the indices of the intersects that are

    Parameters
    ----------
    x_inter, z_inter:
        x and z coordinates of the points created by the get_intersect function.
    x2, z2:
        x and z coordinates of one of the vertices of the polygon inputted into the
        get_intersect function.

    Returns
    -------
    :
        a list of indices j, where the [i]-th intersection point is expected to lie on
        the [j]-th edge.
    """
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


def join_intersect(
    tgt_poly: Coordinates, ref_poly: Coordinates, *, get_arg: bool = False
) -> list[int] | None:
    """
    Add the intersection points between tgt_poly and ref_poly to tgt_poly.

    Parameters
    ----------
    tgt_poly:
        The target polygon's vertices expressed as Coordinates. The intersection
        points should be inserted into this polygon.
    ref_poly:
        The reference polygon's vertices expressed as Coordinates.
    get_arg:
        Whether or not to return the intersection arguments

    Returns
    -------
    The arguments of tgt_poly in which the intersections were added (if get_arg is True)

    Notes
    -----
    Modifies tgt_poly
    """
    xz_inter = get_intersect(tgt_poly.xz, ref_poly.xz)
    args = _intersect_count(*xz_inter.T, *tgt_poly.xz)

    orderr = args.argsort()
    xz_int = xz_inter[orderr]

    args = _intersect_count(*xz_int.T, *tgt_poly.xz)

    # TODO @CoronelBuendia: Check for duplicates and order correctly based on distance
    # 3585
    # u, counts = np.unique(args, return_counts=True)

    count = 0
    for i, arg in enumerate(args):
        # Two intersection points, one after the other
        bump = 0 if i > 0 and args[i - 1] == arg else 1
        if not np.isclose(tgt_poly.xz.T, xz_int[i]).all(axis=1).any():
            # Only increment counter if the intersection isn't already in the Coordinates
            tgt_poly.insert([xz_int[i][0], 0, xz_int[i][1]], index=arg + count + bump)
            count += 1

    if get_arg:
        args = []
        for x, z in zip(xz_inter.T, strict=False):
            args.append(tgt_poly.argmin([x, 0, z]))
        return list(set(args))
    return None


def choose_direction(
    vector: npt.NDArray[float],
    lower_pt: npt.NDArray[float],
    higher_pt: npt.NDArray[float],
):
    """
    Flip the vector to the correct side (multiply by +1 or -1) so that
    when lower_pt is projected onto the vector, it has a smaller value than
    when higher_pt is projected onto the vector.

    Returns
    -------
    :
        The flipped vector.
    """
    if (vector @ lower_pt) > (vector @ higher_pt):
        return -vector
    return vector

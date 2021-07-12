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
FE transformation matrices and methods
"""
import numpy as np
from bluemira.geometry._deprecated_tools import rotation_matrix
from bluemira.structural.constants import FLOAT_TYPE


def _nb_isclose(a, b, rtol=1.0e-5, atol=0.0e-8):
    """
    Enable future compatibility with numba (0.46).

    Was found in general to slow test run speed... remains to be seen if
    it will have the same effect on real problems, but it's likely
    """
    return abs(a - b) <= atol + rtol * abs(b)


def _direction_cosine_matrix(dx, dy, dz):
    """
    Calculates the direction cosine (transformation) matrix for an arbitrary
    vector in space. A number of painful edge cases are handled...

    Works such that the output dcm satisfies the following properties:

    local = dcm @ global
    global = dcm.T @ local

    Parameters
    ----------
    dx: float
        The absolute length of the vector in the x global coordinate
    dy: float
        The absolute length of the vector in the y global coordinate
    dz: float
        The absolute length of the vector in the z global coordinate

    Returns
    -------
    dcm: np.array((3, 3))
        The direction cosine matrix with reference to the global coordinate
        system
    """
    # Please be careful when modifing this function, and instead work on the
    # debugging version which is tested...
    length = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    a = dx / length
    b = dy / length
    c = dz / length
    d = (a ** 2 + b ** 2) ** 0.5
    # TODO: Why does the less intuitive, simpler algebra form work better??
    # https://ocw.mit.edu/courses/civil-and-environmental-engineering/1-571-structural-analysis-and-control-spring-2004/readings/connor_ch5.pdf
    if np.isclose(a, 0) and np.isclose(b, 0):
        dcm = np.array(
            [[0.0, 0.0, -np.sign(c)], [0.0, 1.0, 0.0], [np.sign(c), 0.0, 0.0]]
        )
    else:
        dcm = np.array([[a, -b / d, -a * c / d], [b, a / d, -b * c / d], [c, 0, d]]).T

    return dcm

    u = np.array([dx, dy, dz], dtype=FLOAT_TYPE)
    x_local = u / np.linalg.norm(u)  # The local unit vector in global coords

    # Global coordinate system
    x_global = np.array([1.0, 0.0, 0.0])
    y_global = np.array([0.0, 1.0, 0.0])
    z_global = np.array([0.0, 0.0, 1.0])

    # Handle edge cases which can cause NaN's later..
    if x_local[0] == 1:
        # This is just the global coordinate system!
        return np.eye(3)

    if x_local[0] == -1:
        # The mirrored global coordinate system!
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=FLOAT_TYPE)

    if _nb_isclose(abs(x_local[1]), 1):
        # Corresponds to a local y-vector sitting on the global x-vector
        # (rotation about z-axis)
        cos_theta = 0.0
        sin_theta = np.sign(x_local[1])  # Determines which way local x points
        return np.array(
            [
                [cos_theta, -sin_theta, 0.0],
                [sin_theta, cos_theta, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=FLOAT_TYPE,
        )

    if _nb_isclose(abs(x_local[2]), 1.0):
        # Corresponds to a local z-vector sitting on the global x-vector
        # (rotation about y-axis)
        cos_theta = 0.0
        sin_theta = -np.sign(x_local[2])
        return np.array(
            [
                [cos_theta, 0.0, sin_theta],
                [0.0, 1.0, 0.0],
                [-sin_theta, 0.0, cos_theta],
            ],
            dtype=FLOAT_TYPE,
        )

    # Build local y and z vectors, following right-handed convention
    y_local = np.array([0.0, 1.0, 0.0], dtype=FLOAT_TYPE)
    y_local -= np.dot(x_local, y_local) * x_local
    y_local /= np.linalg.norm(y_local)
    z_local = np.cross(x_local, y_local)

    # Calculate C_ij cosines
    c11 = np.dot(x_global, x_local)
    c12 = np.dot(x_global, y_local)
    c13 = np.dot(x_global, z_local)
    c21 = np.dot(y_global, x_local)
    c22 = np.dot(y_global, y_local)
    c23 = np.dot(y_global, z_local)
    c31 = np.dot(z_global, x_local)
    c32 = np.dot(z_global, y_local)
    c33 = np.dot(z_global, z_local)
    return np.array([[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]])


def _direction_cosine_matrix_debugging(dx, dy, dz, debug=False):
    """
    Slow, ugly, safe
    """
    dcm = 0
    u = np.array([dx, dy, dz])
    x_local = u / np.linalg.norm(u)

    x_global = np.array([1.0, 0, 0])
    y_global = np.array([0, 1.0, 0])
    z_global = np.array([0, 0, 1.0])
    globa = np.array([x_global, y_global, z_global])

    if x_local[0] == 1:
        # corresponds to the global coordinate system
        dcm = np.eye(3)
        if debug:
            return dcm, globa
        else:
            return dcm
    if x_local[0] == -1:
        # corresponds to the mirrored coordinate system
        dcm = np.eye(3)
        dcm[0, 0] = -1
        local = globa.copy()
        local[0][0] = -1
        if debug:
            return dcm, local
        else:
            return dcm
    if abs(x_local[1]) == 1:
        # corresponds to a local y-vector sitting on the global x-vector
        # (rotation about z-axis)
        cos_theta = 0
        sin_theta = np.sign(x_local[1])
        dcm = np.array(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
        if debug:
            local = np.array([[0, sin_theta, 0], [-sin_theta, 0, 0], [0, 0, 1]])
            return dcm, local
        else:
            return dcm
    if x_local[2] == 1:
        # corresponds to a local z-vector sitting on the global x-vector
        # rotation about the y-axis
        cos_theta = 0
        sin_theta = -1
        dcm = np.array(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )
        if debug:
            local = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            return dcm, local
        else:
            return dcm

    y_local = np.array([0, 1.0, 0])
    y_local -= np.dot(x_local, y_local) * x_local
    y_local /= np.linalg.norm(y_local)
    z_local = np.cross(x_local, y_local)

    local = np.array([x_local, y_local, z_local])

    if not isinstance(dcm, int):
        if debug:
            return dcm, local
        else:
            return dcm
    c11 = np.dot(x_global, x_local)
    c12 = np.dot(x_global, y_local)
    c13 = np.dot(x_global, z_local)
    c21 = np.dot(y_global, x_local)
    c22 = np.dot(y_global, y_local)
    c23 = np.dot(y_global, z_local)
    c31 = np.dot(z_global, x_local)
    c32 = np.dot(z_global, y_local)
    c33 = np.dot(z_global, z_local)
    dcm = np.array([[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]])
    if not debug:
        return dcm
    else:
        return dcm, local


def lambda_matrix(dx, dy, dz):
    """
    3-D transformation matrix, generalised to a 2-node beam.
    Effectively only handles the tiling of the direction cosine matrix
    transform.

    Returns
    -------
    llambda: np.array((12, 12))
        The transformation matrix
    """
    dcm = _direction_cosine_matrix(dx, dy, dz)
    llambda = np.zeros((12, 12))
    llambda[0:3, 0:3] = dcm
    llambda[3:6, 3:6] = dcm
    llambda[6:9, 6:9] = dcm
    llambda[9:12, 9:12] = dcm
    return llambda


def cyclic_pattern(geometry, axis, angle, n, include_first=True):
    """
    Build a cyclic pattern of a Geometry.

    Parameters
    ----------
    geometry: Geometry
        The geometry to pattern
    axis: Iterable(3)
        The axis vector about which to pattern
    angle: float
        The pattern angle [degrees]
    n: int
        The number of sectors to pattern
    include_first: bool
        Whether or not to include the first sector in the result

    Returns
    -------
    patterned: Geometry
        The patterned and merged geometry
    """
    # Dodge cyclic import
    from BLUEPRINT.beams.geometry import Geometry, DeformedGeometry

    if include_first:
        patterned = geometry.copy()
    else:
        if isinstance(geometry, DeformedGeometry):
            patterned = DeformedGeometry(Geometry(), geometry._scale)
        else:
            patterned = Geometry()

    for i in range(1, n):
        sector = geometry.copy()
        theta = np.deg2rad(i * angle)
        t_matrix = rotation_matrix(theta, axis)
        sector.rotate(t_matrix)
        patterned.merge(sector)
        del sector  # Save some RAM

    return patterned


if __name__ == "__main__":
    from bluemira import test

    test()

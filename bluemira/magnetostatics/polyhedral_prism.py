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
Polyhedral prism current source
"""

from typing import Union

import numba as nb
import numpy as np

from bluemira.base.constants import MU_0, MU_0_4PI
from bluemira.magnetostatics.baseclass import PolyhedralCrossSectionCurrentSource
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.tools import process_xyz_array


@nb.jit(nopython=True)
def omega_t(r, r1, r2, r3):
    """
    Solid angle seen from the calculation point subtended by the face
    triangle normal must be pointing outwards from the face
    """
    r1_r = r1 - r
    r2_r = r2 - r
    r3_r = r3 - r
    r1r = np.linalg.norm(r1_r)
    r2r = np.linalg.norm(r2_r)
    r3r = np.linalg.norm(r3_r)
    d = (
        r1r * r2r * r3r
        + r3r * np.dot(r1_r, r2_r)
        + r2r * np.dot(r1_r, r3_r)
        + r1r * np.dot(r2_r, r3_r)
    )
    return 2 * np.arctan2(np.dot(r1_r, np.cross(r2_r, r3_r)), d)


@nb.jit(nopython=True)
def line_integral(r, r1, r2):
    """
    w_e(r)
    """
    r1r = np.linalg.norm(r1 - r)
    r2r = np.linalg.norm(r2 - r)
    r2r1 = np.linalg.norm(r2 - r1)
    a = r2r + r1r + r2r1
    b = r2r + r1r - r2r1
    return np.log(a / b)


@nb.jit(nopython=True)
def get_face_midpoint(face_points):
    """
    Get an arbitrary point on the face
    """
    return np.sum(face_points[:-1], axis=0) / (len(face_points) - 1)


@nb.jit(nopython=True)
def surface_integral(face_points, face_normal, point):
    """
    W_f(r)
    """
    integral = 0.0
    r_f = get_face_midpoint(face_points)
    omega_f = 0.0
    for i in range(len(face_points) - 1):
        p0 = face_points[i]
        p1 = face_points[i + 1]
        r_e = 0.5 * (p0 + p1)
        u_e = p1 - p0
        u_e /= np.linalg.norm(u_e)
        # u_e = -np.cross(face_normal, u_e)
        # u_e /= np.linalg.norm(u_e)
        integral += np.dot(
            np.cross(face_normal, r_e - point),
            u_e * line_integral(point, p0, p1),
        )
        # import matplotlib.pyplot as plt
        # ax = plt.gca()
        # ax.quiver(*r_e, *assert np.allcloseu_e)

        r1, r2, r3 = p0, p1, r_f
        normal = np.cross(r2 - r1, r3 - r1)
        normal /= np.linalg.norm(normal)
        # if not np.allclose(normal, face_normal):
        #     print(normal, face_normal)
        omega_f += omega_t(point, r1, r2, r3)
    return integral - np.dot(r_f - point, face_normal) * omega_f


def vector_potential(rho_vector, face_points, face_normals, point):
    """
    Calculate the vector potential
    """
    integral = np.zeros(3)
    for i, normal in enumerate(face_normals):
        r_f = get_face_midpoint(face_points[i])
        integral += np.dot(
            r_f - point, normal * surface_integral(face_points[i], normal, point)
        )
    return MU_0 / (8 * np.pi) * np.dot(rho_vector, integral)


def field(rho_vector, face_points, face_normals, point):
    """
    Calculate the magnetic field
    """
    field = np.zeros(3)
    for i, normal in enumerate(face_normals):
        field += np.cross(rho_vector, normal) * surface_integral(
            face_points[i], normal, point
        )
    return MU_0_4PI * field


class PolyhedralPrismCurrentSource(PolyhedralCrossSectionCurrentSource):
    """
    3-D trapezoidal prism current source with a polyhedral cross-section and
    uniform current distribution.

    The current direction is along the local y coordinate.

    Parameters
    ----------
    origin:
        The origin of the current source in global coordinates [m]
    ds:
        The direction vector of the current source in global coordinates [m]
    normal:
        The normalised normal vector of the current source in global coordinates [m]
    t_vec:
        The normalised tangent vector of the current source in global coordinates [m]
    breadth:
        The breadth of the current source (half-width) [m]
    depth:
        The depth of the current source (half-height) [m]
    alpha:
        The first angle of the trapezoidal prism [°] [0, 180)
    beta:
        The second angle of the trapezoidal prism [°] [0, 180)
    current:
        The current flowing through the source [A]

    Notes
    -----
    Negative angles are allowed, but both angles must be either 0 or negative.
    """

    def __init__(
        self,
        origin: np.ndarray,
        ds: np.ndarray,
        normal: np.ndarray,
        t_vec: np.ndarray,
        breadth: float,
        depth: float,
        alpha: float,
        beta: float,
        current: float,
    ):
        alpha, beta = np.deg2rad(alpha), np.deg2rad(beta)
        self.origin = origin

        length = np.linalg.norm(ds)
        self._check_angle_values(alpha, beta)
        self._check_raise_self_intersection(length, breadth, alpha, beta)
        self._halflength = 0.5 * length
        # Normalised direction cosine matrix
        self.dcm = np.array([t_vec, ds / length, normal])
        self.length = 0.5 * (length - breadth * np.tan(alpha) - breadth * np.tan(beta))
        self.breadth = breadth
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        # Current density
        self.rho = current / (4 * breadth * depth)
        self.points = self._calculate_points()

    def _check_angle_values(self, alpha, beta):
        """
        Check that end-cap angles are acceptable.
        """
        sign_alpha = np.sign(alpha)
        sign_beta = np.sign(beta)
        one_zero = np.any(np.array([sign_alpha, sign_beta]) == 0.0)  # noqa: PLR2004
        if not one_zero and sign_alpha != sign_beta:
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: end-cap angles "
                f"must have the same sign {alpha=:.3f}, {beta=:.3f}."
            )
        if not (0 <= abs(alpha) < 0.5 * np.pi):
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: {alpha=:.3f} is outside"
                " bounds of [0, 180°)."
            )
        if not (0 <= abs(beta) < 0.5 * np.pi):
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: {beta=:.3f} is outside "
                "bounds of [0, 180°)."
            )

    def _check_raise_self_intersection(
        self, length: float, breadth: float, alpha: float, beta: float
    ):
        """
        Check for bad combinations of source length and end-cap angles.
        """
        a = np.tan(alpha) * breadth
        b = np.tan(beta) * breadth
        if (a + b) > length:
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: source length and "
                "angles imply a self-intersecting trapezoidal prism."
            )

    @process_xyz_array
    def field(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Calculate the magnetic field at a point due to the current source.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the field
        y:
            The y coordinate(s) of the points at which to calculate the field
        z:
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        The magnetic field vector {Bx, By, Bz} in [T]
        """
        point = np.array([x, y, z])
        return field(self.rho * self.dcm[1], self.face_points, self.face_normals, point)

    def _calculate_points(self):
        """
        Calculate extrema points of the current source for plotting and debugging.
        """
        b = self._halflength
        c = self.depth
        d = self.breadth
        # Lower rectangle
        p1 = np.array([-d, -b + d * np.tan(self.beta), -c])
        p2 = np.array([d, -b - d * np.tan(self.beta), -c])
        p3 = np.array([d, -b - d * np.tan(self.beta), c])
        p4 = np.array([-d, -b + d * np.tan(self.beta), c])

        # Upper rectangle
        p5 = np.array([-d, b - d * np.tan(self.alpha), -c])
        p6 = np.array([d, b + d * np.tan(self.alpha), -c])
        p7 = np.array([d, b + d * np.tan(self.alpha), c])
        p8 = np.array([-d, b - d * np.tan(self.alpha), c])

        self.face_points = [
            [p1, p2, p3, p4, p1],
            [p1, p5, p6, p2, p1],
            [p2, p6, p7, p3, p2],
            [p3, p7, p8, p4, p3],
            [p4, p8, p5, p1, p4],
            [p5, p8, p7, p6, p5],
        ]
        self.face_points = [self._local_to_global(p) for p in self.face_points]
        normals = [np.cross(p[1] - p[0], p[2] - p[1]) for p in self.face_points]
        normals = [n / np.linalg.norm(n) for n in normals]
        self.face_normals = normals

        points = [
            np.vstack([p1, p2, p3, p4, p1]),
            np.vstack([p5, p6, p7, p8, p5]),
            # Lines between rectangle corners
            np.vstack([p1, p5]),
            np.vstack([p2, p6]),
            np.vstack([p3, p7]),
            np.vstack([p4, p8]),
        ]

        return np.array([self._local_to_global(p) for p in points], dtype=object)
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
Polyhedral prism current source using the volume integral method

Easiest to follow was M. Fabbri, "Magnetic Flux Density and Vector
Potential of Uniform Polyhedral Sources", IEEE TRANSACTIONS ON MAGNETICS,
VOL. 44, NO. 1, JANUARY 2008

but
"""

from copy import deepcopy
from typing import Union

import numba as nb
import numpy as np

from bluemira.base.constants import MU_0, MU_0_4PI
from bluemira.geometry.coordinates import Coordinates, get_area_2d
from bluemira.magnetostatics.baseclass import (
    PolyhedralCrossSectionCurrentSource,
    PrismEndCapMixin,
)
from bluemira.magnetostatics.tools import process_xyz_array

ZERO_DIV_GUARD_EPS = 1e-14


@nb.jit(nopython=True, cache=True)
def vector_norm_eps(r: np.ndarray) -> float:
    """
    Dodge singularities in omega_t and line_integral when field point
    lies on an edge.

    Introduces an error which is negligible provided the volume is small
    the error is only introduced in a cylindrical volume of radius EPS**1/2
    around the edge.

    Parameters
    ----------
    r:
        Vector coordinates (3)

    Returns
    -------
    Vector norm

    Notes
    -----
    \t:math:`\\lvert \\mathbf{r}\\rvert = \\sqrt{\\lvert \\mathbf{r}\\rvert^2+\\epsilon^2}`
    """  # noqa: W505 E501
    r_norm = np.linalg.norm(r)
    return np.sqrt(r_norm**2 + ZERO_DIV_GUARD_EPS)


@nb.jit(nopython=True, cache=True)
def omega_t(r: np.ndarray, r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> float:
    """
    Solid angle seen from the calculation point subtended by the face
    triangle normal must be pointing outwards from the face

    Parameters
    ----------
    r:
        Point at which the field is being calculated
    r1:
        First point of the triangle
    r2:
        Second point of the triangle
    r3: float
        Third point of the triangle

    Returns
    -------
    Solid angle [rad]

    Notes
    -----
    \t:math:`\\Omega_{T} = 2\\textrm{arctan}\\dfrac{(\\mathbf{r_{1}}-\\mathbf{r})\\cdot(\\mathbf{r_{2}}-\\mathbf{r})\\times(\\mathbf{r_{3}}-\\mathbf{r})}{D}`

    with:

    \t:math:`D=\\lvert \\mathbf{r_1}-\\mathbf{r}\\rvert\\lvert\\mathbf{r_2}-\\mathbf{r} \\rvert\\lvert \\mathbf{r_3} - \\mathbf{r}\\rvert + \\lvert \\mathbf{r_3} - \\mathbf{r}\\rvert (\\mathbf{r_1}-\\mathbf{r}) \\cdot (\\mathbf{r_2} - \\mathbf{r}) + \\lvert\\mathbf{r_2} - \\mathbf{r} \\rvert (\\mathbf{r_1}-\\mathbf{r})\\cdot(\\mathbf{r_3}-\\mathbf{r}) + \\lvert \\mathbf{r_1} - \\mathbf{r}\\rvert (\\mathbf{r_2} - \\mathbf{r}) \\cdot (\\mathbf{r_3} - \\mathbf{r})`

    noting that the normal vector (outwards from the triangle) is:

    \t:math:`\\mathbf{n_T} = \\dfrac{(\\mathbf{r_2} - \\mathbf{r_1}) \\times (\\mathbf{r_3} - \\mathbf{r_1})}{\\lvert (\\mathbf{r_2} - \\mathbf{r_1}) \\times (\\mathbf{r_3} - \\mathbf{r_1}) \\rvert}`
    """  # noqa: W505 E501
    r1_r = r1 - r
    r2_r = r2 - r
    r3_r = r3 - r
    r1r = vector_norm_eps(r1_r)
    r2r = vector_norm_eps(r2_r)
    r3r = vector_norm_eps(r3_r)
    d = (
        r1r * r2r * r3r
        + r3r * np.dot(r1_r, r2_r)
        + r2r * np.dot(r1_r, r3_r)
        + r1r * np.dot(r2_r, r3_r)
    )
    a = np.dot(r1_r, np.cross(r2_r, r3_r))
    # Not sure this is an issue...
    # if abs(a) < ZERO_GUARD_EPS and (-ZERO_GUARD_EPS < d < 0):
    #     return 0 # and not pi as per IEEE
    return 2 * np.arctan2(a, d)


@nb.jit(nopython=True, cache=True)
def edge_integral(r: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> float:
    """
    Evaluate the edge integral w_e(r) of the W function at a point

    Parameters
    ----------
    r:
        Point at which the calculation is being performed
    r1:
        First point of the edge
    r2:
        Second point of the edge

    Returns
    -------
    Value of the edge integral w_e(r)

    Notes
    -----
    \t:math:`w_{e}(\\mathbf{r}) = \\textrm{ln}\\dfrac{\\lvert \\mathbf{r_2} - \\mathbf{r} \\rvert + \\lvert\\mathbf{r_1} - \\mathbf{r} \\rvert + \\lvert \\mathbf{r_2} - \\mathbf{r_1} \\rvert}{\\lvert \\mathbf{r_2} - \\mathbf{r} \\rvert + \\lvert\\mathbf{r_1} - \\mathbf{r} \\rvert - \\lvert \\mathbf{r_2} - \\mathbf{r_1} \\rvert}`
    """  # noqa: W505 E501
    r1r = vector_norm_eps(r1 - r)
    r2r = vector_norm_eps(r2 - r)
    r2r1 = vector_norm_eps(r2 - r1)
    a = r2r + r1r + r2r1
    b = r2r + r1r - r2r1
    return np.log(a / b)


def get_face_midpoint(face_points: np.ndarray) -> np.ndarray:
    """
    Get an arbitrary point on the face
    """
    return np.sum(face_points[:-1], axis=0) / (len(face_points) - 1)


@nb.jit(nopython=True, cache=True)
def surface_integral(
    face_points: np.ndarray,
    face_normal: np.ndarray,
    mid_point: np.ndarray,
    point: np.ndarray,
) -> float:
    """
    W_f(r)
    """
    omega_f = 0.0
    integral = 0.0
    for i in range(len(face_points) - 1):
        p0 = face_points[i]
        p1 = face_points[i + 1]
        u_e = p1 - p0
        u_e /= np.linalg.norm(u_e)
        integral += np.dot(
            np.cross(face_normal, p0 - point),  # r_e is an arbitrary point
            u_e * edge_integral(point, p0, p1),
        )
        # Calculate omega_f as the sum of subtended angles with a triangle
        # for each edge
        omega_f += omega_t(point, p0, p1, mid_point)
    return integral - np.dot(mid_point - point, face_normal) * omega_f


@nb.jit(nopython=True, cache=True)
def vector_potential(
    current_direction: np.ndarray,
    face_points: np.ndarray,
    face_normals: np.ndarray,
    mid_points: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    """
    Calculate the vector potential

    Parameters
    ----------
    current_direction:
        Normalised current direction vector (3)
    face_points:
        Array of points on each face (n_face, n_points, 3)
    face_normals:
        Array of normalised normal vectors to the faces (pointing outwards)
        (n_face, 3)
    mid_points:
        Array of face midpoints (n_face, 3)
    point:
        Point at which to calculate the vector potential (3)

    Returns
    -------
    Vector potential at the point (response to unit current density)
    """
    integral = np.zeros(3)
    for i, normal in enumerate(face_normals):
        integral += np.dot(
            mid_points[i] - point,
            normal * surface_integral(face_points[i], normal, point),
        )
    return MU_0 / (8 * np.pi) * np.dot(current_direction, integral)


# @nb.jit(nopython=True, cache=True)
def field(
    current_direction: np.ndarray,
    face_points: np.ndarray,
    face_normals: np.ndarray,
    mid_points: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    """
    Calculate the magnetic field

    Parameters
    ----------
    current_direction:
        Normalised current direction vector (3)
    face_points:
        Array of points on each face (n_face, n_points, 3)
    face_normals:
        Array of normalised normal vectors to the faces (pointing outwards)
        (n_face, 3)
    mid_points:
        Array of face midpoints (n_face, 3)
    point:
        Point at which to calculate the magnetic field (3)

    Returns
    -------
    Magnetic field vector at the point (response to unit current density)
    """
    field = np.zeros(3)
    for i, normal in enumerate(face_normals):
        field += np.cross(current_direction, normal) * surface_integral(
            face_points[i], normal, mid_points[i], point
        )
    return MU_0_4PI * field


class PolyhedralPrismCurrentSource(
    PrismEndCapMixin, PolyhedralCrossSectionCurrentSource
):
    """
    3-D polyhedral prism current source with a polyhedral cross-section and
    uniform current distribution.

    The current direction is along the local y coordinate.

    The cross-section is specified in the local x-z plane.

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
    xs_coordinates:
        Coordinates of the conductor cross-section (specified in the x-z plane)
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
        xs_coordinates: Coordinates,
        alpha: float,
        beta: float,
        current: float,
    ):
        alpha, beta = np.deg2rad(alpha), np.deg2rad(beta)
        self.origin = origin

        length = np.linalg.norm(ds)
        self._halflength = 0.5 * length
        self._check_angle_values(alpha, beta)
        m_breadth = np.max(np.abs(xs_coordinates.x))
        self._check_raise_self_intersection(length, m_breadth, alpha, beta)

        # Normalised direction cosine matrix
        self.dcm = np.array([t_vec, ds / length, normal])
        self._set_cross_section(xs_coordinates)

        self.alpha = alpha
        self.beta = beta

        # Current density
        self.set_current(current)
        self.points = self._calculate_points()

    def _set_cross_section(self, xs_coordinates: Coordinates):
        xs_coordinates = deepcopy(xs_coordinates)
        xs_coordinates.close()
        self.area = get_area_2d(*xs_coordinates.xz)
        self._xs = xs_coordinates
        self._xs.set_ccw([0, 1, 0])

    @process_xyz_array
    def field(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
    ) -> np.ndarray:
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
        return self.rho * field(
            self.dcm[1], self.face_points, self.face_normals, self.mid_points, point
        )

    @process_xyz_array
    def vector_potential(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate the vector potential at a point due to the current source.

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
        The vector potential {Ax, Ay, Az} in [T]
        """
        point = np.array([x, y, z])
        return self.rho * vector_potential(
            self.dcm[1], self.face_points, self.face_normals, self.mid_points, point
        )

    def _calculate_points(self):
        """
        Calculate extrema points of the current source for integration and plotting
        purposes
        """
        # Lower shape
        n_rect_faces = len(self._xs) - 1
        lower = deepcopy(self._xs.xyz)
        # Project and translate points onto end cap plane
        lower[1] += -self._halflength - lower[0] * np.tan(self.beta)
        lower_points = self._local_to_global(lower.T)

        # Upper shape
        upper = deepcopy(self._xs.xyz)
        # Project and translate points onto end cap plane
        upper[1] += self._halflength + upper[0] * np.tan(self.alpha)
        upper_points = self._local_to_global(upper.T)

        face_points = [lower_points]
        for i in range(n_rect_faces):
            # Assemble rectangular joining faces
            fp = [
                lower_points[i],
                upper_points[i],
                upper_points[i + 1],
                lower_points[i + 1],
                lower_points[i],
            ]
            face_points.append(fp)
        # Important to make sure the normal faces outwards!
        face_points.append(list(upper_points[::-1]))

        mid_points = [get_face_midpoint(face) for face in face_points]

        self.face_points = np.array(face_points)
        self.mid_points = np.array(mid_points)
        normals = [np.cross(p[1] - p[0], p[2] - p[1]) for p in self.face_points]
        self.face_normals = np.array([n / np.linalg.norm(n) for n in normals])

        # Points for plotting only
        points = [np.vstack(lower_points), np.vstack(upper_points)]
        # Lines between corners
        points.extend(
            [np.vstack([lower_points[i], upper_points[i]]) for i in range(n_rect_faces)]
        )

        return np.array(points, dtype=object)

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

The easiest description and structure to follow was M. Fabbri,
"Magnetic Flux Density and Vector Potential of Uniform Polyhedral Sources",
IEEE TRANSACTIONS ON MAGNETICS, VOL. 44, NO. 1, JANUARY 2008

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4407584

Additional information and detail also present in Passaroto's Master's thesis:
https://thesis.unipd.it/retrieve/d0269be2-2e5d-4068-af58-4374193d38a1/Passarotto_Mauro_tesi.pdf
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

__all__ = ["PolyhedralPrismCurrentSource"]

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
def omega_t(
    r: np.ndarray, r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, normal
) -> float:
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

    # TODO: Remove this sanity check
    t_normal = np.cross(r2 - r1, r3 - r1)
    t_normal /= np.linalg.norm(t_normal)
    if not np.allclose(t_normal, normal, rtol=1e-8, atol=1e-8, equal_nan=False):
        print(t_normal, normal)

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
    Evaluate the surface integral W_f(r) on a planar face

    Parameters
    ----------
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
    Value of the surface integral W_f(r)

    Notes
    -----
    \t:math:`W_f(\\mathbf{r}) = -(\\mathbf{r_f}-\\mathbf{r}) \\cdot \\mathbf{n_f} \\Omega_f(\\mathbf{r}) + \\sum_{l_e \\in \\partial S_f} \\mathbf{n_f} \\times (\\mathbf{r_e} - \\mathbf{r}) \\cdot \\mathbf{u_e}w_e(\\mathbf{r})`
    """  # noqa: W505 E501
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
        omega_f += omega_t(point, p0, p1, mid_point, face_normal)
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
        self._origin = origin

        length = np.linalg.norm(ds)
        self._halflength = 0.5 * length
        self._check_angle_values(alpha, beta)
        m_breadth = np.max(np.abs(xs_coordinates.x))
        self._check_raise_self_intersection(length, m_breadth, alpha, beta)

        # Normalised direction cosine matrix
        self._dcm = np.array([t_vec, ds / length, normal])
        self._set_cross_section(xs_coordinates)

        self._alpha = alpha
        self._beta = beta

        # Current density
        self.set_current(current)
        self._points = self._calculate_points()

    def _set_cross_section(self, xs_coordinates: Coordinates):
        xs_coordinates = deepcopy(xs_coordinates)
        xs_coordinates.close()
        self._area = get_area_2d(*xs_coordinates.xz)
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
        return self._rho * field(
            self._dcm[1], self._face_points, self._face_normals, self._mid_points, point
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
        return self._rho * vector_potential(
            self._dcm[1], self._face_points, self._face_normals, self._mid_points, point
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
        lower[1] += -self._halflength - lower[0] * np.tan(self._beta)
        lower_points = self._local_to_global(lower.T)

        # Upper shape
        upper = deepcopy(self._xs.xyz)
        # Project and translate points onto end cap plane
        upper[1] += self._halflength + upper[0] * np.tan(self._alpha)
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

        self._face_points = np.array(face_points)
        self._mid_points = np.array(mid_points)
        normals = [np.cross(p[1] - p[0], p[2] - p[1]) for p in self._face_points]
        self._face_normals = np.array([n / np.linalg.norm(n) for n in normals])

        # Points for plotting only
        points = [np.vstack(lower_points), np.vstack(upper_points)]
        # Lines between corners
        points.extend(
            [np.vstack([lower_points[i], upper_points[i]]) for i in range(n_rect_faces)]
        )

        return np.array(points, dtype=object)


class BotturaPolyhedralPrismCurrentSource(PolyhedralPrismCurrentSource):
    """
    Alternative VIM formulation without the Stokes trick
    """

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
        https://supermagnet.sourceforge.io/notes/CRYO-02-028.pdf
        https://supermagnet.sourceforge.io/notes/CRYO-97-003.pdf
        """
        point = np.array([x, y, z])
        A = _vector_potential(self._face_normals, self._face_points, point)
        return 0.5 * MU_0_4PI * self._rho * A * self._dcm[1]

    @process_xyz_array
    def field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        point = np.array([x, y, z])
        J = self._rho * self._dcm[1]

        return -MU_0_4PI * np.cross(
            J,
            _field_new(
                self._face_normals, self._face_points, point, self._origin, test=False
            ),
        )


# @nb.jit(nopython=True)
def _vector_potential(
    face_normals: np.ndarray,
    face_corners: np.ndarray,
    point: np.ndarray,
):
    A = 0.0

    for i in range(6):  # Faces of the prism
        s = 0.0
        zpp = np.dot(face_normals[i, :], point)
        azpp = abs(zpp)
        zpp2 = zpp**2

        for j in range(4):  # Lines of the face
            zpp_axis = face_normals[i]
            xpp_axis = face_corners[i][j + 1] - face_corners[i][j]
            x_side_len = np.linalg.norm(xpp_axis)
            ypp_axis = -np.cross(zpp_axis, xpp_axis / x_side_len)
            dcm = np.zeros((3, 3))
            dcm[0, :] = xpp_axis
            dcm[1, :] = ypp_axis
            dcm[2, :] = zpp_axis
            point_local = np.dot(dcm, point)

            ypp = point_local[1]

            # Lack of choice of centroid of reference frame, so pick an easy one
            xpp1 = point_local[0]
            xpp2 = point_local[0] + x_side_len

            ypp2_zpp2 = ypp**2 + zpp2
            r1 = np.sqrt(xpp1**2 + ypp2_zpp2)
            r2 = np.sqrt(xpp2**2 + ypp2_zpp2)

            s += ypp * (
                azpp
                / ypp
                * (
                    np.arctan(xpp2 * azpp / (ypp * r2))
                    - np.arctan(xpp1 * azpp / (ypp * r1))
                    + np.arctan(xpp1 / ypp)
                    - np.arctan(xpp2 / ypp)
                )
                + np.log(xpp2 + r2)
                - np.log(xpp1 + r1)
            )

        A += zpp * s
    return A


@nb.jit(nopython=True)
def _line_integral(x: float, y: float, z: float) -> float:
    abs_z = abs(z)
    r = np.sqrt(x**2 + y**2 + z**2)
    if y == 0:
        return np.log(x + r)
    a1 = np.arctan2(x * abs_z, (y * r))
    a2 = np.arctan2(x, y)
    return np.log(x + r) + (abs_z / y) * (a1 - a2)


def _field_new(face_normals, face_points, point, source_origin, test=False):
    B = np.zeros(3)

    for i, face_normal in enumerate(face_normals):  # Faces of the prism
        s = 0.0
        zpp = np.dot(face_normal, point)

        for j in range(4):  # Lines of the face
            corner_1, corner_2 = face_points[i][j], face_points[i][j + 1]
            xpp_axis = corner_2 - corner_1
            xpp_axis /= np.linalg.norm(xpp_axis)
            # Ensure y is pointing outwards
            ypp_axis = -np.cross(face_normal, xpp_axis)
            dcm = np.zeros((3, 3))
            dcm[0, :] = xpp_axis
            dcm[1, :] = ypp_axis
            dcm[2, :] = face_normal

            # NOTE: zpp should be constant for each surface, and ypp for each line, but i am lazy
            xppq1, ypp, zpp = np.dot(dcm, corner_1 - point)
            xppq2 = np.dot(dcm, corner_2 - point)[0]

            s += ypp * (
                _line_integral(xppq2, ypp, zpp) - _line_integral(xppq1, ypp, zpp)
            )

        B += face_normal * s
    return B

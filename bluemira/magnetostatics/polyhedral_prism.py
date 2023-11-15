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

An alternative calculation also available, which should give identical results
from Bottura et al., following essentially C. J. Collie's 1976 RAL work

https://supermagnet.sourceforge.io/notes/CRYO-06-034.pdf
"""

import abc
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


class PolyhedralKernel(abc.ABC):
    """
    Baseclass for the polyhedral prism magnetostatics kernel
    """

    @abc.abstractstaticmethod
    def field(*args) -> np.ndarray:
        """
        Magnetic field
        """

    @abc.abstractstaticmethod
    def vector_potential(*args) -> np.ndarray:
        """
        Vector potential
        """


class Fabbri(PolyhedralKernel):
    """
    Fabbri polyhedral prism formulation

    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4407584
    """

    @staticmethod
    def field(*args) -> np.ndarray:
        """
        Magnetic field
        """
        return field_fabbri(*args)

    @staticmethod
    def vector_potential(*args) -> np.ndarray:
        """
        Vector potential
        """
        return vector_potential_fabbri(*args)


class Bottura(PolyhedralKernel):
    """
    Bottura polyhedral prism formulation

    https://supermagnet.sourceforge.io/notes/CRYO-06-034.pdf
    """

    @staticmethod
    def field(*args) -> np.ndarray:
        """
        Magnetic field
        """
        return _field_bottura(*args)

    @staticmethod
    def vector_potential(*args) -> np.ndarray:
        """
        Vector potential
        """
        return _vector_potential_bottura(*args)


class Ciric(PolyhedralKernel):
    """
    Ciric polyhedral prism formulation

    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=123865&tag=1
    """

    @staticmethod
    def field(*args) -> np.ndarray:
        """
        Magnetic field
        """
        return _field_ciric(*args)

    @staticmethod
    def vector_potential(*args) -> np.ndarray:
        """
        Vector potential
        """
        raise NotImplementedError


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
def edge_integral_fabbri(r: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> float:
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
def surface_integral_fabbri(
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
            u_e * edge_integral_fabbri(point, p0, p1),
        )
        # Calculate omega_f as the sum of subtended angles with a triangle
        # for each edge
        omega_f += omega_t(point, p0, p1, mid_point, face_normal)
    return integral - np.dot(mid_point - point, face_normal) * omega_f


@nb.jit(nopython=True, cache=True)
def vector_potential_fabbri(
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
            normal * surface_integral_fabbri(face_points[i], normal, point),
        )
    return MU_0 / (8 * np.pi) * np.dot(current_direction, integral)


# @nb.jit(nopython=True, cache=True)
def field_fabbri(
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
        field += np.cross(current_direction, normal) * surface_integral_fabbri(
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

        # Kernel (not intended to be user-facing)

        self.__kernel = Fabbri()

    @property
    def _kernel(self):
        return self.__kernel

    @_kernel.setter
    def _kernel(self, value: PolyhedralKernel):
        self.__kernel = value

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
        return self._rho * self.__kernel.field(
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
        return self._rho * self.__kernel.vector_potential(
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


# @nb.jit(nopython=True)
def _vector_potential_bottura(
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
    A = 0.0

    for i, face_normal in enumerate(face_normals):  # Faces of the prism
        surface_integral = _surface_integral_bottura(face_normal, face_points[i], point)
        zpp = np.dot(face_normal, mid_points[i] - point)
        A += zpp * surface_integral
    return 0.5 * MU_0_4PI * A * current_direction


def _field_bottura(
    current_direction: np.ndarray,
    face_points: np.ndarray,
    face_normals: np.ndarray,
    mid_points: np.ndarray,  # noqa: ARG001
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
    B = np.zeros(3)

    for i, face_normal in enumerate(face_normals):  # Faces of the prism
        surface_integral = _surface_integral_bottura(face_normal, face_points[i], point)
        B += face_normal * surface_integral
    return -MU_0_4PI * np.cross(current_direction, B)


def _surface_integral_bottura(face_normal, face_points, point):
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
    \t:math:`W_f(\\mathbf{r}) = \\sum_{j} y_{P}^{''}\\bigg[I_{1}(x_{Q2}^{''}-x_{P}^{''}, y_{P}^{''}, z_{P}^{''}) - I_{1}(x_{Q1}^{''}-x_{P}^{''}, y_{P}^{''}, z_{P}^{''})\\bigg]`
    """  # noqa: W505 E501
    integral = 0.0

    for j in range(len(face_points) - 1):  # Lines of the face
        corner_1, corner_2 = face_points[j], face_points[j + 1]
        xpp_axis = corner_2 - corner_1
        xpp_axis /= np.linalg.norm(xpp_axis)
        # Ensure y is pointing outwards
        ypp_axis = -np.cross(face_normal, xpp_axis)
        dcm = np.zeros((3, 3))
        dcm[0, :] = xpp_axis
        dcm[1, :] = ypp_axis
        dcm[2, :] = face_normal

        # NOTE: zpp should be constant for each surface, and ypp for each line,
        # but i am lazy
        xppq1, ypp, zpp = np.dot(dcm, corner_1 - point)
        xppq2 = np.dot(dcm, corner_2 - point)[0]

        integral += ypp * (
            _line_integral_bottura(xppq2, ypp, zpp)
            - _line_integral_bottura(xppq1, ypp, zpp)
        )
    return integral


@nb.jit(nopython=True, cache=True)
def _line_integral_bottura(x: float, y: float, z: float) -> float:
    """
    Line integral I_1(x, y, z)

    Parameters
    ----------
    x:
        x coordinate
    y:
        y coordinate
    z:
        z coordinate

    Returns
    -------
    Value of the line integral I_1(x, y, z)

    Notes
    -----
    \t:math:`\\int \\dfrac{1}{r+\\lvert z \\rvert}dx \\equiv I_{1}(x, y, z) = \\textrm{ln}(x+r)+\\dfrac{\\lvert z \\rvert}{y}\\bigg(\\textrm{arctan}\\bigg(\\dfrac{x\\lvert z \\rvert}{yr}\\bigg) - \\textrm{arctan}\\bigg(\\dfrac{x}{y}\\bigg)\\bigg)`

    with:
    \t:math:`r \\equiv \\sqrt{x^2+y^2+z^2}`
    """  # noqa: W505 E501
    abs_z = abs(z)
    r = np.sqrt(x**2 + y**2 + z**2)
    if y == 0 or r == 0:
        # This may not be the right solution
        return np.log(x + r)
    a1 = np.arctan2(x * abs_z, (y * r))
    a2 = np.arctan2(x, y)
    return np.log(x + r) + (abs_z / y) * (a1 - a2)


def _field_ciric(  # noqa: PLR0915
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
    B = np.zeros(3)

    # Coordinate reordering
    rectangles = []
    for i, r in enumerate(face_points):
        if i == 0:
            # First end cap
            new = [r[0], r[1], r[2], r[3]]
        elif i == 5:  # noqa: PLR2004
            # Second end cap
            new = [r[0], r[1], r[2], r[3]]  # [::-1]

        # Fairly sure the side faces are correct
        elif i == 1:  # noqa: SIM114
            new = [r[3], r[0], r[1], r[2]]
            # new = [r[2], r[3], r[0], r[1]]
        elif i == 2:  # noqa: PLR2004
            new = [r[3], r[0], r[1], r[2]]
            # new = [r[2], r[3], r[0], r[1]]
            # new = [r[2], r[1], r[0], r[3]]
            # [::-1]  # ?
            # new = [r[0], r[1], r[2], r[3]][::-1]
        elif i == 3:  # noqa: PLR2004
            # This can give a negative z_2 value
            new = [r[3], r[0], r[1], r[2]]
            # new = [r[2], r[3], r[0], r[1]]
            # This is a good alternative because it sets the trapezoid "right"..ish
            # new = [r[1], r[2], r[3], r[0]]
        elif i == 4:  # noqa: PLR2004
            new = [r[3], r[0], r[1], r[2]]
            # new = [r[2], r[3], r[0], r[1]]
            # new = [r[3], r[0], r[1], r[2]][::-1]
        rectangles.append(new)

    # "Arbitrary" origin
    o_1 = rectangles[0][0]
    o_2 = rectangles[-1][0]
    o_1 - o_2
    o_3 = rectangles[0][2]
    o_4 = rectangles[-1][2]

    # Arbitrary origin = mid1
    mid1 = 0.5 * (o_1 + o_2)
    mid2 = 0.5 * (o_3 + o_4)

    m_c = mid2 - mid1
    m_c /= np.linalg.norm(m_c)

    # odn = np.linalg.norm(origin_line)
    # ap = point - mid1
    # D = np.linalg.norm(np.cross(ap, origin_line)) / odn
    # D = np.linalg.norm(np.cross(point - o_1, point - o_2)) / np.linalg.norm(o_2 - o_1)

    # Assume that the distance X is to the field point
    X = np.dot(point - mid1, np.cross(m_c, current_direction))  # noqa: N806

    inside = _point_in_volume(point, face_normals, mid_points)
    #
    # All faces
    idx = [0, 1, 2, 3, 4, 5]
    # Side faces only
    idx = [1, 2, 3, 4]
    # Side trapezoids
    # idx = [1, 3]
    # Top bottom rectangles
    # idx = [2, 4]

    rectangles = np.array(rectangles)

    for i, rectangle in zip(idx, rectangles[idx]):
        normal = face_normals[i]
        r_1 = point - rectangle[0]
        r_2 = point - rectangle[1]
        r_3 = point - rectangle[2]
        r_4 = point - rectangle[3]
        l_12 = r_1 - r_2
        l_34 = r_3 - r_4
        l_12n = np.linalg.norm(l_12)
        l_34n = np.linalg.norm(l_34)

        # Conversion to local coordinate system
        zh = r_1 - r_4
        zh /= np.linalg.norm(zh)
        z_2 = np.dot(zh, l_12)

        d = np.sqrt(l_12n**2 - z_2**2)
        xh = np.cross(l_12, zh) / d
        yh = np.cross(zh, xh)
        dcm = np.zeros((3, 3))
        dcm[0, :] = xh
        dcm[1, :] = yh
        dcm[2, :] = zh

        j_sc = np.cross(normal, m_c)

        if True:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(subplot_kw={"projection": "3d"})
            for j, r in enumerate(rectangle):
                ax.plot(*r, marker="o")
                ax.text(*r, str(j))
            # ax.plot(*mid_points[i], marker="x", color="k")
            ax.quiver(*mid_points[i], *0.1 * xh, color="g")
            ax.quiver(*mid_points[i], *0.1 * yh, color="r")
            ax.quiver(*mid_points[i], *0.1 * zh, color="k")
            ax.plot(*point, marker="x", color="b")
            ax.quiver(*rectangle[0], *r_1)
            ax.quiver(*mid_points[i], *j_sc, color="c")
            ax.quiver(*mid_points[i], *0.1 * m_c, color="b")
            plt.show()

        # Check coordinate systems are as expected
        assert np.allclose(  # noqa: S101
            zh,
            (rectangle[3] - rectangle[0]) / np.linalg.norm(rectangle[3] - rectangle[0]),
        )
        assert np.allclose(xh, normal)  # noqa: S101

        r_1n = np.linalg.norm(r_1)
        r_2n = np.linalg.norm(r_2)
        r_3n = np.linalg.norm(r_3)
        r_4n = np.linalg.norm(r_4)
        x = np.dot(xh, r_1)
        y = np.dot(yh, r_1)
        z = np.dot(zh, r_1)
        z_3 = np.dot(zh, r_1 - r_3)
        z_4 = np.dot(zh, r_1 - r_4)

        p_12 = np.dot(xh, np.cross(r_1, r_2))
        p_34 = np.dot(-xh, np.cross(r_3, r_4))
        q_12 = np.dot(r_1, l_12)
        q_34 = np.dot(-r_4, l_34)

        lambda_12 = np.log((r_1n * l_12n - q_12) / ((r_2n + l_12n) * l_12n - q_12))
        lambda_34 = np.log(((r_3n + l_34n) * l_34n - q_34) / (r_4n * l_34n - q_34))
        lambdda = (
            np.log(
                ((r_1n + z) * (r_3n + z - z_3)) / ((r_2n + z - z_2) * (r_4n + z - z_4))
            )
            + lambda_12 * z_2 / l_12n
            + lambda_34 * (z_3 - z_4) / l_34n
        )
        an, ad = z * q_12 - z_2 * r_1n**2, x * r_1n * d
        bn, bd = (z - z_2) * (q_12 - l_12n**2) - z_2 * r_2n**2, x * r_2n * d
        cn, cd = (z - z_3) * (q_34 - l_34n**2) - r_3n**2 * (z_3 - z_4), x * r_3n * d
        dn, dd = (z - z_4) * q_34 - (z_3 - z_4) * r_4n**2, x * r_4n * d

        gamma = (
            np.arctan(an / ad)
            - np.arctan(bn / bd)
            + np.arctan(cn / cd)
            - np.arctan(dn / dd)
        )
        # gamma = (  # This won't work, need to try alternative e.g. omega_t
        #    np.arctan2(an, ad)
        #    - np.arctan2(bn, bd)
        #    + np.arctan2(cn, cd)
        #    - np.arctan2(dn, dd)
        # )
        # gamma = (  # This might work
        #     np.arctan2(ad, an)
        #     - np.arctan2(bd, bn)
        #     + np.arctan2(cd, cn)
        #     - np.arctan2(dd, dn)
        # )
        if np.sign(x) != np.sign(gamma):
            gamma *= -1
            # Check with omega_t  <- need to use anyway for a polygon
        assert np.sign(x) == np.sign(gamma)  # noqa: S101

        r12n = r_1n - r_2n
        r34n = r_3n - r_4n
        lambda_12_l_12_3 = lambda_12 / l_12n**3
        lambda_34_l_34_3 = lambda_34 / l_34n**3
        psic = (
            d * z_2 / l_12n**2 * r12n
            + d * (z_3 - z_4) / l_34n**2 * r34n
            - d**2 * (p_12 * lambda_12_l_12_3 + p_34 * lambda_34_l_34_3)
        )
        n_h = normal
        # Cosines of the angles made by n_h and m_c and m_c x J
        n_p = np.dot(n_h, m_c)
        # TODO: Discrepancy between these two options
        n_pp = np.dot(n_h, current_direction)
        n_pp = np.dot(n_h, np.cross(m_c, current_direction))
        # TODO: Need to account for surface charge only for end faces
        # (decompose polygon into triangles)
        # if i == 0 or i == 5:
        # This is probably not the right way of separating out surface currents
        #    pass  # n_pp = 0

        # Distance from the arbitrary origin (0, 0, 0) to the edge (k-1,k)
        dvec = mid1 - rectangle[0]
        D = np.dot(dvec, np.cross(m_c, current_direction))  # noqa: N806
        B_f = np.zeros(3)
        B_f[0] = (n_pp * D - n_p * (n_p * x + n_pp * y)) * lambdda + (
            (n_p * D - n_p * (-n_pp * x + n_p * y)) * gamma - n_p * n_pp * psic
        )
        B_f[1] = (-(n_p * D - n_p * (-n_pp * x + n_p * y)) * lambdda) + (
            (n_pp * D - n_p * (n_p * x + n_pp * y)) * gamma + n_p**2 * psic
        )
        B_f[2] = (
            n_p * D * d * (lambda_12 / l_12n + lambda_34 / l_34n)
        ) - n_p**2 * d**2 * (
            r12n / l_12n**2
            + r34n / l_34n**2
            + q_12 * lambda_12_l_12_3
            + q_34 * lambda_34_l_34_3
        )

        # TODO: not clear if the resulting field is in local or global coordinates...
        # B += B_f
        # B += np.dot(dcm, B_f)
        B += np.dot(dcm.T, B_f)
        # breakpoint()

    # TODO: factor in area properly once it works
    area = 1
    M_c_r = inside * X * m_c / area  # noqa: N806
    # TODO: figure out M_c units, because I can't believe MU_0 is not involved.
    # M_c_r = 0
    return MU_0_4PI * (B + M_c_r)


def _point_in_volume(point, normals, mid_points):
    """
    Determine if a field point is inside the source
    """
    # Who'd have thought it was this easy?
    for i, normal in enumerate(normals):
        if np.dot(normal, point - mid_points[i]) > 0:
            return False
    return True

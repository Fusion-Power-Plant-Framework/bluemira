# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
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

import numba as nb
import numpy as np

from bluemira.base.constants import EPS, MU_0_4PI
from bluemira.geometry.coordinates import Coordinates, get_area_2d
from bluemira.magnetostatics.baseclass import (
    PolyhedralCrossSectionCurrentSource,
    PrismEndCapMixin,
)
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.tools import process_xyz_array

__all__ = ["PolyhedralPrismCurrentSource"]
# NOTE: Polyhedral kernels are not intended to be user-facing, but
# it's useful for testing.


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
        return _field_fabbri(*args)

    @staticmethod
    def vector_potential(*args) -> np.ndarray:
        """
        Vector potential
        """
        return _vector_potential_fabbri(*args)


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


@nb.jit(nopython=True, cache=True)
def _vector_norm_eps(r: np.ndarray) -> float:
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
    return np.sqrt(r_norm**2 + EPS)  # guard against division by 0


@nb.jit(nopython=True, cache=True)
def _omega_t(r: np.ndarray, r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> float:
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

    r1r = _vector_norm_eps(r1_r)
    r2r = _vector_norm_eps(r2_r)
    r3r = _vector_norm_eps(r3_r)
    d = (
        r1r * r2r * r3r
        + r3r * np.dot(r1_r, r2_r)
        + r2r * np.dot(r1_r, r3_r)
        + r1r * np.dot(r2_r, r3_r)
    )
    a = np.dot(r1_r, np.cross(r2_r, r3_r))
    # Not sure this is an issue...
    # if abs(a) < EPS and (-EPS < d < 0): # guard against division by 0
    #     return 0 # and not pi as per IEEE
    return 2 * np.arctan2(a, d)


@nb.jit(nopython=True, cache=True)
def _edge_integral_fabbri(r: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> float:
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
    r1r = _vector_norm_eps(r1 - r)
    r2r = _vector_norm_eps(r2 - r)
    r2r1 = _vector_norm_eps(r2 - r1)
    a = r2r + r1r + r2r1
    b = r2r + r1r - r2r1
    return np.log(a / b)


@nb.jit(nopython=True, cache=True)
def _get_face_midpoint(face_points: np.ndarray) -> np.ndarray:
    """
    Get an arbitrary point on the face
    """
    return np.sum(face_points[:-1], axis=0) / (len(face_points) - 1)


@nb.jit(nopython=True, cache=True)
def _get_face_normal(face_points: np.ndarray) -> np.ndarray:
    """
    Get the normal of a face
    """
    normal = np.cross(face_points[1] - face_points[0], face_points[2] - face_points[1])
    return normal / np.linalg.norm(normal)


@nb.jit(nopython=True, cache=True)
def _surface_integral_fabbri(
    face_points: np.ndarray,
    face_normal: np.ndarray,
    mid_point: np.ndarray,
    n_sides: int,
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
    n_sides:
        Number of points in the face points (avoid reflected lists)
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
    for i in range(n_sides):
        p0 = face_points[i]
        p1 = face_points[i + 1]
        u_e = p1 - p0
        u_e /= np.linalg.norm(u_e)
        integral += np.dot(
            np.cross(face_normal, p0 - point),  # r_e is an arbitrary point
            u_e * _edge_integral_fabbri(point, p0, p1),
        )
        # Calculate omega_f as the sum of subtended angles with a triangle
        # for each edge
        omega_f += _omega_t(point, p0, p1, mid_point)
    return integral - np.dot(mid_point - point, face_normal) * omega_f


@nb.jit(nopython=True, cache=True)
def _vector_potential_fabbri(
    current_direction: np.ndarray,
    face_points: np.ndarray,
    face_normals: np.ndarray,
    mid_points: np.ndarray,
    n_sides: np.ndarray,
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
    n_sides:
        Number of points in the face points (avoid reflected lists)
    point:
        Point at which to calculate the vector potential (3)

    Returns
    -------
    Vector potential at the point (response to unit current density)
    """
    integral = np.zeros(3)
    for i in range(len(face_normals)):
        integral += np.dot(
            mid_points[i] - point,
            face_normals[i]
            * _surface_integral_fabbri(
                face_points[i], face_normals[i], mid_points[i], n_sides[i], point
            ),
        )

    return 0.5 * MU_0_4PI * np.dot(current_direction, integral)


@nb.jit(nopython=True, cache=True)
def _field_fabbri(
    current_direction: np.ndarray,
    face_points: np.ndarray,
    face_normals: np.ndarray,
    mid_points: np.ndarray,
    n_sides: np.ndarray,
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
    n_sides:
        Number of points in the face points (avoid reflected lists)
    point:
        Point at which to calculate the magnetic field (3)

    Returns
    -------
    Magnetic field vector at the point (response to unit current density)
    """
    field = np.zeros(3)
    for i in range(len(face_normals)):
        normal = face_normals[i]
        field += np.cross(current_direction, normal) * _surface_integral_fabbri(
            face_points[i], normal, mid_points[i], n_sides[i], point
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
    Negative angles are allowed, but both angles must be equal
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

    def _check_angle_values(self, alpha, beta):
        """
        Check that end-cap angles are acceptable.
        """
        if not np.isclose(alpha, beta):
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: {alpha=:.3f} "
                f"!= {beta=:.3f}"
            )
        if not (0 <= abs(alpha) < 0.5 * np.pi):
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: {alpha=:.3f} is outside"
                " bounds of [0, 180°)."
            )

    def _set_cross_section(self, xs_coordinates: Coordinates):
        xs_coordinates = deepcopy(xs_coordinates)
        xs_coordinates.close()
        self._area = get_area_2d(*xs_coordinates.xz)
        self._xs = xs_coordinates
        self._xs.set_ccw([0, 1, 0])

    @process_xyz_array
    def field(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        z: float | np.ndarray,
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
            self._dcm[1],
            self._face_points,
            self._face_normals,
            self._mid_points,
            self._n_sides,
            point,
        )

    @process_xyz_array
    def vector_potential(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        z: float | np.ndarray,
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
            self._dcm[1],
            self._face_points,
            self._face_normals,
            self._mid_points,
            self._n_sides,
            point,
        )

    def _calculate_points(self) -> np.ndarray:
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
        (
            self._face_points,
            self._mid_points,
            self._face_normals,
            self._n_sides,
        ) = _generate_source_geometry(lower_points, upper_points)

        # Points for plotting only
        points = [np.vstack(lower_points), np.vstack(upper_points)]
        # Lines between corners
        points.extend([
            np.vstack([lower_points[i], upper_points[i]]) for i in range(n_rect_faces)
        ])

        return np.array(points, dtype=object)


@nb.jit(nopython=True, cache=True)
def _generate_source_geometry(
    lower_points: np.ndarray, upper_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the polyhedral prism source geometry - faster

    Returns
    -------
    face_points:
        The ordered points of the faces
    mid_points:
        The mid-points for each face
    face_normals:
        The normal vectors for each face (normalised)
    n_sides:
        The array of the number of sides to consider for each face
    """
    # Need to have an array of constant size
    n_end_caps = len(lower_points)
    n_rect_faces = n_end_caps - 1
    n_faces = n_rect_faces + 2
    n_face_max = max(5, n_end_caps)

    face_points = np.zeros((n_faces, n_face_max, 3), dtype=np.float64)
    mid_points = np.zeros((n_faces, 3), dtype=np.float64)
    face_normals = np.zeros((n_faces, 3), dtype=np.float64)

    face_points[0, :n_end_caps, :] = lower_points
    mid_points[0, :] = _get_face_midpoint(lower_points)
    face_normals[0, :] = _get_face_normal(lower_points)

    face_p = np.zeros((5, 3), dtype=np.float64)
    for i in range(n_rect_faces):
        # Assemble rectangular joining faces
        lpi = lower_points[i, :]
        face_p[0, :] = lpi
        face_p[1, :] = upper_points[i, :]
        face_p[2, :] = upper_points[i + 1, :]
        face_p[3, :] = lower_points[i + 1, :]
        face_p[4, :] = lpi
        face_points[i + 1, :5, :] = face_p
        mid_points[i + 1, :] = _get_face_midpoint(face_p)
        face_normals[i + 1, :] = _get_face_normal(face_p)

    face_points[-1, :n_end_caps, :] = upper_points[::-1]
    mid_points[-1, :] = _get_face_midpoint(face_points[-1, :, :])
    face_normals[-1, :] = _get_face_normal(face_points[-1, :, :])

    n_sides = 4 * np.ones(n_faces, dtype=np.int32)
    n_sides[0] = n_end_caps - 1
    n_sides[-1] = n_sides[0]

    return face_points, mid_points, face_normals, n_sides


@nb.jit(nopython=True, cache=True)
def _vector_potential_bottura(
    current_direction: np.ndarray,
    face_points: np.ndarray,
    face_normals: np.ndarray,
    mid_points: np.ndarray,
    n_sides: np.ndarray,
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
    n_sides:
        Number of points in the face points (avoid reflected lists)
    point:
        Point at which to calculate the vector potential (3)

    Returns
    -------
    Vector potential at the point (response to unit current density)
    """
    A = 0.0

    for i, face_normal in enumerate(face_normals):  # Faces of the prism
        surface_integral = _surface_integral_bottura(
            face_normal, face_points[i], n_sides[i], point
        )
        zpp = np.dot(face_normal, mid_points[i] - point)
        A += zpp * surface_integral
    return 0.5 * MU_0_4PI * A * current_direction


@nb.jit(nopython=True, cache=True)
def _field_bottura(
    current_direction: np.ndarray,
    face_points: np.ndarray,
    face_normals: np.ndarray,
    mid_points: np.ndarray,  # noqa: ARG001
    n_sides: np.ndarray,
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
    n_sides:
        Number of points in the face points (avoid reflected lists)
    point:
        Point at which to calculate the magnetic field (3)

    Returns
    -------
    Magnetic field vector at the point (response to unit current density)
    """
    B = np.zeros(3)

    for i, face_normal in enumerate(face_normals):  # Faces of the prism
        surface_integral = _surface_integral_bottura(
            face_normal, face_points[i], n_sides[i], point
        )
        B += face_normal * surface_integral
    return -MU_0_4PI * np.cross(current_direction, B)


@nb.jit(nopython=True, cache=True)
def _surface_integral_bottura(
    face_normal: np.ndarray, face_points: np.ndarray, n_sides: int, point: np.ndarray
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
    \t:math:`W_f(\\mathbf{r}) = \\sum_{j} y_{P}^{''}\\bigg[I_{1}(x_{Q2}^{''}-x_{P}^{''}, y_{P}^{''}, z_{P}^{''}) - I_{1}(x_{Q1}^{''}-x_{P}^{''}, y_{P}^{''}, z_{P}^{''})\\bigg]`
    """  # noqa: W505 E501
    integral = 0.0

    for j in range(n_sides):  # Lines of the face
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

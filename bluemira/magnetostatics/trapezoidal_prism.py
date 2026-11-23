# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Analytical expressions for the field inside an arbitrarily shaped winding pack
of rectangular cross-section, following equations as described in:

.. doi:: 10.1002/jnm.594

including corrections from:

.. doi:: 10.1002/jnm.675

The vector potential is calculated with the closed-form expression for the
Newtonian potential of a uniform polyhedron (the trapezoidal prism is a hexahedron
with planar faces), following e.g.:

.. doi:: 10.1007/BF01228427
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

import numba as nb
import numpy as np
import numpy.typing as npt

from bluemira.base.constants import MU_0_4PI
from bluemira.geometry.coordinates import rotation_matrix
from bluemira.magnetostatics.baseclass import CrossSectionCurrentSource, PrismEndCapMixin
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.tools import process_xyz_array
from bluemira.utilities.plot_tools import Plot3D

__all__ = ["TrapezoidalPrismCurrentSource"]


@nb.jit(nopython=True, cache=True)
def primitive_sxn_bound(
    cos_theta: float, sin_theta: float, r: float, q: float, t: float
) -> float:
    """
    Function primitive of Bx evaluated at a bound.

    Parameters
    ----------
    cos_theta:
        The cosine of theta in radians
    sin_theta:
        The sine of theta in radians
    r:
        The r local coordinate value
    q:
        The q local coordinate value
    t:
        The t local coordinate value

    Returns
    -------
    result:
        The value of the primitive function at integral bound t

    Notes
    -----
    Uses corrected formulae available at:
    .. doi:: 10.1002/jnm.675

    Singularities all resolve to: lim(ln(1)) --> 0
    """
    # First compute the divisors of each of the terms to determine if there
    # are singularities
    divisor_1 = cos_theta * np.sqrt(t**2 + q**2)
    divisor_2 = cos_theta * np.sqrt(r**2 * cos_theta**2 + q**2)
    divisor_3 = q * np.sqrt(
        t**2 + 2 * r * t * sin_theta * cos_theta + (r**2 + q**2) * cos_theta**2
    )

    # All singularities resolve to 0?
    result = 0
    if divisor_1 != 0:
        result += t * np.arcsinh((t * sin_theta + r * cos_theta) / divisor_1)
    if divisor_2 != 0:
        result += r * cos_theta * np.arcsinh((t + r * cos_theta * sin_theta) / divisor_2)
    if divisor_3 != 0:
        result += q * np.arctan((q**2 * sin_theta - t * r * cos_theta) / divisor_3)

    return result


@nb.jit(nopython=True, cache=True)
def primitive_sxn(theta: float, r: float, q: float, l1: float, l2: float) -> float:
    """
    Analytical integral of Bx function primitive.

    Parameters
    ----------
    theta:
        The angle in radians
    r:
        The r local coordinate value
    q:
        The q local coordinate value
    l1:
        The first local coordinate bound
    l2:
        The second local coordinate bound

    Returns
    -------
    :
        The value of the integral of the Bx function primitive
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return primitive_sxn_bound(cos_theta, sin_theta, r, q, l2) - primitive_sxn_bound(
        cos_theta, sin_theta, r, q, l1
    )


@nb.jit(nopython=True, cache=True)
def primitive_szn_bound(
    cos_theta: float, sin_theta: float, r: float, ll: float, t: float
) -> float:
    """
    Function primitive of Bz evaluated at a bound.

    Parameters
    ----------
    cos_theta:
        The cosine of theta
    sin_theta:
        The sine of theta
    r:
        The r local coordinate value
    ll:
        The l local coordinate value
    t:
        The t local coordinate value

    Returns
    -------
    result:
        The value of the primitive function at integral bound t

    Notes
    -----
    Singularities all resolve to: lim(ln(1)) --> 0
    """
    sqrt_term = np.sqrt(
        t**2 * cos_theta**2
        + ll**2
        + 2 * r * ll * sin_theta * cos_theta
        + r**2 * cos_theta**2
    )

    # First compute the divisors of each of the terms to determine if there
    # are singularities
    divisor_1 = cos_theta * np.sqrt(t**2 + r**2 * cos_theta**2)
    divisor_2 = cos_theta * np.sqrt(t**2 + ll**2)
    divisor_3 = np.sqrt(ll**2 + 2 * ll * r * sin_theta * cos_theta + r**2 * cos_theta**2)
    divisor_4 = r * cos_theta * sqrt_term
    divisor_5 = ll * sqrt_term

    # All singularities resolve to 0?
    result = 0
    if divisor_1 != 0:
        result += (
            t * sin_theta * np.arcsinh((ll + r * sin_theta * cos_theta) / divisor_1)
        )
    if divisor_2 != 0:
        result -= t * np.arcsinh((ll * sin_theta + r * cos_theta) / divisor_2)
    if divisor_3 != 0:
        result -= r * cos_theta**2 * np.arcsinh((t * cos_theta) / divisor_3)
    if divisor_4 != 0:
        result -= (
            r
            * cos_theta
            * sin_theta
            * np.arctan((t * (ll + r * sin_theta * cos_theta)) / divisor_4)
        )
    if divisor_5 != 0:
        result += ll * np.arctan((t * (ll * sin_theta + r * cos_theta)) / divisor_5)

    return result


@nb.jit(nopython=True, cache=True)
def primitive_szn(theta: float, r: float, ll: float, q1: float, q2: float) -> float:
    """
    Analytical integral of Bz function primitive.

    Parameters
    ----------
    theta:
        The angle in radians
    r:
        The r local coordinate value
    ll:
        The l local coordinate value
    q1:
        The first local coordinate bound
    q2:
        The second local coordinate bound

    Returns
    -------
    :
        The value of the integral of the Bx function primitive
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return primitive_szn_bound(cos_theta, sin_theta, r, ll, q2) - primitive_szn_bound(
        cos_theta, sin_theta, r, ll, q1
    )


@nb.jit(nopython=True, cache=True)
def Bx_analytical_prism(
    alpha: float,
    beta: float,
    l1: float,
    l2: float,
    q1: float,
    q2: float,
    r1: float,
    r2: float,
) -> float:
    """
    Calculate magnetic field in the local x coordinate direction due to a
    trapezoidal prism current source.

    Parameters
    ----------
    alpha:
        The first trapezoidal angle [rad]
    beta:
        The second trapezoidal angle [rad]
    l1:
        The local l1 coordinate [m]
    l2:
        The local l2 coordinate [m]
    q1:
        The local q1 coordinate [m]
    q2:
        The local q2 coordinate [m]
    r1:
        The local r1 coordinate [m]
    r2: float
        The local r2 coordinate [m]

    Returns
    -------
    :
        The magnetic field response in the x coordinate direction
    """
    return (
        primitive_sxn(alpha, r1, q2, l1, l2)
        - primitive_sxn(alpha, r1, q1, l1, l2)
        + primitive_sxn(beta, r2, q2, l1, l2)
        - primitive_sxn(beta, r2, q1, l1, l2)
    )


@nb.jit(nopython=True, cache=True)
def Bz_analytical_prism(
    alpha: float,
    beta: float,
    l1: float,
    l2: float,
    q1: float,
    q2: float,
    r1: float,
    r2: float,
) -> float:
    """
    Calculate magnetic field in the local z coordinate direction due to a
    trapezoidal prism current source.

    Parameters
    ----------
    alpha:
        The first trapezoidal angle [rad]
    beta:
        The second trapezoidal angle [rad]
    l1:
        The local l1 coordinate [m]
    l2:
        The local l2 coordinate [m]
    q1:
        The local q1 coordinate [m]
    q2:
        The local q2 coordinate [m]
    r1:
        The local r1 coordinate [m]
    r2:
        The local r2 coordinate [m]

    Returns
    -------
    :
        The magnetic field response in the z coordinate direction
    """
    return (
        primitive_szn(alpha, r1, l2, q1, q2)
        - primitive_szn(alpha, r1, l1, q1, q2)
        + primitive_szn(beta, r2, l2, q1, q2)
        - primitive_szn(beta, r2, l1, q1, q2)
    )


# Vertex indices of each face of the trapezoidal prism, ordered anti-clockwise
# when viewed from outside the prism (i.e. right-handed about the outward normal).
# See TrapezoidalPrismCurrentSource._calculate_vertices for the vertex numbering.
_PRISM_FACES = np.array(
    [
        [0, 3, 2, 1],  # z' = -c
        [4, 5, 6, 7],  # z' = +c
        [0, 4, 7, 3],  # x' = -d
        [1, 2, 6, 5],  # x' = +d
        [0, 1, 5, 4],  # beta end-cap (current inlet)
        [3, 7, 6, 2],  # alpha end-cap (current outlet)
    ],
    dtype=np.int64,
)

# Vertex index polylines tracing the edges of the trapezoidal prism, for plotting
_PRISM_PLOT_EDGES = (
    [0, 1, 5, 4, 0],  # beta end-cap
    [3, 2, 6, 7, 3],  # alpha end-cap
    [0, 3],
    [1, 2],
    [5, 6],
    [4, 7],
)


@nb.jit(nopython=True, cache=True)
def _signed_solid_angle_triangle(
    r1: npt.NDArray[np.float64],
    r2: npt.NDArray[np.float64],
    r3: npt.NDArray[np.float64],
    n1: float,
    n2: float,
    n3: float,
) -> float:
    """
    Signed solid angle subtended by a triangle at the origin.

    Parameters
    ----------
    r1:
        First vertex relative to the observation point (3)
    r2:
        Second vertex relative to the observation point (3)
    r3:
        Third vertex relative to the observation point (3)
    n1:
        Norm of r1
    n2:
        Norm of r2
    n3:
        Norm of r3

    Returns
    -------
    :
        The signed solid angle [sr], positive when the observation point lies
        on the opposite side of the triangle to its right-handed normal
        (r2 - r1) x (r3 - r1)

    Notes
    -----
    Van Oosterom & Strackee formula:

    .. doi:: 10.1109/TBME.1983.325207
    """
    numerator = np.dot(r1, np.cross(r2, r3))
    denominator = (
        n1 * n2 * n3 + np.dot(r1, r2) * n3 + np.dot(r2, r3) * n1 + np.dot(r3, r1) * n2
    )
    return 2.0 * np.arctan2(numerator, denominator)


@nb.jit(nopython=True, cache=True)
def _polyhedron_inverse_distance_integral(
    vertices: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int64],
    point: npt.NDArray[np.float64],
) -> float:
    r"""
    Volume integral of 1/R over a polyhedron with planar faces, where R is the
    distance to the observation point.

    Parameters
    ----------
    vertices:
        The vertices of the polyhedron (n_vertices, 3)
    faces:
        Vertex indices of each face, ordered anti-clockwise when viewed from
        outside the polyhedron (n_faces, n_face_vertices)
    point:
        The observation point (3)

    Returns
    -------
    :
        The value of the integral [m^2]

    Notes
    -----
    Valid inside, outside, and on the surface of the polyhedron.

    .. math::
        \int_V \frac{dV}{R} = \frac{1}{2}\sum_f h_f
        \left[\sum_{e \in f} h_{fe} L_e - h_f \omega_f\right]

    where:

    - :math:`h_f` is the signed distance from the point to the plane of face f,
      positive when the point is on the inner side of the face,
    - :math:`h_{fe}` is the signed in-plane distance from the projection of the
      point onto the plane of face f to the line of edge e, positive when the
      projection is on the inner side of the edge,
    - :math:`L_e = \ln((R_1 + R_2 + l_e)/(R_1 + R_2 - l_e))`, with :math:`R_1` and
      :math:`R_2` the distances from the point to the ends of edge e, and
      :math:`l_e` its length,
    - :math:`\omega_f` is the signed solid angle subtended by face f, positive when
      the point is on the inner side of the face.

    :math:`L_e` is singular when the point lies on edge e (including its ends),
    where :math:`h_{fe} = 0`, so the term is set to 0. :math:`\omega_f` is
    discontinuous across the plane of face f, where :math:`h_f = 0`.
    """
    result = 0.0
    for face in faces:
        n_v = len(face)
        # Vertex positions relative to the observation point, and their distances
        r = vertices[face] - point
        r_norm = np.sqrt(np.sum(r**2, axis=1))

        # Outward unit normal of the (planar) face, and distance to its plane
        normal = np.cross(r[1] - r[0], r[2] - r[0])
        normal /= np.linalg.norm(normal)
        h_f = np.dot(normal, r[0])

        edge_sum = 0.0
        for i in range(n_v):
            j = (i + 1) % n_v
            edge = r[j] - r[i]
            length = np.linalg.norm(edge)
            # In-plane distance to the line of the edge (positive inside the face)
            h_fe = np.dot(np.cross(edge, normal), r[i]) / length
            r_sum = r_norm[i] + r_norm[j]
            # r_sum == length iff the point lies on the edge, where h_fe = 0 and the
            # (singular) term vanishes
            if r_sum > length:
                edge_sum += h_fe * np.log((r_sum + length) / (r_sum - length))

        # Solid angle of the face, by fan triangulation from the first vertex
        omega = 0.0
        for i in range(1, n_v - 1):
            omega += _signed_solid_angle_triangle(
                r[0], r[i], r[i + 1], r_norm[0], r_norm[i], r_norm[i + 1]
            )

        result += h_f * (edge_sum - h_f * omega)

    return 0.5 * result


class TrapezoidalPrismCurrentSource(PrismEndCapMixin, CrossSectionCurrentSource):
    """
    3-D trapezoidal prism current source with a rectangular cross-section and
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
        origin: npt.NDArray[np.float64],
        ds: npt.NDArray[np.float64],
        normal: npt.NDArray[np.float64],
        t_vec: npt.NDArray[np.float64],
        breadth: float,
        depth: float,
        alpha: float,
        beta: float,
        current: float,
    ):
        alpha, beta = np.deg2rad(alpha), np.deg2rad(beta)

        length = np.linalg.norm(ds)
        self._check_angle_values(alpha, beta)
        self._check_raise_self_intersection(float(length), breadth, alpha, beta)
        self._halflength = 0.5 * length
        # Normalised direction cosine matrix
        self._dcm = np.array([t_vec, ds / length, normal], dtype=float)
        # The field integration domain is centred at y' = 0.5 * breadth *
        # (tan(alpha) - tan(beta)) in the local coordinate system. Shift the local
        # origin so that the prism is centred on the specified origin.
        self._y_offset = 0.5 * breadth * (np.tan(alpha) - np.tan(beta))
        self._origin = np.asarray(origin, dtype=float) - self._y_offset * self._dcm[1]
        self._length = 0.5 * (length - breadth * np.tan(alpha) - breadth * np.tan(beta))
        self._breadth = breadth
        self._depth = depth
        self._alpha = alpha
        self._beta = beta
        # Current density
        self._area = 4 * breadth * depth
        self.set_current(current)
        self._vertices = self._calculate_vertices()

    def _xyzlocal_to_rql(
        self, x_local: float, y_local: float, z_local: float
    ) -> tuple[float, float, float, float, float, float]:
        """
        Convert local x, y, z coordinates to working coordinates.

        Returns
        -------
        :
            Working coordinates
        """
        b = self._length
        c = self._depth
        d = self._breadth

        l1 = -d - x_local
        l2 = d - x_local
        q1 = -c - z_local
        q2 = c - z_local
        r1 = (d + x_local) * np.tan(self._alpha) + b - y_local
        r2 = (d + x_local) * np.tan(self._beta) + b + y_local
        return l1, l2, q1, q2, r1, r2

    def _BxByBz(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculate the field at a point in local coordinates.

        Returns
        -------
        :
            (Bx, By, Bz) at a point

        Note
        ----
            By set to 0.
        """
        l1, l2, q1, q2, r1, r2 = self._xyzlocal_to_rql(*point)
        bx = Bx_analytical_prism(self._alpha, self._beta, l1, l2, q1, q2, r1, r2)
        bz = Bz_analytical_prism(self._alpha, self._beta, l1, l2, q1, q2, r1, r2)
        return np.array([bx, 0, bz])

    @process_xyz_array
    def field(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
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
        :
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        point = np.array([x, y, z])
        # Convert to local coordinates
        point = self._global_to_local(np.array([point]))[0]
        # Evaluate field in local coordinates
        b_local = MU_0_4PI * self._rho * self._BxByBz(point)
        # Convert vector back to global coordinates
        return self._dcm.T @ b_local

    @process_xyz_array
    def vector_potential(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
        r"""
        Calculate the vector potential at a point due to the current source.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the potential
        y:
            The y coordinate(s) of the points at which to calculate the potential
        z:
            The z coordinate(s) of the points at which to calculate the potential

        Returns
        -------
        :
            The vector potential {Ax, Ay, Az} in [V.s/m]

        Notes
        -----
        The current density is uniform and along the local y coordinate, so in local
        coordinates:

        .. math::
            A_y = \frac{\mu_0 J}{4\pi}\int_V \frac{dV}{R}, \quad A_x = A_z = 0

        This is the potential whose curl is the Biot-Savart field of this source
        (see the field method), and is valid inside the conductor.
        """
        point = np.array([x, y, z])
        # Convert to local coordinates
        point = self._global_to_local(np.array([point]))[0]
        # Evaluate vector potential in local coordinates
        a_y = _polyhedron_inverse_distance_integral(self._vertices, _PRISM_FACES, point)
        a_local = MU_0_4PI * self._rho * np.array([0.0, a_y, 0.0])
        # Convert vector back to global coordinates
        return self._dcm.T @ a_local

    def _calculate_vertices(self) -> npt.NDArray[np.float64]:
        """
        Calculate the vertices of the trapezoidal prism in local coordinates, as
        used in the integration domain of the field calculation and for plotting.

        Returns
        -------
        :
            The vertices (8, 3), numbered as in _PRISM_FACES

        Notes
        -----
        Integration domain:
            x' in [-d, d], z' in [-c, c],
            y' in [-(x' + d) tan(beta) - b, (x' + d) tan(alpha) + b]
        """
        b = self._length
        c = self._depth
        d = self._breadth
        y_lo = -2 * d * np.tan(self._beta) - b
        y_hi = 2 * d * np.tan(self._alpha) + b
        return np.array([
            [-d, -b, -c],
            [d, y_lo, -c],
            [d, y_hi, -c],
            [-d, b, -c],
            [-d, -b, c],
            [d, y_lo, c],
            [d, y_hi, c],
            [-d, b, c],
        ])

    @property
    def _centre(self) -> npt.NDArray[np.float64]:
        """
        The centre of the current source in global coordinates (the specified origin)
        """
        return self._origin + self._y_offset * self._dcm[1]

    @property
    def _points(self) -> npt.NDArray[np.float64]:
        """
        Edge lines of the current source in global coordinates, for plotting.

        Returns
        -------
        :
            Array of (n, 3) polylines
        """
        global_vertices = self._local_to_global(self._vertices)
        return np.array(
            [global_vertices[edge] for edge in _PRISM_PLOT_EDGES], dtype=object
        )

    def rotate(self, angle: float, axis: npt.NDArray[np.float64] | str):
        """
        Rotate the CurrentSource about an axis.

        Parameters
        ----------
        angle:
            The rotation degree [degree]
        axis:
            The axis of rotation
        """
        r = rotation_matrix(np.deg2rad(angle), axis).T
        self._origin @= r
        self._dcm @= r

    def translate(self, vector: npt.NDArray[np.float64]):
        """
        Translate the CurrentSource.

        Parameters
        ----------
        vector:
            Translation vector

        Raises
        ------
        MagnetostaticsError
            If the translation vector is not of length 3
        """
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (3,):
            raise MagnetostaticsError(
                f"Translation vector must be of shape (3,), not {vector.shape}."
            )
        self._origin += vector

    def plot(self, ax: Axes | None = None, *, show_coord_sys: bool = False, **kwargs):  # noqa: ARG002
        """
        Plot the CurrentSource.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        show_coord_sys: bool
            Whether or not to plot the coordinate systems
        """
        if ax is None:
            ax = Plot3D()
        super().plot(ax=ax, show_coord_sys=False)
        centre = self._centre
        if show_coord_sys:
            ax.scatter(*centre, color="k")
            for axis in self._dcm:
                ax.quiver(*centre, *axis, length=1, color="r")
        p1 = centre - 0.5 * self._dcm[1]
        p2 = centre + 0.5 * self._dcm[1]
        points = np.array([p1, p2])
        ax.plot(*points.T, color="r")
        ax.plot([points[-1][0]], [points[-1][1]], [points[-1][2]], marker="^", color="r")

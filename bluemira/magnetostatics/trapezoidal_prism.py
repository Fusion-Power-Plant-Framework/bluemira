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
"""

import numba as nb
import numpy as np
import numpy.typing as npt

from bluemira.base.constants import MU_0_4PI
from bluemira.magnetostatics.baseclass import CrossSectionCurrentSource, PrismEndCapMixin
from bluemira.magnetostatics.tools import process_xyz_array

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
        self._origin = origin

        length = np.linalg.norm(ds)
        self._check_angle_values(alpha, beta)
        self._check_raise_self_intersection(length, breadth, alpha, beta)
        self._halflength = 0.5 * length
        # Normalised direction cosine matrix
        self._dcm = np.array([t_vec, ds / length, normal], dtype=float)
        self._length = 0.5 * (length - breadth * np.tan(alpha) - breadth * np.tan(beta))
        self._breadth = breadth
        self._depth = depth
        self._alpha = alpha
        self._beta = beta
        # Current density
        self._area = 4 * breadth * depth
        self.set_current(current)
        self._points = self._calculate_points()

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
        point = self._global_to_local([point])[0]
        # Evaluate field in local coordinates
        b_local = MU_0_4PI * self._rho * self._BxByBz(point)
        # Convert vector back to global coordinates
        return self._dcm.T @ b_local

    def _calculate_points(self) -> npt.NDArray[np.float64]:
        """
        Calculate extrema points of the current source for plotting and debugging.

        Returns
        -------
        :
            extrema points
        """
        b = self._halflength
        c = self._depth
        d = self._breadth
        # Lower rectangle
        p1 = np.array([-d, -b + d * np.tan(self._beta), -c])
        p2 = np.array([d, -b - d * np.tan(self._beta), -c])
        p3 = np.array([d, -b - d * np.tan(self._beta), c])
        p4 = np.array([-d, -b + d * np.tan(self._beta), c])

        # Upper rectangle
        p5 = np.array([-d, b - d * np.tan(self._alpha), -c])
        p6 = np.array([d, b + d * np.tan(self._alpha), -c])
        p7 = np.array([d, b + d * np.tan(self._alpha), c])
        p8 = np.array([-d, b - d * np.tan(self._alpha), c])

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

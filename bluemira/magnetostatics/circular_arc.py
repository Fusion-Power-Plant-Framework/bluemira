# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Analytical expressions for the field due to a circular current arc of
rectangular cross-section, following equations as described in:

.. doi:: 10.1109/TMAG.1985.1064259

The analytical treatment of singularities in Feng's 1988 paper proved,
despite my best efforts, to be impractical, at the very least. Whilst I
cannot claim the analytical treatments of the singularities are wrong
or incomplete, I can say that after many attempts my implementations
still ran into numerical singularities, and the values returned did not
correspond well to results found with a 2-D semi-analytical methods.
Instead, a brute-force integration takes places in the vast majority of
singular cases (as described by Feng). This is remarkably accurate when
comparing results to fields obtained by other methods; of the
order of 10 µT/MA discrepancy at singular points. MC 2026
"""  # noqa: RUF002, yes I meant micro

from itertools import pairwise

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import numpy.typing as npt
from scipy import LowLevelCallable

from bluemira.base.constants import EPS, MU_0_4PI
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry._private_tools import make_circle_arc
from bluemira.magnetostatics.baseclass import CrossSectionCurrentSource
from bluemira.magnetostatics.error import MagnetostaticsIntegrationError
from bluemira.magnetostatics.tools import (
    integrate,
    jit_llc4,
    jit_llc6,
    process_xyz_array,
)

__all__ = ["CircularArcCurrentSource"]


LOG_EPS = np.finfo(float).tiny
TWO_PI = 2.0 * np.pi
SPLIT_TOL = 1e-12  # [rad]

# Full integrands free of singularities


@nb.njit(cache=True)
def _sqrt_and_log_arg(
    sin_psi: float, cos_psi: float, r_pc: float, r_j: float, z_k: float
) -> tuple[float, float]:
    """
    Calculate the shared square-root and logarithm argument stably.

    Parameters
    ----------
    sin_psi:
        Sine of the angle
    cos_psi:
        Cosine of the angle
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]

    Returns
    -------
    :
        The square-root term and stable logarithm argument.
    """
    offset = r_j - r_pc * cos_psi
    transverse_sq = (r_pc * sin_psi) ** 2 + z_k**2
    sqrt_term = np.sqrt(offset**2 + transverse_sq)

    if offset < 0 and transverse_sq > 0:
        log_arg = transverse_sq / (sqrt_term - offset)
    else:
        log_arg = offset + sqrt_term

    return sqrt_term, log_arg


@jit_llc4
def brc_integrand_full(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the Brc integrand without singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    :
        The result of the integrand at a single point
    """
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    sqrt_term, log_arg = _sqrt_and_log_arg(sin_psi, cos_psi, r_pc, r_j, z_k)
    if log_arg <= 0:
        log_arg = LOG_EPS
    return cos_psi * sqrt_term + r_pc * cos_psi**2 * np.log(log_arg)


@jit_llc4
def bzc_integrand_full_p1(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the Bzc integrand without singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    sqrt_term, log_arg_1 = _sqrt_and_log_arg(sin_psi, cos_psi, r_pc, r_j, z_k)
    term_1 = 0.0
    if z_k != 0:
        if log_arg_1 <= 0:
            log_arg_1 = LOG_EPS

        term_1 = -z_k * np.log(log_arg_1)

    log_arg_2 = -z_k + sqrt_term
    if z_k > 0:
        radial_offset = r_j - r_pc * cos_psi
        radial_sq = radial_offset**2 + (r_pc * sin_psi) ** 2
        if radial_sq > 0:
            log_arg_2 = radial_sq / (sqrt_term + z_k)
    if log_arg_2 <= 0:
        log_arg_2 = LOG_EPS
    return term_1 - r_pc * cos_psi * np.log(log_arg_2)


@jit_llc4
def bf3_integrand(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the BF3 integrand

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point

    Notes
    -----
    Treats the sin(psi) = 0 singularity
    """
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    if sin_psi != 0:
        sqrt_term, _ = _sqrt_and_log_arg(sin_psi, cos_psi, r_pc, r_j, z_k)
        # NOTE: Arctan2 not a viable option here
        return (
            r_pc
            * sin_psi
            * np.arctan((z_k * (r_j - r_pc * cos_psi)) / (r_pc * sin_psi * sqrt_term))
        )
    return 0


@jit_llc4
def btc_integrand_full(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the Btc integrand without singularities.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius, inner or outer, of the coil [m]
    z_k:
        The z coordinate, upper or lower, of the coil [m]

    Returns
    -------
    :
        The result of the integrand at a single point
    """
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    sqrt_term, log_arg = _sqrt_and_log_arg(sin_psi, cos_psi, r_pc, r_j, z_k)
    return sin_psi * sqrt_term + r_pc * sin_psi * cos_psi * np.log(log_arg)


# Primitive functions


def primitive_brc(
    r_pc: float, r_j: float, z_k: float, phi_pc: float, theta: float
) -> float:
    """
    Calculate the Brc primitives and treat singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    phi_pc:
        Angle of the point at which to evaluate field [rad]
    theta:
        Azimuthal angle of the circular arc

    Returns
    -------
    The result of the Brc primitive
    """
    args = (r_pc, r_j, z_k)  # The function arguments for integration
    return integrate(brc_integrand_full, args, -phi_pc, theta - phi_pc)


def primitive_btc(
    r_pc: float, r_j: float, z_k: float, phi_pc: float, theta: float
) -> float:
    """
    Calculate the Btc primitives and treat singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius, inner or outer, of the coil [m]
    z_k:
        The z coordinate, upper or lower, of the coil [m]
    phi_pc:
        Angle of the point at which to evaluate field [rad]
    theta:
        Azimuthal angle of the circular arc [rad]

    Returns
    -------
    :
        The result of the Btc primitive.
    """
    args = (r_pc, r_j, z_k)
    return integrate(btc_integrand_full, args, -phi_pc, theta - phi_pc)


def primitive_bzc(
    r_pc: float, r_j: float, z_k: float, phi_pc: float, theta: float
) -> float:
    """
    Calculate the Bzc primitives and treat singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    phi_pc:
        Angle of the point at which to evaluate field [rad]
    theta:
        Azimuthal angle of the circular arc

    Returns
    -------
    The result of the Bzc primitive
    """
    args = (r_pc, r_j, z_k)  # The function arguments for integration
    result = integrate(bzc_integrand_full_p1, args, -phi_pc, theta - phi_pc)
    if z_k != 0 and r_pc != 0:
        # The only singularities we now bother to catch (and they all = 0 if hit)
        # z_k == 0 -> 0
        # r_pc == 0 -> 0
        # This gets rid of zero division errors in the integration
        result += integrate(bf3_integrand, args, -phi_pc, theta - phi_pc)
    return result


# Full field calculations in working coordinates


def Bx_analytical_circular(
    r1: float, r2: float, z1: float, z2: float, theta: float, r_p: float, theta_p: float
) -> float:
    """
    Calculate magnetic field in the local x coordinate direction due to a
    circular arc current source.

    Parameters
    ----------
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z1:
        The first modified z coordinate [m]
    z2:
        The second modified z coordinate [m]
    theta:
        Azimuthal angle of the circular arc [rad]
    r_p:
        The radius of the point at which to evaluate the field [m]
    theta_p:
        The angle of the point at which to evaluate the field [rad]

    Returns
    -------
    The magnetic field response in the x coordinate direction
    """
    return (
        primitive_brc(r_p, r1, z1, theta_p, theta)
        - primitive_brc(r_p, r1, z2, theta_p, theta)
        - primitive_brc(r_p, r2, z1, theta_p, theta)
        + primitive_brc(r_p, r2, z2, theta_p, theta)
    )


def Bt_analytical_circular(
    r1: float, r2: float, z1: float, z2: float, theta: float, r_p: float, theta_p: float
) -> float:
    """
    Calculate magnetic field in the local tangential coordinate direction due
    to a circular arc current source.

    Parameters
    ----------
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z1:
        The first modified z coordinate [m]
    z2:
        The second modified z coordinate [m]
    theta:
        Azimuthal angle of the circular arc [rad]
    r_p:
        The radius of the point at which to evaluate the field [m]
    theta_p:
        The angle of the point at which to evaluate the field [rad]

    Returns
    -------
    :
        The magnetic field response in the local tangential direction.
    """
    return (
        primitive_btc(r_p, r1, z1, theta_p, theta)
        - primitive_btc(r_p, r1, z2, theta_p, theta)
        - primitive_btc(r_p, r2, z1, theta_p, theta)
        + primitive_btc(r_p, r2, z2, theta_p, theta)
    )


def Bz_analytical_circular(
    r1: float, r2: float, z1: float, z2: float, theta: float, r_p: float, theta_p: float
) -> float:
    """
    Calculate magnetic field in the local z coordinate direction due to a
    circular arc current source.

    Parameters
    ----------
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z1:
        The first modified z coordinate [m]
    z2:
        The second modified z coordinate [m]
    theta:
        Azimuthal angle of the circular arc [rad]
    r_p:
        The radius of the point at which to evaluate the field [m]
    theta_p:
        The angle of the point at which to evaluate the field [rad]

    Returns
    -------
    The magnetic field response in the z coordinate direction
    """
    return (
        primitive_bzc(r_p, r1, z1, theta_p, theta)
        - primitive_bzc(r_p, r1, z2, theta_p, theta)
        - primitive_bzc(r_p, r2, z1, theta_p, theta)
        + primitive_bzc(r_p, r2, z2, theta_p, theta)
    )


# Vector potential


@nb.jit(nopython=True, cache=True)
def _potential_primitive(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the vector potential cross-section primitive at a corner.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate the potential [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]

    Returns
    -------
    :
        The value of the primitive at a single point

    Notes
    -----
    Each term is skipped where its denominator is zero; its limit there is 0.
    """
    cos_psi = np.cos(psi)
    c = r_pc * cos_psi
    s = r_pc * np.sin(psi)
    u = r_j - c
    rho_sq = u**2 + s**2
    q_sq = s**2 + z_k**2
    dist = np.sqrt(rho_sq + z_k**2)

    result = 0.5 * z_k * dist
    if rho_sq > 0:
        result += 0.5 * (r_j**2 - c**2 + s**2) * np.arcsinh(z_k / np.sqrt(rho_sq))
    if q_sq > 0:
        result += c * z_k * np.arcsinh(u / np.sqrt(q_sq))
    if s != 0:
        result -= c * s * np.arctan(u * z_k / (s * dist))
    return result


@nb.jit(nopython=True, cache=True)
def _potential_kernel(
    psi: float, r_pc: float, r1: float, r2: float, z1: float, z2: float
) -> float:
    """
    Calculate the integral of r'/R over the cross-section, at a given angle.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate the potential [m]
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z1:
        The first modified z coordinate [m]
    z2:
        The second modified z coordinate [m]

    Returns
    -------
    :
        The cross-section integral [m^2]
    """
    return (
        _potential_primitive(psi, r_pc, r2, z1)
        - _potential_primitive(psi, r_pc, r2, z2)
        - _potential_primitive(psi, r_pc, r1, z1)
        + _potential_primitive(psi, r_pc, r1, z2)
    )


@jit_llc6
def ar_integrand(
    psi: float, r_pc: float, r1: float, r2: float, z1: float, z2: float
) -> float:
    """
    Calculate the radial vector potential integrand.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate the potential [m]
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z1:
        The first modified z coordinate [m]
    z2:
        The second modified z coordinate [m]

    Returns
    -------
    :
        The result of the integrand at a single point
    """
    return -np.sin(psi) * _potential_kernel(psi, r_pc, r1, r2, z1, z2)


@jit_llc6
def at_integrand(
    psi: float, r_pc: float, r1: float, r2: float, z1: float, z2: float
) -> float:
    """
    Calculate the tangential vector potential integrand.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate the potential [m]
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z1:
        The first modified z coordinate [m]
    z2:
        The second modified z coordinate [m]

    Returns
    -------
    :
        The result of the integrand at a single point
    """
    return np.cos(psi) * _potential_kernel(psi, r_pc, r1, r2, z1, z2)


def _integrate_split(
    func: LowLevelCallable, args: tuple[float, ...], bound1: float, bound2: float
) -> float:
    """
    Integrate between bounds, always splitting at multiples of pi, where the
    vector potential integrands can be kinked or sharply peaked.

    Parameters
    ----------
    func:
        The function to integrate
    args:
        The static arguments to the function
    bound1:
        The first integration bound [rad]
    bound2:
        The second integration bound [rad]

    Returns
    -------
    :
        The value of the integral of the function between the bounds
    """
    lower, upper = min(bound1, bound2), max(bound1, bound2)
    n = np.arange(np.ceil(lower / np.pi), np.floor(upper / np.pi) + 1)
    splits = n * np.pi
    # Avoid slivers when a bound is a multiple of pi to within rounding
    splits = splits[(splits - lower > SPLIT_TOL) & (upper - splits > SPLIT_TOL)]
    result = sum(
        integrate(func, args, a, b) for a, b in pairwise([lower, *splits, upper])
    )
    return np.sign(bound2 - bound1) * result


class CircularArcCurrentSource(CrossSectionCurrentSource):
    """
    3-D circular arc prism current source with a rectangular cross-section and
    uniform current distribution.

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
    radius:
        The radius of the circular arc from the origin [m]
    dtheta:
        The azimuthal width of the arc [°]
    current:
        The current flowing through the source [A]

    Notes
    -----
    The origin is at the centre of the circular arc, with the ds vector pointing
    towards the start of the circular arc.

    Cylindrical coordinates are used for calculations under the hood.
    """

    def __init__(
        self,
        origin: npt.NDArray[np.float64],
        ds: npt.NDArray[np.float64],
        normal: npt.NDArray[np.float64],
        t_vec: npt.NDArray[np.float64],
        breadth: float,
        depth: float,
        radius: float,
        dtheta: float,
        current: float,
    ):
        self._origin = origin
        self._breadth = breadth
        self._depth = depth
        self._length = 0.5 * (breadth + depth)  # For plotting only
        self._radius = radius
        self._update_r1r2()

        self._dtheta = np.deg2rad(dtheta)
        self._rho = current / (4 * breadth * depth)
        self._dcm = np.array([ds, normal, t_vec], dtype=float)
        self._points = self._calculate_points()

    @property
    def radius(self) -> float:
        """
        The radius of the CircularArcCurrentSource
        """
        return self._radius

    @radius.setter
    def radius(self, radius: float):
        """
        Set the radius.

        Parameters
        ----------
        radius:
            The radius of the CircularArcCurrentSource
        """
        self._radius = radius
        self._update_r1r2()

    @property
    def breadth(self) -> float:
        """
        The breadth of the CircularArcCurrentSource
        """
        return self._breadth

    @breadth.setter
    def breadth(self, breadth: float):
        """
        Set the breadth of the CircularArcCurrentSource.

        Parameters
        ----------
        breadth:
            The breadth of the CircularArcCurrentSource
        """
        self._breadth = breadth
        self._update_r1r2()

    def _update_r1r2(self):
        """
        Update
        """
        self._r1 = self.radius - self.breadth
        self._r2 = self.radius + self.breadth

    @staticmethod
    def _local_to_cylindrical(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Convert from local to cylindrical coordinates.

        Returns
        -------
        :
            Cylindrical coordinates of point.
        """
        x, y, z = point
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return np.array([rho, theta, z])

    def _cylindrical_to_working(self, zp: float) -> tuple[float, float, float, float]:
        """
        Convert from local cylindrical coordinates to working coordinates.

        Returns
        -------
        :
            r +- breadth and z +- depth.
        """
        z1 = zp + self._depth
        z2 = zp - self._depth
        return self._r1, self._r2, z1, z2

    def _BxByBz(self, rp: float, tp: float, zp: float) -> npt.NDArray[np.float64]:
        """
        Calculate the field at a point in local coordinates.

        Returns
        -------
        :
            (Bx, By, Bz) at a point
        """
        r1, r2, z1, z2 = self._cylindrical_to_working(zp)

        br = Bx_analytical_circular(r1, r2, z1, z2, self._dtheta, rp, tp)
        bz = Bz_analytical_circular(r1, r2, z1, z2, self._dtheta, rp, tp)
        if np.isclose(abs(self._dtheta), TWO_PI, rtol=0.0, atol=EPS):
            bt = 0.0
        else:
            bt = Bt_analytical_circular(r1, r2, z1, z2, self._dtheta, rp, tp)

        bx = br * np.cos(tp) - bt * np.sin(tp)
        by = br * np.sin(tp) + bt * np.cos(tp)

        return np.array([bx, by, bz])

    @process_xyz_array
    def field(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
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
        # Convert to local cylindrical coordinates
        point = self._global_to_local(np.array([point]))[0]
        rp, tp, zp = self._local_to_cylindrical(point)
        # Calculate field in local coordinates
        try:
            b_local = MU_0_4PI * self._rho * self._BxByBz(rp, tp, zp)

        except MagnetostaticsIntegrationError as e:
            # If all else fails, perform an 8-point Gauss-Legendre volume-averaged
            # calculation.
            # So far, these have only been triggered on "surprising" singularities
            # not located on the surface or inside of the source.
            bluemira_warn(
                f"{e!s} \nFallback triggered: 8-point Gauss-Legendre volume-averaged"
                f" field being return for point at {x=:.6f}, {y=:.6f}, {z=:.6f}"
            )
            offset = 1e-6 / np.sqrt(3)
            quadrature_points = (
                np.array([x + dx, y + dy, z + dz])
                for dx in (-offset, offset)
                for dy in (-offset, offset)
                for dz in (-offset, offset)
            )

            b = np.zeros(3)
            for p in quadrature_points:
                point = self._global_to_local(np.asarray([p]))[0]
                rp, tp, zp = self._local_to_cylindrical(point)
                b += self._BxByBz(rp, tp, zp)

            b_local = MU_0_4PI * self._rho * b / 8

        return self._dcm.T @ b_local

    def _vector_potential_local(
        self, rp: float, tp: float, zp: float
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the vector potential at a point in local coordinates.

        Returns
        -------
        :
            (Ax, Ay, Az) at a point

        Notes
        -----
        The potential is calculated without its leading mu_0 J / (4 pi) factor.
        Az is 0.
        """
        args = (rp, *self._cylindrical_to_working(zp))
        lower, upper = -tp, self._dtheta - tp

        at = _integrate_split(at_integrand, args, lower, upper)
        if np.isclose(abs(self._dtheta), TWO_PI, rtol=0.0, atol=EPS):
            ar = 0.0
        else:
            ar = _integrate_split(ar_integrand, args, lower, upper)

        ax = ar * np.cos(tp) - at * np.sin(tp)
        ay = ar * np.sin(tp) + at * np.cos(tp)

        return np.array([ax, ay, 0.0])

    @process_xyz_array
    def vector_potential(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
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
        With :math:`\psi = \phi' - \phi_{pc}` and uniform tangential current density:

        .. math::
            A_r &= -\frac{\mu_0 J}{4\pi}
            \int_{-\phi_{pc}}^{\theta-\phi_{pc}} \sin\psi~G(\psi)~d\psi

            A_t &= \frac{\mu_0 J}{4\pi}
            \int_{-\phi_{pc}}^{\theta-\phi_{pc}} \cos\psi~G(\psi)~d\psi

            G(\psi) &= \iint \frac{r'}{R}~dr'dz'
            = -\sum_{j=1}^{2}\sum_{k=1}^{2}(-1)^{j+k} F(R_j, Z_k)

            F &= \frac{Z_k D}{2} + \frac{R_j^2 - R_{pc}^2\cos 2\psi}{2}
            \sinh^{-1}\frac{Z_k}{\sqrt{u^2+s^2}}
            + c Z_k\sinh^{-1}\frac{u}{\sqrt{s^2+Z_k^2}}
            - c s\tan^{-1}\frac{u Z_k}{s D}

        where :math:`c = R_{pc}\cos\psi`, :math:`s = R_{pc}\sin\psi`,
        :math:`u = R_j - c`, and :math:`D = \sqrt{u^2 + s^2 + Z_k^2}`.

        The integrands are bounded everywhere, including inside the conductor and
        on its faces, edges and corners.

        The curl of this potential is the field of this source. The potential of
        an open arc is not divergence-free at its ends.

        G is a difference of corner values, so precision reduces at points very far
        from the source relative to the size of its cross-section.
        """
        point = np.array([x, y, z])
        # Convert to local cylindrical coordinates
        point = self._global_to_local(np.array([point]))[0]
        rp, tp, zp = self._local_to_cylindrical(point)
        # Calculate vector potential in local coordinates
        a_local = MU_0_4PI * self._rho * self._vector_potential_local(rp, tp, zp)
        # Convert vector back to global coordinates
        return self._dcm.T @ a_local

    def _calculate_points(self) -> npt.NDArray[np.float64]:
        """
        Calculate extrema points of the current source for plotting and debugging.

        Returns
        -------
        :
            extrema points
        """
        r = self.radius
        a = self.breadth
        b = self._depth

        # Circle arcs
        n = 200
        theta = self._dtheta
        ones = np.ones(n)
        arc_1x, arc_1y = make_circle_arc(r - a, 0, 0, angle=theta, n_points=n)
        arc_2x, arc_2y = make_circle_arc(r + a, 0, 0, angle=theta, n_points=n)
        arc_3x, arc_3y = make_circle_arc(r + a, 0, 0, angle=theta, n_points=n)
        arc_4x, arc_4y = make_circle_arc(r - a, 0, 0, angle=theta, n_points=n)
        arc_1 = np.array([arc_1x, arc_1y, -b * ones]).T
        arc_2 = np.array([arc_2x, arc_2y, -b * ones]).T
        arc_3 = np.array([arc_3x, arc_3y, b * ones]).T
        arc_4 = np.array([arc_4x, arc_4y, b * ones]).T

        n_slices = int(2 + self._dtheta // (0.25 * np.pi))
        slices = np.linspace(0, n - 1, n_slices, endpoint=True, dtype=int)
        points = [arc_1, arc_2, arc_3, arc_4]

        # Rectangles
        points.extend([
            np.vstack([arc_1[s], arc_2[s], arc_3[s], arc_4[s], arc_1[s]]) for s in slices
        ])

        return np.array([self._local_to_global(p) for p in points], dtype=object)

    def plot(
        self, ax: plt.Axes | None = None, *, show_coord_sys: bool = False, **kwargs
    ):
        """
        Plot the CircularArcCurrentSource.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        show_coord_sys: bool
            Whether or not to plot the coordinate systems
        """
        super().plot(ax=ax, show_coord_sys=show_coord_sys, **kwargs)
        ax = plt.gca()
        theta = self._dtheta
        x, y = make_circle_arc(
            self.radius, 0, 0, angle=theta / 2, start_angle=theta / 4, n_points=200
        )
        centre_arc = np.array([x, y, np.zeros(200)]).T
        points = self._local_to_global(centre_arc)
        ax.plot(*points.T, color="r")
        ax.plot([points[-1][0]], [points[-1][1]], [points[-1][2]], marker="^", color="r")

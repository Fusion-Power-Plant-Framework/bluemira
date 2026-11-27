# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
r"""
Vector potential and magnetic field of a circular current arc of rectangular
cross-section with uniform azimuthal current density.

The integrals over the cross-section are done analytically, leaving one integral
over angle, following Urankar (:doi:`10.1109/TMAG.1982.1062166`). The field
integrands are derivatives of the vector potential kernel. The full derivation is in
the magnetostatics section of the documentation.

Numerical treatment
-------------------
* The corner sums are evaluated inside the integrand, with the
  :math:`\sinh^{-1}` terms grouped in pairs that share a denominator, so that their
  logarithmic parts cancel analytically between corners rather than numerically.
  Degenerate denominators are replaced by their analytic limits.
* What remains are genuine, integrable logarithmic singularities at
  :math:`\psi = 0` when the point lies on a face of the conductor. Close to a face,
  these become peaks of width :math:`\sim |Z_k| / r_{pc}` (z faces) or
  :math:`\sim |r_j - r_{pc}| / r_{pc}` (r faces). The adaptive quadrature is given
  breakpoints at multiples of :math:`\pi` (the :math:`|\sin\psi|` kinks) and, for
  narrow peaks, at logarithmically graded offsets around multiples of
  :math:`2\pi`. :math:`B_r`, :math:`B_t` (from :math:`\partial G/\partial z_p`)
  peak only at the z faces; :math:`B_z`, :math:`A_r` and :math:`A_t` peak at both.
* The kernels scale with the cross-sectional area, so the absolute quadrature
  tolerance is scaled by :math:`4 b d`.
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import numpy.typing as npt

from bluemira.base.constants import EPS, MU_0_4PI
from bluemira.geometry._private_tools import make_circle_arc
from bluemira.magnetostatics.baseclass import CrossSectionCurrentSource
from bluemira.magnetostatics.tools import integrate, jit_llc6, process_xyz_array

__all__ = ["CircularArcCurrentSource"]


TWO_PI = 2.0 * np.pi
SPLIT_TOL = 1e-12  # [rad]
# Absolute quadrature tolerance per unit cross-sectional area: the kernels scale
# with the area, so a fixed absolute tolerance is too loose for thin conductors
QUAD_EPSABS_PER_AREA = 1e-9
# Maximum number of subintervals of each quadrature
QUAD_LIMIT = 200
# Graded breakpoints are placed around logarithmic peaks narrower than this [rad]
MAX_PEAK_WIDTH = 1e-2

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


# Field from the derivatives of the vector potential kernel


@nb.jit(nopython=True, cache=True)
def _dfdz_radial_pair(psi: float, r_pc: float, r1: float, r2: float, z_k: float):
    """
    Calculate dF/dZ(r2, z_k) - dF/dZ(r1, z_k), where dF/dZ = D + c asinh(u / q).

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate the field [m]
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]

    Returns
    -------
    :
        The radial difference of the primitive at a single point

    Notes
    -----
    The log(q) parts of the two asinh terms cancel analytically when u1 and u2
    have the same sign. Where q = 0 and they do not, the integrand has a genuine
    (integrable) logarithmic singularity in psi.
    """
    c = r_pc * np.cos(psi)
    s = r_pc * np.sin(psi)
    u1 = r1 - c
    u2 = r2 - c
    q_sq = s**2 + z_k**2
    result = np.sqrt(u2**2 + q_sq) - np.sqrt(u1**2 + q_sq)
    if c != 0:
        if q_sq > 0:
            q = np.sqrt(q_sq)
            result += c * (np.arcsinh(u2 / q) - np.arcsinh(u1 / q))
        elif u1 * u2 > 0:
            result += c * np.sign(u2) * np.log(u2 / u1)
        else:
            result += c * (np.sign(u2) - np.sign(u1)) * np.inf
    return result


@nb.jit(nopython=True, cache=True)
def _field_z_kernel(
    psi: float, r_pc: float, r1: float, r2: float, z1: float, z2: float
) -> float:
    """
    Calculate dG/dz_p, the z-derivative of the vector potential kernel.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate the field [m]
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
        The cross-section integral of dG/dz_p [m]
    """
    return _dfdz_radial_pair(psi, r_pc, r1, r2, z1) - _dfdz_radial_pair(
        psi, r_pc, r1, r2, z2
    )


@jit_llc6
def br_integrand(
    psi: float, r_pc: float, r1: float, r2: float, z1: float, z2: float
) -> float:
    """
    Calculate the radial field integrand, Br = -dA_t/dz.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate the field [m]
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
    return -np.cos(psi) * _field_z_kernel(psi, r_pc, r1, r2, z1, z2)


@jit_llc6
def bt_integrand(
    psi: float, r_pc: float, r1: float, r2: float, z1: float, z2: float
) -> float:
    """
    Calculate the tangential field integrand, Bt = dA_r/dz.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate the field [m]
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
    return -np.sin(psi) * _field_z_kernel(psi, r_pc, r1, r2, z1, z2)


@nb.jit(nopython=True, cache=True)
def _asinh_pair(a: float, b: float, d: float) -> float:
    """
    Calculate asinh(a / d) - asinh(b / d) for d >= 0, taking the d -> 0 limit.

    Parameters
    ----------
    a:
        First numerator
    b:
        Second numerator
    d:
        Common denominator (non-negative)

    Returns
    -------
    :
        The difference, which is infinite where d = 0 and a, b differ in sign
    """
    if d > 0:
        return np.arcsinh(a / d) - np.arcsinh(b / d)
    if a * b > 0:
        return np.sign(a) * np.log(a / b)
    return (np.sign(a) - np.sign(b)) * np.inf


@jit_llc6
def bz_integrand(
    psi: float, r_pc: float, r1: float, r2: float, z1: float, z2: float
) -> float:
    """
    Calculate the axial field integrand, Bz = e_r'.grad(G), from the corner sum of
    dF/dc = -c asinh(Z / rho) + Z asinh(u / q) - s arctan(u Z / (s D)).

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate the field [m]
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

    Notes
    -----
    The asinh terms are grouped in pairs so that their log parts cancel
    analytically where they should. What remains are genuine (integrable)
    logarithmic singularities in psi at psi = 0 on the faces of the conductor.
    """
    c = r_pc * np.cos(psi)
    s = r_pc * np.sin(psi)
    u1 = r1 - c
    u2 = r2 - c
    result = 0.0

    # Z asinh(u / q), paired over r_j at fixed z_k
    if z1 != 0:
        result += z1 * _asinh_pair(u2, u1, np.sqrt(s**2 + z1**2))
    if z2 != 0:
        result -= z2 * _asinh_pair(u2, u1, np.sqrt(s**2 + z2**2))

    # -c asinh(Z / rho), paired over z_k at fixed r_j
    if c != 0:
        result -= c * (
            _asinh_pair(z1, z2, np.sqrt(u2**2 + s**2))
            - _asinh_pair(z1, z2, np.sqrt(u1**2 + s**2))
        )

    # -s arctan(u Z / (s D)), corner by corner
    if s != 0:
        s_sq = s**2
        result -= s * (
            np.arctan(u2 * z1 / (s * np.sqrt(u2**2 + s_sq + z1**2)))
            - np.arctan(u2 * z2 / (s * np.sqrt(u2**2 + s_sq + z2**2)))
            - np.arctan(u1 * z1 / (s * np.sqrt(u1**2 + s_sq + z1**2)))
            + np.arctan(u1 * z2 / (s * np.sqrt(u1**2 + s_sq + z2**2)))
        )
    return result


def _psi_breakpoints(
    bound1: float, bound2: float, widths: tuple[float, ...]
) -> list[float]:
    """
    Breakpoints of the psi quadratures: multiples of pi and, when the point is
    close to a face plane, graded offsets around multiples of 2 pi.

    Parameters
    ----------
    bound1:
        The first integration bound [rad]
    bound2:
        The second integration bound [rad]
    widths:
        The angular widths of the logarithmic peaks at psi = 0 [rad]

    Returns
    -------
    :
        The breakpoints [rad]

    Notes
    -----
    The |sin psi| kinks of the kernels lie at multiples of pi. Close to a face
    plane, the integrand has a logarithmic peak of width ~ |z_k| / r_pc (z faces) or
    ~ |r_j - r_pc| / r_pc (r faces) around psi = 0 (mod 2 pi), which adaptive
    quadrature can under-resolve with a confident error estimate.
    """
    lower, upper = min(bound1, bound2), max(bound1, bound2)
    n = np.arange(np.ceil(lower / np.pi), np.floor(upper / np.pi) + 1)
    points = list(n * np.pi)
    for width in widths:
        if not 0 < width < MAX_PEAK_WIDTH:
            continue
        offsets = width * 10.0 ** np.arange(4)
        for k in np.arange(np.ceil(lower / TWO_PI), np.floor(upper / TWO_PI) + 1):
            points.extend(k * TWO_PI + offsets)
            points.extend(k * TWO_PI - offsets)
    return points


def _rotate_to_cartesian(vr: float, vt: float, theta: float) -> tuple[float, float]:
    """
    Rotate (radial, tangential) vector components at angle theta to (x, y).

    Returns
    -------
    :
        The x and y components
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return vr * cos_t - vt * sin_t, vr * sin_t + vt * cos_t


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
        self._area = 4 * breadth * depth
        self._rho = current / self._area
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

    def _quadrature_setup(
        self, rp: float, zp: float
    ) -> tuple[tuple[float, ...], float, float, float]:
        """
        Arguments, peak widths and absolute tolerance of the psi quadratures.

        Returns
        -------
        :
            Integrand arguments, z-face peak width [rad], r-face peak width [rad],
            absolute tolerance
        """
        r1, r2, z1, z2 = self._cylindrical_to_working(zp)
        if rp > 0:
            z_width = min(abs(z1), abs(z2)) / rp
            r_width = min(abs(r1 - rp), abs(r2 - rp)) / rp
        else:
            z_width = r_width = 0.0
        epsabs = QUAD_EPSABS_PER_AREA * self._area
        return (rp, r1, r2, z1, z2), z_width, r_width, epsabs

    def _psi_quadrature(
        self, rp: float, tp: float, zp: float
    ) -> tuple[Callable[[Callable, tuple[float, ...]], float], float, float]:
        """
        Build the psi quadrature over the arc for a point in local cylindrical
        coordinates.

        Returns
        -------
        :
            A function integrating an integrand with breakpoints graded for the
            given peak widths, the z-face peak width [rad] and the r-face peak
            width [rad]
        """
        args, z_width, r_width, epsabs = self._quadrature_setup(rp, zp)
        lower, upper = -tp, self._dtheta - tp

        def quad(integrand: Callable, widths: tuple[float, ...]) -> float:
            return integrate(
                integrand,
                args,
                lower,
                upper,
                breakpoints=_psi_breakpoints(lower, upper, widths),
                epsabs=epsabs,
                limit=QUAD_LIMIT,
            )

        return quad, z_width, r_width

    @property
    def _full_circle(self) -> bool:
        """Whether the arc is a full circle, so that the Bt and Ar terms vanish."""
        return np.isclose(abs(self._dtheta), TWO_PI, rtol=0.0, atol=EPS)

    def _BxByBz(self, rp: float, tp: float, zp: float) -> npt.NDArray[np.float64]:
        """
        Calculate the field at a point in local coordinates.

        Returns
        -------
        :
            (Bx, By, Bz) at a point
        """
        quad, z_width, r_width = self._psi_quadrature(rp, tp, zp)
        br = quad(br_integrand, (z_width,))
        bz = quad(bz_integrand, (z_width, r_width))
        bt = 0.0 if self._full_circle else quad(bt_integrand, (z_width,))
        return np.array([*_rotate_to_cartesian(br, bt, tp), bz])

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
        b_local = MU_0_4PI * self._rho * self._BxByBz(rp, tp, zp)
        # Convert field back to global coordinates
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
        quad, z_width, r_width = self._psi_quadrature(rp, tp, zp)
        widths = (z_width, r_width)
        at = quad(at_integrand, widths)
        ar = 0.0 if self._full_circle else quad(ar_integrand, widths)
        return np.array([*_rotate_to_cartesian(ar, at, tp), 0.0])

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

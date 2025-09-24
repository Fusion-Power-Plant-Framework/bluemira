# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of simple equilibrium physics calculations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RectBivariateSpline

from bluemira.base.constants import MU_0
from bluemira.base.parameter_frame._frame import ParameterFrame
from bluemira.base.parameter_frame._parameter import Parameter
from bluemira.equilibria.constants import PSI_NORM
from bluemira.equilibria.find import in_plasma
from bluemira.equilibria.grid import revolved_volume, volume_integral

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bluemira.equilibria.equilibrium import Equilibrium
    from bluemira.equilibria.find import Opoint, Xpoint
    from bluemira.equilibria.flux_surfaces import ClosedFluxSurface
    from bluemira.geometry.coordinates import Coordinates


def calc_psi_norm(
    psi: npt.ArrayLike, opsi: float, xpsi: float
) -> float | npt.NDArray[np.float64]:
    """
    Calculate normalised magnetic flux.

    \t:math:`\\dfrac{\\psi_{O}-\\psi}{\\psi_{O}-\\psi_{X}}`

    Parameters
    ----------
    psi:
        The magnetic flux per radian
    opsi:
        The psi value at the O-point
    xpsi:
        The psi value at the X-point

    Returns
    -------
    The normalised magnetic flux value(s)
    """
    return (opsi - psi) / (opsi - xpsi)


def calc_psi(
    psi_norm: npt.ArrayLike, opsi: float, xpsi: float
) -> float | npt.NDArray[np.float64]:
    """
    Calculate the absolute psi values from normalised psi values

    \t:math:`\\psi_{O}-\\psi_{norm}(\\psi_{O}-\\psi_{X})`

    Parameters
    ----------
    psi_norm:
        The normalised psi values
    opsi:
        The psi value at the O-point
    xpsi:
        The psi value at the X-point

    Returns
    -------
    psi:
        The magnetic flux per radian
    """
    return opsi - psi_norm * (opsi - xpsi)


def calc_tau_flattop(psi_sof: float, psi_eof: float, v_burn: float) -> float:
    """
    Calculates the flat-top length

    \t:math:`\\tau_{flat-top}=\\dfrac{\\psi_{SOF}-\\psi_{EOF}}{V_{burn}}`

    Parameters
    ----------
    psi_sof:
        The start of flat-top magnetic flux at the plasma boundary [V.s]
    psi_eof:
        The end of flat-top magnetic flux at the plasma boundary [V.s]
    v_burn:
        The plasma loop voltage during burn [V]

    Returns
    -------
    The duration of the flat-top [s]
    """
    return (psi_sof - psi_eof) / v_burn


def calc_psib(
    psi_bd: float, R_0: float, I_p: float, li: float, c_ejima: float = 0.4
) -> float:
    """
    Calculates the boundary flux at start of flat-top, after the breakdown

    \t:math:`\\psi_b=\\psi(t_{BD})-L_i I_p-\\Delta\\psi_{res}`

    with:
    \t:math:`L_i=\\dfrac{\\mu_0R_0l_i}{2}`

    \t:math:`\\Delta\\psi_{res}=C_{Ejima}\\mu_0R_0I_p`

    Parameters
    ----------
    psi_bd:
        The flux at the breakdown [V.s]
    R_0:
        The machine major radius [m]
    I_p:
        The desired flat-top plasma current [A]
    li:
        The normalised plasma inductance

    Returns
    -------
    The flux at the boundary at start of flat-top [V.s]
    """
    return psi_bd - 0.5 * MU_0 * R_0 * li * I_p - c_ejima * MU_0 * R_0 * I_p


def calc_k0(psi_xx0: float, psi_zz0: float) -> float:
    """
    Calculates the plasma elongation on the plasma axis (rho = 0).

    Parameters
    ----------
    psi_xx0:
        Second derivative of psi in X at the plasma axis (R_0, Z_0)
    psi_zz0:
        Second derivative of psi in Z at the plasma axis (R_0, Z_0)

    Returns
    -------
    Plasma elongation at the plasma axis
    """
    return np.sqrt(psi_xx0 / psi_zz0)


def calc_q0(eq: Equilibrium) -> float:
    """
    Calculates the plasma MHD safety factor on the plasma axis (rho=0).
    Freidberg, Ideal MHD, eq 6.42, p 134

    Parameters
    ----------
    eq:
        Equilibrium for which to calculate the safety factor on axis

    Returns
    -------
    The MHD safety factor on the plasma axis
    """
    opoint = eq.get_OX_points()[0][0]
    psi_xx0 = eq.psi_func(opoint.x, opoint.z, dx=2, grid=False)
    psi_zz0 = eq.psi_func(opoint.x, opoint.z, dy=2, grid=False)
    b_0 = eq.Bt(opoint.x)
    jfunc = RectBivariateSpline(eq.x[:, 0], eq.z[0, :], eq._jtor)
    j_0 = jfunc(opoint.x, opoint.z, grid=False)
    k_0 = calc_k0(psi_xx0, psi_zz0)
    return (b_0 / (MU_0 * opoint.x * j_0)) * (1 + k_0**2) / k_0


def calc_dx_sep(eq: Equilibrium) -> float:
    """
    Calculate the magnitude of the minimum separation between the flux
    surfaces of null points in the equilibrium at the outboard midplane.

    Parameters
    ----------
    eq:
        Equilibrium for which to calculate dx_sep

    Returns
    -------
    Separation distance at the outboard midplane between the active
    null and the next closest flux surface with a null [m]
    """
    o_points, x_points = eq.get_OX_points()
    x, z = eq.get_LCFS().xz
    lfs = np.argmax(x)
    lfp = eq.get_midplane(x[lfs], z[lfs], x_points[0].psi)
    d_x = []
    count = 0  # Necessary because of retrieval of eqdsks with limiters
    for xp in x_points:
        if "Xpoint" in xp.__class__.__name__:
            if count > 0:
                psinorm = calc_psi_norm(xp.psi, o_points[0].psi, x_points[0].psi)
                if psinorm > 1:
                    d_x.append(eq.get_midplane(*lfp, xp.psi)[0])
            count += 1
    return np.min(d_x) - lfp[0]


def calc_volume(fs: Coordinates) -> float:
    """
    Calculates plasma volume [m^3]

    Parameters
    ----------
    fs:
        Closed flux surface

    Returns
    -------
        Plasma volume within closed flux surface

    """
    return revolved_volume(*fs.xz)


def _calc_Bp2_int(
    Bp: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64] | None,
    x: npt.NDArray[np.float64],
    dx: npt.NDArray[np.float64],
    dz: npt.NDArray[np.float64],
) -> float:
    """
    Calculates the volume integral of the poloidal field squared.

    Parameters
    ----------
    Bp:
        Poloidal field at x and z-coordinates
    mask:
        Mask for chosen closed flux surface
    x:
        X-coordinates
    dx:
        Discretisation size in the x-coordinate
    dz:
        Discretisation size in the z-coordinate

    Returns
    -------
        volume integral (masked if chosen)
    """
    Bp2 = Bp**2 if mask is None else Bp**2 * mask
    return volume_integral(Bp2, x, dx, dz)


def _calc_p_int(
    p: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    dx: npt.NDArray[np.float64],
    dz: npt.NDArray[np.float64],
) -> float:
    """
    Calculates the volume integral of plasma pressure.

    Parameters
    ----------
    p:
        Plasma pressure
    x:
        X-coordinates
    dx:
        Discretisation size in the x-coordinate
    dz:
        Discretisation size in the z-coordinate

    Returns
    -------
        volume integral
    """
    return volume_integral(p, x, dx, dz)


def calc_energy(eq: Equilibrium) -> float:
    """
    Calculates the stored poloidal magnetic energy in the plasma [J]

    \t:math:`W=\\dfrac{LI^2}{2}`

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate stored energy

    Returns
    -------
        Stored poloidal magnetic energy
    """
    return _calc_Bp2_int(
        Bp=eq.Bp(),
        mask=in_plasma(eq.x, eq.z, eq.psi()),
        x=eq.x,
        dx=eq.dx,
        dz=eq.dz,
    ) / (2 * MU_0)


def _calc_Li_from_energy(p_energy: float, i_p: float) -> float:
    """
    Calculates the internal inductance of the plasma [H]

    Parameters
    ----------
    p_energy:
        Poloidal magnetic energy
    i_p:
        Plasma current

    Returns
    -------
        Internal inductance of the plasma
    """
    return 2 * p_energy / i_p**2


def calc_Li(eq: Equilibrium) -> float:
    """
    Calculates the internal inductance of the plasma [H]

    \t:math:`L_i=\\dfrac{2W}{I_{p}^{2}}`

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate internal inductance

    Returns
    -------
        Internal inductance of the plasma
    """
    p_energy = calc_energy(eq)
    return _calc_Li_from_energy(p_energy, eq.profiles.I_p)


def _calc_li_from_Li(big_li: float, R_0: float) -> float:
    """
    Calculates the normalised internal inductance of the plasma

    \t:math:`l_i=\\dfrac{2L_i}{\\mu_{0}R_{0}}`

    Parameters
    ----------
    big_li:
        Internal inductance of plasma
    R_0:
        Major Radius

    Returns
    -------
        Normalised internal inductance of the plasma
    """
    return 2 * big_li / (MU_0 * R_0)


def calc_li(eq: Equilibrium) -> float:
    """
    Calculates the normalised internal inductance of the plasma

    \t:math:`l_i=\\dfrac{2L_i}{\\mu_{0}R_{0}}`

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate internal inductance

    Returns
    -------
        Normalised internal inductance of the plasma
    """
    li = calc_Li(eq)
    return _calc_li_from_Li(li, eq._R_0)


def calc_li3(eq: Equilibrium) -> float:
    """
    Calculates the normalised internal plasma inductance (ITER approximate
    calculation)

    see :doi:`10.1088/0029-5515/48/12/125002`

    \t:math:`li(3)=\\dfrac{2V\\langle B_p^2\\rangle}{(\\mu_0I_p)^2R_0}`

    with:
    \t:math:`\\langle B_p^2\\rangle=\\dfrac{1}{V}\\int B_p^2dV`

    where: Bp is the poloidal magnetic field and V is the plasma volume

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate internal inductance

    Returns
    -------
        Approximate normalised internal inductance of the plasma
    """
    return _calc_li3minargs(
        eq.x, eq.z, eq.psi(), eq.Bp(), eq.profiles.R_0, eq.profiles.I_p, eq.dx, eq.dz
    )


def _calc_li3minargs(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    Bp: npt.NDArray[np.float64],
    R_0: float,
    I_p: float,
    dx: float,
    dz: float,
    mask: npt.NDArray[np.float64] | None = None,
    o_points: Iterable[Opoint] | None = None,
    x_points: Iterable[Xpoint] | None = None,
) -> float:
    """
    Calculate the normalised plasma internal inductance with arguments only.

    \t:math:`\\dfrac{2 B_{p, average}}{R_{0} (\\mu_{0} I_{p})**2}`

    Used in the optimisation of the plasma profiles.

    Parameters
    ----------
    x:
        X-coordinates
    z:
        Z-coordinates
    psi:
        The poloidal magnetic flux map [V.s/rad]
    Bp:
        Poloidal field at x and z-coordinates
    R_0:
        Major radius
    I_p:
        Plasma current
    dx:
        Discretisation size in the x-coordinate
    dz:
        Discretisation size in the z-coordinate
    mask:
        Mask for chosen closed flux surface
    o_points:
        O-points to use to calculate psinorm
    x_points:
        X-points to use to calculate psinorm

    Returns
    -------
        Approximate normalised internal inductance of the plasma
    """
    if mask is None:
        mask = in_plasma(x, z, psi, o_points=o_points, x_points=x_points)
    return 2 * _calc_Bp2_int(Bp, mask, x, dx, dz) / (R_0 * (MU_0 * I_p) ** 2)


def calc_p_average(eq: Equilibrium) -> float:
    """
    Calculate the average plasma pressure.

    \t:math:`\\langle p \\rangle = \\dfrac{1}{V_{p}}\\int \\mathbf{p}dxdz`:

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate p_average

    Returns
    -------
    The average plasma pressure [Pa]
    """
    return _calc_p_average(
        eq.pressure_map(PSI_NORM),
        eq.get_LCFS(),
        eq.x,
        eq.dz,
        eq.dz,
    )


def _calc_p_average(
    pressure_map: npt.NDArray,
    fs: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    dx: float,
    dz: float,
) -> float:
    """
    Calculate the average plasma pressure.

    \t:math:`\\langle p \\rangle = \\dfrac{1}{V_{p}}\\int \\mathbf{p}dxdz`:

    Parameters
    ----------
    pressure_map:
        Pressure at normalised psi values within chosen closed flux surface
    fs:
        Coordinates of the chosen closed flux surface
    x:
        X-coordinates
    dx:
        Discretisation size in the x-coordinate
    dz:
        Discretisation size in the z-coordinate

    Returns
    -------
    The average plasma pressure [Pa]
    """
    v_plasma = calc_volume(fs)
    return _calc_p_int(pressure_map, x, dx, dz) / v_plasma


def calc_beta_t(eq: Equilibrium) -> float:
    """
    Calculate the ratio of plasma pressure to toroidal magnetic pressure.

    \t:math:`\\beta_t = \\dfrac{2\\mu_0\\langle p \\rangle}{B_t^2}`

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate beta_t

    Returns
    -------
    Ratio of plasma to toroidal magnetic pressure
    """
    return _calc_beta_t(
        eq.pressure_map(PSI_NORM), eq.get_LCFS(), eq.x, eq.dx, eq.dz, eq.profiles._B_0
    )


def _calc_beta_t(
    pressure_map: npt.NDArray[np.float64],
    fs: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    dx: float,
    dz: float,
    B_0: float,
) -> float:
    """
    Calculate the ratio of plasma pressure to toroidal magnetic pressure.

    \t:math:`\\beta_t = \\dfrac{2\\mu_0\\langle p \\rangle}{B_t^2}`

    Parameters
    ----------
    pressure_map:
        Pressure at normalised psi values within chosen closed flux surface
    fs:
        Coordinates of the chosen closed flux surface
    x:
        X-coordinates
    dx:
        Discretisation size in the x-coordinate
    dz:
        Discretisation size in the z-coordinate
    B_0:
        Toroidal field at reactor major radius [T]

    Returns
    -------
    Ratio of plasma to toroidal magnetic pressure
    """
    p_avg = _calc_p_average(pressure_map, fs, x, dx, dz)
    return 2 * MU_0 * p_avg / B_0**2


def calc_beta_p(eq: Equilibrium) -> float:
    """
    Calculate the ratio of plasma pressure to poloidal magnetic pressure

    \t:math:`\\beta_p = \\dfrac{2\\mu_0\\langle p \\rangle}{B_p^2}`

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate beta_p

    Returns
    -------
    Ratio of plasma to magnetic pressure
    """
    return _calc_beta_p(
        eq.pressure_map(PSI_NORM),
        eq.Bp(),
        eq._get_core_mask(PSI_NORM),
        eq.dx,
        eq.dz,
    )


def calc_beta_p_approximate(eq: Equilibrium) -> float:
    """
    Calculate the ratio of plasma pressure to magnetic pressure. This is
    following the definitions of Friedberg, Ideal MHD, pp. 68-69, which is an
    approximation.

    \t:math:`\\beta_p = \\dfrac{2\\mu_0\\langle p \\rangle}{B_p^2}`

    Note
    ----
    Be careful, this approximation is not good for high elongation plasmas,
    try comparing to calc_beta_p before using.

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate beta_p

    Returns
    -------
    Ratio of plasma to poloidal magnetic pressure
    """
    return _calc_beta_p_approx(
        eq.pressure_map(PSI_NORM), eq.get_LCFS(), eq.dx, eq.dz, eq.profiles.I_p
    )


def _calc_beta_p(
    pressure_map: npt.NDArray[np.float64],
    Bp: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    dx: float,
    dz: float,
) -> float:
    """
    Calculate the ratio of plasma pressure to poloidal magnetic pressure

    \t:math:`\\beta_p = \\dfrac{2\\mu_0\\langle p \\rangle}{B_p^2}`

    Parameters
    ----------
    pressure_map:
        Pressure at normalised psi values within chosen closed flux surface
    fs:
        Coordinates of the chosen closed flux surface
    Bp:
        Poloidal field at x and z-coordinates
    mask:
        Mask for chosen closed flux surface
    x:
        X-coordinates
    dx:
        Discretisation size in the x-coordinate
    dz:
        Discretisation size in the z-coordinate

    Returns
    -------
    Ratio of plasma to magnetic pressure
    """
    p_int = _calc_p_int(pressure_map, x, dx, dz)
    Bp2_int = _calc_Bp2_int(Bp, mask, x, dx, dz)
    return 2 * MU_0 * p_int / Bp2_int


def _calc_beta_p_approx(
    pressure_map: npt.NDArray[np.float64],
    fs: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    dx: float,
    dz: float,
    I_p: float,
) -> float:
    """
    Calculate the ratio of plasma pressure to magnetic pressure. This is
    following the definitions of Friedberg, Ideal MHD, pp. 68-69, which is an
    approximation.

    \t:math:`\\beta_p = \\dfrac{2\\mu_0\\langle p \\rangle}{B_p^2}`

    Note
    ----
    Be careful, this approximation is not good for high elongation plasmas,
    try comparing to calc_beta_p before using.


    Parameters
    ----------
    pressure_map:
        Pressure at normalised psi values within chosen closed flux surface
    fs:
        Coordinates of the chosen closed flux surface
    x:
        X-coordinates
    dx:
        Discretisation size in the x-coordinate
    dz:
        Discretisation size in the z-coordinate
    I_p:
        Plasma current

    Returns
    -------
    Ratio of plasma to poloidal magnetic pressure
    """
    p_avg = calc_p_average(pressure_map, fs, x, dx, dz)
    Bp = MU_0 * I_p / fs.length
    return 2 * MU_0 * p_avg / Bp**2


@dataclass
class EqSummary(ParameterFrame):
    """
    Calculates interesting values in one go.
    """

    W: Parameter[float]
    Li: Parameter[float]
    li: Parameter[float]
    li_3: Parameter[float]
    V: Parameter[float]
    beta_p: Parameter[float]
    q_95: Parameter[float]
    kappa_95: Parameter[float]
    delta_95: Parameter[float]
    zeta_95: Parameter[float]
    kappa: Parameter[float]
    delta: Parameter[float]
    zeta: Parameter[float]
    R_0: Parameter[float]
    A: Parameter[float]
    a: Parameter[float]
    # dXsep : Parameter[float]
    I_p: Parameter[float]
    dx_shaf: Parameter[float]
    dz_shaf: Parameter[float]

    @classmethod
    def from_equilibrium(
        cls,
        eq: Equilibrium,
        f95: ClosedFluxSurface,
        f100: ClosedFluxSurface,
        *,
        is_double_null: bool,
    ):
        """
        Create summary from equilibrium
        """  # noqa: DOC201
        R_0, I_p = eq.profiles.R_0, eq.profiles.I_p
        energy = calc_energy(eq)
        li_true = _calc_Li_from_energy(energy, I_p)

        if is_double_null:
            kappa_95 = f95.kappa
            delta_95 = f95.delta
            zeta_95 = f95.zeta
            kappa = f100.kappa
            delta = f100.delta
            zeta = f100.zeta

        else:
            kappa_95 = f95.kappa_upper
            delta_95 = f95.delta_upper
            zeta_95 = f95.delta_upper
            kappa = f100.kappa_upper
            delta = f100.delta_upper
            zeta = f100.delta_upper

        # d['dXsep'] = self.calc_dXsep()
        dx_shaf, dz_shaf = f100.shafranov_shift(eq)
        eq_name = eq.label
        return cls(
            W=Parameter("W", energy, "J", eq_name),
            Li=Parameter("Li", li_true, "", eq_name, long_name="internal_inductance"),
            li=Parameter(
                "li", _calc_li_from_Li(li_true, R_0), "", eq_name, long_name=""
            ),
            li_3=Parameter(
                "li_3",
                calc_li3(eq),
                "",
                eq_name,
                description="normalised internal plasma inductance",
            ),
            V=Parameter(
                "V", calc_volume(f100.coords), "m^3", eq_name, long_name="plasma_volume"
            ),
            beta_p=Parameter(
                "beta_p",
                calc_beta_p(eq),
                "",
                eq_name,
            ),
            q_95=Parameter("q_95", f95.safety_factor(eq), "", eq_name),
            R_0=Parameter("R_0", f100.major_radius, "m", eq_name),
            A=Parameter("A", f100.aspect_ratio, "", eq_name, long_name="aspect_ratio"),
            a=Parameter("a", f100.area, "m^2", eq_name, long_name="plasma_area"),
            I_p=Parameter("I_p", I_p, "A", eq_name, long_name="plasma_current"),
            dx_shaf=Parameter(
                "dx_shaf", dx_shaf, "m", eq_name, long_name="x_shafranov_shift"
            ),
            dz_shaf=Parameter(
                "dz_shaf", dz_shaf, "m", eq_name, long_name="z_shafranov_shift"
            ),
            kappa_95=Parameter("kappa_95", kappa_95, "", eq_name),
            delta_95=Parameter("delta_95", delta_95, "", eq_name),
            zeta_95=Parameter("zeta_95", zeta_95, "", eq_name),
            kappa=Parameter("kappa", kappa, "", eq_name),
            delta=Parameter("delta", delta, "", eq_name),
            zeta=Parameter("zeta", zeta, "", eq_name),
        )


def beta(pressure: npt.NDArray[np.float64], field: float) -> float:
    """
    The ratio of plasma pressure to magnetic pressure

    \t:math:`\\beta = \\dfrac{\\langle p \\rangle}{B^2/2\\mu_0}`

    Parameters
    ----------
    pressure:
        Plasma pressure, from which the mean is to be calculated [Pa]
    field:
        Mean total field strength [T]

    Returns
    -------
    Ratio of plasma to magnetic pressure
    """
    return np.mean(pressure) / (field**2 / 2 * MU_0)


def normalise_beta(beta: float, a: float, b_tor: float, I_p: float) -> float:
    """
    Converts beta to normalised beta

    \t:math:`\\beta_{N} = \\beta\\dfrac{aB_{T}}{I_{p}}`

    Parameters
    ----------
    beta:
        Ratio of plasma to magnetic pressure
    a:
        Plasma minor radius [m]
    b_tor:
        Toroidal field [T]
    I_p:
        Plasma current [A]

    Returns
    -------
    Normalised ratio of plasma to magnetic pressure (Troyon factor)
    """
    return beta * a * b_tor / I_p


def beta_N_to_beta(  # noqa: N802
    beta_N: float,  # noqa: N803
    a: float,
    Btor: float,
    I_p: float,
) -> float:
    """
    Converts normalised beta to beta

    \t:math:`\\beta = \\beta_{N}\\dfrac{I_{p}}{aB_{T}}`

    Parameters
    ----------
    beta_N:
        Normalised ratio of plasma to magnetic pressure (Troyon factor)
    a:
        Plasma minor radius [m]
    b_tor:
        Toroidal field [T]
    I_p:
        Plasma current [A]

    Returns
    -------
    Ratio of plasma to magnetic pressure
    """
    return beta_N * I_p / (a * Btor)


def calc_infinite_solenoid_flux(r_cs_min: float, r_cs_max: float, B_max: float) -> float:
    """
    Calculate the maximum flux achievable from an infinite solenoid given a peak field.

    .. math::
        B_{max} \\dfrac{\\pi}{3} (r_{cs, max}**2 r_{cs, min}**2
        + r_{cs, max} r_{cs, min})

    Parameters
    ----------
    r_cs_min:
        Inner radius of the infinite solenoid [m]
    r_cs_max:
        Outer radius of the infinite solenoid [m]
    B_max:
        Peak allowable field in the solenoid [T]

    Returns
    -------
    Maximum achievable flux from an infinite solenoid [V.s]
    """
    return B_max * np.pi / 3 * (r_cs_max**2 + r_cs_min**2 + r_cs_max * r_cs_min)

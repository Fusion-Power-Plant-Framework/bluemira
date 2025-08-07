# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of functions used to approximate toroidal harmonics.
"""

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma, poch

from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.equilibria.coils._grouping import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.find import _in_plasma, find_flux_surf
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.geometry.coordinates import Coordinates
from bluemira.utilities.tools import (
    cylindrical_to_toroidal,
    sig_fig_round,
    toroidal_to_cylindrical,
)


def f_hypergeometric(a, b, c, z, n_max=20):
    """Evaluates the hypergeometric power series up to n_max.
    Valid for \\|z\\| < 1

    .. math::
        F(a, b; c; z) = \\sum_0^{n_max} \\frac{(a)_{s} (b)_{s}}{Gamma(c + s) s!} z^{s}

    See https://dlmf.nist.gov/15.2#E2 and https://dlmf.nist.gov/5.2#iii for more
    information.

    Parameters
    ----------
    a:
        hypergeometric function term, as defined in the equation above.
    b:
        hypergeometric function term, as defined in the equation above.
    c:
        hypergeometric function term, as defined in the equation above.
    z:
        hypergeometric function term, as defined in the equation above.
        Require \\|z\\|<1.
    n_max:
        upper limit of summation, default=20

    Returns
    -------
    F:
        hypergeometric function result.
    """
    F = 0
    for s in range(n_max + 1):
        gamma_value = np.asarray(gamma(c + s) * factorial(s), dtype=float)
        F += (poch(a, s) * poch(b, s)) / gamma_value * z**s
    return F


def legendre_p(lam, mu, x, n_max=20):
    """Evaluates the associated Legendre function of the first kind of degree lambda and order
    minus mu as a function of x. See https://dlmf.nist.gov/14.3#E18 for more information.
    Works for half integer order.

    Valid for 1<x<infinity, and real \\mu and \\nu

    .. math::
        P_{\\lambda}^{-\\mu}(x) = 2^{-\\mu} x^{\\lambda - \\mu} (x^2 - 1)^{\\mu/2}
                        F(\\frac{1}{2}(\\mu - \\lambda), \\frac{1}{2}(\\mu - \\lambda + 1);
                            \\mu + 1; 1 - \\frac{1}{x^2})

        where F is the hypergeometric function defined above as f_hypergeometric.

    Parameters
    ----------
    lam:
        degree of the associated Legendre function of the first kind.
    mu:
        order -mu of the associated Legendre function of the first kind.
    x:
        points at which to evaluate legendreP.
    n_max:
        upper value for summation in f_hypergeometric.

    Returns
    -------
    legP:
        value of legendreP.
    """  # noqa: W505, E501
    a = 1 / 2 * (mu - lam)
    b = 1 / 2 * (mu - lam + 1)
    c = mu + 1
    z = 1 - 1 / (x**2)
    F_sum = f_hypergeometric(a=a, b=b, c=c, z=z, n_max=n_max)  # noqa: N806
    legP = 2 ** (-mu) * x ** (lam - mu) * (x**2 - 1) ** (mu / 2) * F_sum  # noqa: N806
    return legP  # noqa: RET504


def legendre_q(lam, mu, x, n_max=20):
    """Evaluates Olver's definition of the associated Legendre function of the second
    kind of degree lambda and order minus mu as a function of x. See
    https://dlmf.nist.gov/14, https://dlmf.nist.gov/14.3#E10, and
    https://dlmf.nist.gov/14.3#E7 for more information.
    Works for half integer order.

    Valid for 1<x<infinity, and real \\mu and \\nu

    .. math::
        \\textbf{Q}_{\\lambda}^{\\mu}(x) = \\frac{\\pi^{\\frac{1}{2}} (x^2 - 1)^
                            {\\frac{\\mu}{2}}}{2^{\\lambda + 1} x^{\\lambda + \\mu + 1}}
                                F(\\frac{1}{2}(\\lambda + \\mu)+1, \\frac{1}{2}(\\lambda
                                + \\mu); \\lambda + \\frac{3}{2}; \\frac{1}{x^2})

        where F is the hypergeometric function defined above as f_hypergeometric.

    Parameters
    ----------
    lam:
        degree of the associated Legendre function of the second kind.
    mu:
        order mu of the associated Legendre function of the second kind.
    x:
        points at which to evaluate legendreQ.
    n_max:
        upper value for summation in f_hypergeometric.

    Returns
    -------
    legQ:
        value of legendreQ.
    """
    a = 1 / 2 * (lam + mu) + 1
    b = 1 / 2 * (lam + mu + 1)
    c = lam + 3 / 2
    z = 1 / (x**2)
    F_sum = f_hypergeometric(a=a, b=b, c=c, z=z, n_max=n_max)  # noqa: N806
    legQ = (  # noqa: N806
        (np.pi ** (1 / 2) * (x**2 - 1) ** (mu / 2))
        / (2 ** (lam + 1) * x ** (lam + mu + 1))
        * F_sum
    )

    if isinstance(legQ, np.float64):
        if x == 1:
            legQ = np.inf  # noqa: N806
    elif len(np.shape(legQ)) > 2:  # noqa: PLR2004
        legQ[:, x == 1] = np.inf
    else:
        legQ[x == 1] = np.inf
    return legQ


@dataclass
class ToroidalHarmonicsParams:
    """
    A Dataclass holding necessary parameters for the toroidal harmonics approximation.
    """

    R_0: float
    """R coordinate of the focus point in cylindrical coordinates"""
    Z_0: float
    """Z coordinate of the focus point in cylindrical coordinates"""
    R: np.ndarray
    """R coordinates of the grid in cylindrical coordinates"""
    Z: np.ndarray
    """Z coordinates of the grid in cylindrical coordinates"""
    R_coils: np.ndarray
    """R coordinates of the coils in cylindrical coordinates"""
    Z_coils: np.ndarray
    """Z coordinates of the coils in cylindrical coordinates"""
    tau: np.ndarray
    """tau coordinates of the grid in toroidal coordinates"""
    sigma: np.ndarray
    """sigma coordinates of the grid in toroidal coordinates"""
    tau_c: np.ndarray
    """tau coordinates of the coils in toroidal coordinates"""
    sigma_c: np.ndarray
    """sigma coordinates of the coils in toroidal coordinates"""
    th_coil_names: list
    """names of coils to use with TH approximation (always outside the LCFS tau limit)"""


def toroidal_harmonic_grid_and_coil_setup(
    eq: Equilibrium, R_0: float, Z_0: float, radius: float | None = None
) -> ToroidalHarmonicsParams:
    """
    Set up the grid and coils to be used in toroidal harmonic approximation.

    Use the LCFS to find the region over which to approximate psi using TH.
    Find the coils located outside this region, which can be used in the TH
    approximation, and find the coils located inside this region.

    Parameters
    ----------
    eq:
        Starting equilibrium to use in our approximation
    R_0:
        R coordinate of the focus point in cylindrical coordinates
    Z_0:
        Z coordinate of the focus point in cylindrical coordinates

    Returns
    -------
    ToroidalHarmonicsParams:
        Dataclass holding necessary parameters for the TH approximation
    """
    # Find region over which to approximate psi using TH by finding LCFS tau limit
    if radius is None:
        lcfs = eq.get_LCFS()
        lcfs_tau, _ = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=lcfs.x, Z=lcfs.z)
        tau_lcfs_limit = np.min(lcfs_tau)
    else:
        x_points = np.array([R_0 - radius, R_0, R_0 + radius, R_0])
        z_points = np.array([0, radius, 0, -radius])
        circle_tau, _ = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=x_points, Z=z_points)
        tau_lcfs_limit = np.min(circle_tau)

    # Using approximate value for d2_min and tau_max to avoid infinities and divide by 0
    # errors
    # From the toroidal coordinate transform functions we have that
    # $\tau = \ln \frac{d_1}{d_2}$, $d_1^2 = (R + R_0)^2 + (z - z_0)^2$, and
    # $d_2^2 = (R - R_0)^2 + (z - z_0)^2$
    # We want to approximate the maximum value of tau and the minimum value of d_2.
    # The maximum value of tau occurs when d_1 is at its maximum value and d_2 is at its
    # minimum value. d_1 is largest at the focus, where R = R_0 and z = z_0, and so from
    # the relation above we have that d2_max = 2 * R_0. This gives us that
    # tau_max = ln(2 * R_0 / d2_min). d_2 is smallest at the focus, and this would give
    # d_2 = 0. However, this would lead to a divide by 0 error in calculating tau_max
    # and so we set d2_min to be a small number to avoid this error.

    d2_min = 0.05
    tau_max = np.log(2 * R_0 / d2_min)
    n_tau = 200
    tau = np.linspace(tau_lcfs_limit, tau_max, n_tau)
    n_sigma = 150
    sigma = np.linspace(-np.pi, np.pi, n_sigma)

    # Create grid in toroidal coordinates
    tau, sigma = np.meshgrid(tau, sigma)

    # Convert to cylindrical coordinates
    R, Z = toroidal_to_cylindrical(R_0=R_0, z_0=Z_0, tau=tau, sigma=sigma)  # noqa: N806
    R_coils = eq.coilset.x  # noqa: N806
    Z_coils = eq.coilset.z  # noqa: N806
    tau_c, sigma_c = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=R_coils, Z=Z_coils)

    c_names = np.array(eq.coilset.control)

    # Find coils that can be used in TH approximation, and those that cannot be used
    if tau_lcfs_limit < np.min(tau_c):
        not_too_close_coils = c_names[tau_c < tau_lcfs_limit].tolist()
        bluemira_debug(
            "Names of coils that can be used in the TH"
            f" approximation: {not_too_close_coils}."
        )
        th_coil_names = not_too_close_coils
    else:
        th_coil_names = c_names.tolist()

    eq.coilset.control = th_coil_names
    R_coils, Z_coils = eq.coilset.get_control_coils().x, eq.coilset.get_control_coils().z  # noqa: N806
    tau_c, sigma_c = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=R_coils, Z=Z_coils)

    return ToroidalHarmonicsParams(
        R_0,
        Z_0,
        R,
        Z,
        R_coils,
        Z_coils,
        tau,
        sigma,
        tau_c,
        sigma_c,
        th_coil_names,
    )


def coil_toroidal_harmonic_amplitude_matrix(
    input_coils: CoilSet,
    th_params: ToroidalHarmonicsParams,
    cos_degrees_chosen: np.ndarray | None = None,
    sin_degrees_chosen: np.ndarray | None = None,
    sig_figures: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct coefficient matrices from toroidal harmonic amplitudes at given coil
    locations, for the specified cos and sin degrees.

    To get the individual cos and sin arrays of toroidal harmonic amplitudes/coefficients
    (Am_cos, Am_sin) which can be used in a toroidal harmonic approximation of the
    vacuum/coil contribution to the poloidal flux (psi) do:

    A_m = matrix harmonic amplitudes @ vector of coil currents

    Am_cos and Am_sin can be used as constraints in optimisation, see
    ToroidalHarmonicsConstraint.

    N.B. for a single filament (coil):

    .. math::

        A_{m} = \\frac{\\mu_{0} I_{c}}{2^{5/2}} \\frac{(2m+1)!!}{2^m m!}
        \\frac{\\sinh{\\tau_{c}}}{\\Delta_{c}^{1/2}}
        P_{m-\\frac{1}{2}}^{-1}(\\cosh{\\tau_c})

        A_{m}^{\\sin} = A_m \\sin(m \\sigma_c)
        A_{m}^{\\cos} = A_m \\cos(m \\sigma_c)


    Where m = poloidal mode number, :math: P_{\\lambda}^{-\\mu} are the associated
    Legendre functions of the first kind of degree lambda and order minus mu, and :math:
    \\Delta_c = \\cosh{\\tau_c} - \\cos{\\sigma_c}.

    Note: the factorial term \\frac{(2m+1)!!}{2^m m!} is equivalent to 1 if m = 0,
    otherwise \\prod_{i=0}^{m-1} \\left( 1 + \\frac{1}{2(m-i)}\\right)

    Parameters
    ----------
    input_coils:
        Bluemira CoilSet
    th_params:
        Dataclass holding necessary parameters for the TH approximation
    cos_degrees_chosen:
        Degrees chosen to be used for the cos components
    sin_degrees_chosen:
        Degrees chosen to be used for the sin components
    sig_figures:
        Number of significant figures for rounding currents2harmonics values

    Returns
    -------
    Am_cos:
        Cos component of matrix of harmonic amplitudes,
    Am_sin:
        Sin component of matrix of harmonic amplitudes

    """
    R_0 = th_params.R_0
    Z_0 = th_params.Z_0

    # Coils
    x_c = []
    z_c = []
    for n in th_params.th_coil_names:
        x_c.append(input_coils[n].x)
        z_c.append(input_coils[n].z)

    x_c = np.array(x_c)
    z_c = np.array(z_c)

    # Toroidal coords
    tau_c, sigma_c = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=x_c, Z=z_c)
    # Useful combination
    Deltac = np.cosh(tau_c) - np.cos(sigma_c)  # noqa: N806

    # [number of degrees, number of coils]
    currents2harmonics_cos = np.zeros([len(cos_degrees_chosen), np.size(tau_c)])
    currents2harmonics_sin = np.zeros([len(sin_degrees_chosen), np.size(tau_c)])

    # TH coefficients from function of the current distribution
    # outside of the region containing the core plasma
    # TH coefficients = currents2harmonics @ coil currents
    factorial_term_cos = np.array([
        np.prod(1 + 0.5 / np.arange(1, m + 1)) for m in cos_degrees_chosen
    ])
    factorial_term_sin = np.array([
        np.prod(1 + 0.5 / np.arange(1, m + 1)) for m in sin_degrees_chosen
    ])

    cos_empty = len(cos_degrees_chosen) == 0
    sin_empty = len(sin_degrees_chosen) == 0

    if cos_empty:
        # No cos degrees selected
        Am_cos = []  # noqa: N806
    else:
        currents2harmonics_cos[:, :] = (
            (MU_0 * 1.0 / 2.0 ** (5.0 / 2.0))
            * factorial_term_cos[:, None]
            * (np.sinh(tau_c)[None, :] / np.sqrt(Deltac)[None, :])
            * legendre_p(
                cos_degrees_chosen[:, None] - 1 / 2, 1, np.cosh(tau_c)[None, :], n_max=30
            )
        )
        sigma_c_mult_degree_cos = [m * th_params.sigma_c for m in cos_degrees_chosen]
        Am_cos = currents2harmonics_cos * np.cos(sigma_c_mult_degree_cos)  # noqa: N806
        Am_cos = sig_fig_round(Am_cos, sig_figures)  # noqa: N806

    if sin_empty:
        # No sin degrees selected
        Am_sin = []  # noqa: N806
    else:
        currents2harmonics_sin[:, :] = (
            (MU_0 * 1.0 / 2.0 ** (5.0 / 2.0))
            * factorial_term_sin[:, None]
            * (np.sinh(tau_c)[None, :] / np.sqrt(Deltac)[None, :])
            * legendre_p(
                sin_degrees_chosen[:, None] - 1 / 2, 1, np.cosh(tau_c)[None, :], n_max=30
            )
        )
        sigma_c_mult_degree_sin = [m * th_params.sigma_c for m in sin_degrees_chosen]
        Am_sin = currents2harmonics_sin * np.sin(sigma_c_mult_degree_sin)  # noqa: N806
        Am_sin = sig_fig_round(Am_sin, sig_figures)  # noqa: N806
    return Am_cos, Am_sin


def toroidal_harmonic_approximate_psi(
    eq: Equilibrium,
    th_params: ToroidalHarmonicsParams,
    cos_degrees_chosen: np.ndarray | None = None,
    sin_degrees_chosen: np.ndarray | None = None,
    # TODO @clmould: add different ways to set th grid size
    # e.g. limit_type: TH_GRID_LIMIT = TH_GRID_LIMIT.LCFS or TH_GRID_LIMIT.COILSET
    # 3870
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate psi using toroidal harmonic amplitudes for the specified cos and sin
    degrees as calculated in coil_toroidal_harmonic_amplitude_matrix.

    ..math::
        A_{m} = \\frac{\\mu_{0} I_{c}}{2^{5/2}} \\frac{(2m+1)!!}{2^m m!}
        \\frac{\\sinh{\\tau_{c}}}{\\Delta_{c}^{1/2}}
        P_{m-\\frac{1}{2}}^{-1}(\\cosh(\\tau_c))

        A_{m}^{\\sin} = A_m \\sin(m \\sigma_c)
        A_{m}^{\\cos} = A_m \\cos(m \\sigma_c)
        A(\\tau, \\sigma) = \\sum_{m=0}^{\\infty} A_{m}^{\\cos} \\epsilon_{m} m!
        \\sqrt{\\frac{2}{\\pi}}\\Delta^{\\frac{1}{2}} \\textbf{Q}_{m-\\frac{1}{2}}^{1}
        (\\cosh \\tau) \\cos(m \\sigma) + A_{m}^{\\sin}\\epsilon_{m} m! \\sqrt{\\frac{2}
        {\\pi}} \\Delta^{\\frac{1}{2}} \\textbf{Q}_{m-\\frac{1}{2}}^{1}(\\cosh \\tau)
        \\sin(m \\sigma)


    Parameters
    ----------
    eq:
        Bluemira Equilibrium
    th_params:
        Dataclass holding necessary parameters for the TH approximation
    cos_degrees_chosen:
        Degrees chosen to be used for the cos components
    sin_degrees_chosen:
        Degrees chosen to be used for the sin components

    Returns
    -------
    approx_coilset_psi:
        Matrix of coilset psi values approximated using TH
    Am_cos @ currents:
        TH cos coefficients for required number of degrees
    Am_sin @ currents:
        TH sin coefficients for required number of degrees

    """
    # Get coil positions and currents from equilibrium
    currents = np.array([eq.coilset[name].current for name in th_params.th_coil_names])

    cos_empty = len(cos_degrees_chosen) == 0
    sin_empty = len(sin_degrees_chosen) == 0

    # Initialise psi and A arrays
    approx_coilset_psi = np.zeros_like(th_params.R)
    A = np.zeros_like(th_params.R)
    # Useful combination
    Delta = np.cosh(th_params.tau) - np.cos(th_params.sigma)  # noqa: N806
    # Get sigma values for the grid
    sigma_mult_degree_cos = [m * th_params.sigma for m in cos_degrees_chosen]
    sigma_mult_degree_sin = [m * th_params.sigma for m in sin_degrees_chosen]

    factorial_m_cos = np.array([factorial(m) for m in cos_degrees_chosen])
    factorial_m_sin = np.array([factorial(m) for m in sin_degrees_chosen])

    # TH coefficient matrix
    Am_cos_current_function, Am_sin_current_function = (  # noqa: N806
        coil_toroidal_harmonic_amplitude_matrix(
            input_coils=eq.coilset,
            th_params=th_params,
            cos_degrees_chosen=cos_degrees_chosen,
            sin_degrees_chosen=sin_degrees_chosen,
        )
    )

    if cos_empty:
        A_cos = 0  # noqa: N806
    else:
        epsilon = 2 * np.ones(len(cos_degrees_chosen))
        epsilon[0] = 1
        Am_cos_matrix = (  # noqa: N806
            0
            if cos_empty
            else np.einsum(
                "ij, ikl, i, ikl -> ijkl",
                Am_cos_current_function,
                np.cos(sigma_mult_degree_cos),
                factorial_m_cos,
                legendre_q(
                    cos_degrees_chosen[:, None, None] - 1 / 2,
                    1,
                    np.cosh(th_params.tau),
                    n_max=30,
                ),
            )
        )
        A_cos = np.sqrt(2 / np.pi) * (  # noqa: N806
            np.einsum(
                "ijkl, i, kl, j -> kl",
                Am_cos_matrix,
                epsilon,
                np.sqrt(Delta),
                currents,
            )
        )
    if sin_empty:
        A_sin = 0  # noqa: N806
    else:
        epsilon = 2 * np.ones(len(sin_degrees_chosen))
        epsilon[0] = 1
        Am_sin_matrix = (  # noqa: N806
            0
            if sin_empty
            else np.einsum(
                "ij, ikl, i, ikl-> ijkl",
                Am_sin_current_function,
                np.sin(sigma_mult_degree_sin),
                factorial_m_sin,
                legendre_q(
                    sin_degrees_chosen[:, None, None] - 1 / 2,
                    1,
                    np.cosh(th_params.tau),
                    n_max=30,
                ),
            )
        )
        A_sin = np.sqrt(2 / np.pi) * (  # noqa: N806
            np.einsum(
                "ijkl, i, kl, j -> kl",
                Am_sin_matrix,
                epsilon,
                np.sqrt(Delta),
                currents,
            )
        )

    A = A_cos + A_sin
    # Calc approx coilset psi using \psi = A * R
    approx_coilset_psi = A * th_params.R
    Am_cos = [] if cos_empty else Am_cos_current_function @ currents  # noqa: N806
    Am_sin = [] if sin_empty else Am_sin_current_function @ currents  # noqa: N806
    return approx_coilset_psi, Am_cos, Am_sin


def toroidal_harmonic_approximation(  # noqa: PLR0915, PLR0914, RET503
    eq: Equilibrium,
    th_params: ToroidalHarmonicsParams | None = None,
    max_error_value: float = 0.5,
    psi_norm: float = 0.95,
    tol: float = 0.001,
    *,
    plot: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ToroidalHarmonicsParams,
]:
    """
    Calculate the toroidal harmonic (TH) amplitudes/coefficients.

    The objective: to get the lowest number of degrees to sufficiently approximate
    the coilset psi.

    Our selection requirements are satisfied when:
        - the RMS error for the coilset psi is lower than the max_error_value
        - adding an extra degree doesn't significantly lower the achieved error.


    Parameters
    ----------
    eq:
        Equilibria to use as starting point for approximation.
        We will approximate psi using THs - the aim is to keep the
        core plasma contribution fixed (using TH amplitudes as constraints)
        while being able to vary the vacuum (coil) contribution, so that
        we do not need to re-solve for the equilibria during optimisation
    th_params:
        Dataclass containing necessary parameters for use in TH approximation
    max_error_value:
        Maximum allowable error between equilibria psi and TH approximation
    psi_norm:
        Normalised flux value of the surface of interest.
        None value will default to LCFS.
    tol:
        Value used for error comparison to determine a sufficient combination of
        degrees
    plot:
        Whether or not to plot the results

    Returns
    -------
    error_success:
        The value of the error for the combination of degrees chosen
    combo_success:
        The degrees chosen
    total_psi_success:
        The total psi calculated using the TH approximation for the vacuum
        contribution using the combination of degrees chosen
    vacuum_psi_success:
        The TH approximation for the vacuum psi using the combination of degrees chosen
    cos_amplitudes_success:
        The cos amplitudes for the combination of degrees chosen
    sin_amplitudes_success:
        The sin amplitudes for the combination of degrees chosen
    th_params:
        Dataclass containing necessary parameters for use in TH approximation

    Raises
    ------
    EquilibriaError
        Problem not setup for harmonics

    """
    if th_params is None:
        R_0, Z_0 = eq.effective_centre()
        th_params = toroidal_harmonic_grid_and_coil_setup(eq=eq, R_0=R_0, Z_0=Z_0)

    # Get original flux surface from Bluemira for equilibrium
    original_fs = (
        eq.get_LCFS() if np.isclose(psi_norm, 1.0) else eq.get_flux_surface(psi_norm)
    )

    if eq.grid is None or eq.plasma is None:
        raise EquilibriaError("eq not setup for TH approximation.")
    # Use R and Z from th_params so we can compare psi over the same grid
    R_approx = th_params.R  # noqa: N806
    Z_approx = th_params.Z  # noqa: N806

    # Non TH contribution to psi field
    non_th_contribution_psi = eq.plasma.psi(R_approx, Z_approx)
    excluded_coils = list(set(eq.coilset.name) - set(th_params.th_coil_names))

    for coil in excluded_coils:
        non_th_contribution_psi += eq.coilset[coil].psi(R_approx, Z_approx)

    # Can't have more degrees than sampled psi
    max_degree = len(th_params.th_coil_names) - 1
    # Have cos and sin components so this must be half
    allowable_n_degrees = int(np.trunc(max_degree / 2))
    # allowable_n_degrees = (
    #     allowable_n_degrees if max_degree % 2 == 0 else allowable_n_degrees + 1
    # )

    # Find LCFS from TH approx
    approx_eq = deepcopy(eq)
    approx_eq.coilset.control = th_params.th_coil_names
    o_points, x_points = approx_eq.get_OX_points()

    degree_id = np.arange(0, max_degree)
    degree_values = np.arange(0, allowable_n_degrees)
    degree_values = np.append(degree_values, degree_values)

    # Initialise arrays to hold errors, combinations, amplitudes and psi values
    # Loop over combinations of degrees and save results which satisfy error condition
    errors_old = []
    combo_old = []
    cos_degrees_old = []
    sin_degrees_old = []
    cos_amplitudes_old = []
    sin_amplitudes_old = []
    total_psis_old = []
    vacuum_psis_old = []
    for n in np.arange(2, max_degree):
        errors = []
        combo = []
        cos_degrees = []
        sin_degrees = []
        cos_amplitudes = []
        sin_amplitudes = []
        total_psis = []
        vacuum_psis = []
        for c in combinations(degree_id, n):
            deg_id = [  # noqa: C416
                i for i in c
            ]
            cos_degrees_chosen = np.array([
                degree_values[i] for i in deg_id if i < allowable_n_degrees
            ])
            sin_degrees_chosen = np.array([
                degree_values[i] for i in deg_id if i >= allowable_n_degrees
            ])

            # Calculate psi using the combination of degrees selected in this iteration
            approximate_coilset_psi, cos_amps, sin_amps = (
                toroidal_harmonic_approximate_psi(
                    eq=eq,
                    th_params=th_params,
                    cos_degrees_chosen=cos_degrees_chosen,
                    sin_degrees_chosen=sin_degrees_chosen,
                )
            )
            # Want to mask to be able to calculate error in the plasma region only
            mask_matrix = np.zeros_like(R_approx)
            mask = _in_plasma(
                R_approx, Z_approx, mask_matrix, original_fs.xz.T, include_edges=True
            )

            total_psi_approx = approximate_coilset_psi + non_th_contribution_psi
            total_psi_approx_masked = mask * total_psi_approx
            total_psi_bluemira = (
                eq.coilset.psi(R_approx, Z_approx) + non_th_contribution_psi
            )
            total_psi_bluemira_masked = mask * total_psi_bluemira

            error = np.sqrt(
                np.sum(
                    (total_psi_bluemira_masked - total_psi_approx_masked) ** 2
                    / (len(R_approx) * len(Z_approx))
                )
            )

            # If error for this combination satisfies the max_error_value, then save this
            # result to the lists
            if error < max_error_value:
                errors.append(error)
                combo.append(c)
                cos_degrees.append(cos_degrees_chosen)
                sin_degrees.append(sin_degrees_chosen)
                total_psis.append(total_psi_approx)
                vacuum_psis.append(approximate_coilset_psi)
                cos_amplitudes.append(cos_amps)
                sin_amplitudes.append(sin_amps)

        # If sufficiently small change by adding extra degree, then
        # use the previous total number of degrees
        full = (len(errors_old) != 0) & (len(errors) != 0)
        succeeded = (np.min(errors_old) - np.min(errors) < tol) if full else False
        # TODO shall we add a condition that the psi_norm flux surface for _old should
        # be closed?
        if succeeded:
            index_chosen = np.argmin(errors_old)
            error_success = errors_old[index_chosen]
            combo_success = combo_old[index_chosen]
            cos_degrees_success = cos_degrees_old[index_chosen]
            sin_degrees_success = sin_degrees_old[index_chosen]
            total_psi_success = total_psis_old[index_chosen]
            vacuum_psi_success = vacuum_psis_old[index_chosen]
            cos_amplitude_success = cos_amplitudes_old[index_chosen]
            sin_amplitude_success = sin_amplitudes_old[index_chosen]

            if plot:
                plotting(
                    R_approx=R_approx,
                    Z_approx=Z_approx,
                    total_psi_success=total_psi_success,
                    psi_norm=psi_norm,
                    o_points=o_points,
                    x_points=x_points,
                    total_psi_bluemira=total_psi_bluemira,
                    th_params=th_params,
                    original_fs=original_fs,
                )

            return (
                error_success,
                combo_success,
                cos_degrees_success,
                sin_degrees_success,
                total_psi_success,
                vacuum_psi_success,
                cos_amplitude_success,
                sin_amplitude_success,
                th_params,
            )
        elif n == max_degree:  # noqa: RET505
            raise EquilibriaError(
                f"No combination of up to {max_degree} degrees gives an error of"
                f"{max_error_value} for chosen equilibrium! Please adjust the allowable error"
                "value or error tolerance values."
            )

        errors_old = errors
        combo_old = combo
        cos_degrees_old = cos_degrees
        sin_degrees_old = sin_degrees
        cos_amplitudes_old = cos_amplitudes
        sin_amplitudes_old = sin_amplitudes
        total_psis_old = total_psis
        vacuum_psis_old = vacuum_psis


def plotting(
    R_approx,
    Z_approx,
    total_psi_success,
    psi_norm,
    o_points,
    x_points,
    total_psi_bluemira,
    th_params,
    original_fs,
):
    # Find the flux surface to plot
    flux_surface = find_flux_surf(
        R_approx,
        Z_approx,
        total_psi_success,
        psi_norm,
        o_points=o_points,
        x_points=x_points,
    )
    approximation_flux_surface = Coordinates({
        "x": flux_surface[0],
        "z": flux_surface[1],
    })
    # Plotting if successful
    nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
    cmap = PLOT_DEFAULTS["psi"]["cmap"]
    # Plot TH approx for vacuum psi
    # f, ax = plt.subplots()
    # im = ax.contourf(
    #     th_params.R,
    #     th_params.Z,
    #     vacuum_psi_success,
    #     levels=nlevels,
    #     cmap=cmap,
    # )
    # ax.plot(
    #     approximation_flux_surface.x,
    #     approximation_flux_surface.z,
    #     color="r",
    #     label="TH FS",
    # )
    # ax.plot(
    #     original_fs.x,
    #     original_fs.z,
    #     color="blue",
    #     linestyle="dashed",
    #     label="BM FS",
    # )
    # ax.legend(loc="upper right")
    # f.colorbar(mappable=im)
    # plt.title("TH approximation for vacuum psi")
    # plt.show()

    # # Plot total psi using TH approx for vacuum psi
    # f, ax = plt.subplots()
    # im = ax.contourf(
    #     th_params.R,
    #     th_params.Z,
    #     total_psi_success,
    #     levels=nlevels,
    #     cmap=cmap,
    # )
    # ax.plot(
    #     approximation_flux_surface.x,
    #     approximation_flux_surface.z,
    #     color="r",
    #     label="TH FS",
    # )
    # ax.plot(
    #     original_fs.x,
    #     original_fs.z,
    #     color="blue",
    #     linestyle="dashed",
    #     label="BM FS",
    # )
    # ax.legend(loc="upper right")
    # f.colorbar(mappable=im)
    # plt.title("Total psi using TH approximation for vacuum psi")
    # plt.show()

    # Plot abs relative difference between approx total psi vs bluemira psi
    f, ax = plt.subplots()
    diff = np.abs(total_psi_success - total_psi_bluemira) / np.max(
        np.abs(total_psi_bluemira)
    )
    im = ax.contourf(th_params.R, th_params.Z, diff, levels=nlevels, cmap=cmap)
    ax.plot(
        approximation_flux_surface.x,
        approximation_flux_surface.z,
        color="r",
        label="TH FS",
    )
    ax.plot(
        original_fs.x,
        original_fs.z,
        color="blue",
        linestyle="dashed",
        label="BM FS",
    )
    ax.legend(loc="upper right")
    f.colorbar(mappable=im)
    # plt.title(
    #     "Absolute relative difference in total psi between TH approx and "
    #     "bluemira"
    # )
    plt.title("errors_old chosen")
    plt.show()

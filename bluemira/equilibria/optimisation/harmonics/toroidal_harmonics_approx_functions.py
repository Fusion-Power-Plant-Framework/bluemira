# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of functions used to approximate toroidal harmonics.
"""

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma, poch

from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
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
        hypergeometric function term, as defined in the equation above
    b:
        hypergeometric function term, as defined in the equation above
    c:
        hypergeometric function term, as defined in the equation above
    z:
        hypergeometric function term, as defined in the equation above
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
    """Evaluates the associated Legendre function of the first kind as a function of x.
    See https://dlmf.nist.gov/14.3#E18 for more information.
    Works for half integers.

    Valid for 1<x<infinity, and real \\mu and \\lambda

    .. math::
        P_{\\lambda}^{-\\mu}(x) = 2^{-\\mu} x^{\\lambda - \\mu} (x^2 - 1)^{\\mu/2}
                        F(\\frac{1}{2}(\\mu - \\lambda), \\frac{1}{2}(\\mu - \\lambda + 1);
                            \\mu + 1; 1 - \\frac{1}{x^2})

        where F is the hypergeometric function defined above as f_hypergeometric.

    Parameters
    ----------
    lam:
        legendre function term, as defined in the equation above
    mu:
        legendre function term, as defined in the equation above
    x:
        points at which to evaluate legendreP
    n_max:
        upper value for summation in f_hypergeometric

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
    kind as a function of x. See
    https://dlmf.nist.gov/14, https://dlmf.nist.gov/14.3#E10, and
    https://dlmf.nist.gov/14.3#E7 for more information.
    Works for half integers.

    Valid for 1<x<infinity, and real \\mu and \\lambda

    .. math::
        \\textbf{Q}_{\\lambda}^{\\mu}(x) = \\frac{\\pi^{\\frac{1}{2}} (x^2 - 1)^
                            {\\frac{\\mu}{2}}}{2^{\\lambda + 1} x^{\\lambda + \\mu + 1}}
                                F(\\frac{1}{2}(\\lambda + \\mu)+1, \\frac{1}{2}(\\lambda
                                + \\mu); \\lambda + \\frac{3}{2}; \\frac{1}{x^2})

        where F is the hypergeometric function defined above as f_hypergeometric.

    Parameters
    ----------
    lam:
        legendre function term, as defined in the equation above
    mu:
        legendre function term, as defined in the equation above
    x:
        points at which to evaluate legendreQ
    n_max:
        upper value for summation in f_hypergeometric

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
    A Dataclass holding necessary parameters for the toroidal harmonics approximation
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
    approximation, and find the coils located inside this region which need to be held
    fixed.

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
    cos_m_chosen: np.ndarray | None = None,
    sin_m_chosen: np.ndarray | None = None,
    sig_figures: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct coefficient matrices from toroidal harmonic amplitudes at given coil
    locations, for the specified cos and sin poloidal mode numbers (m).

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


    Where:
    m is the poloidal mode number,
    :math: P_{\\lambda}^{-\\mu} are the associated Legendre functions of the first kind,
    with lambda = m - 1/2 and mu = 1 (i.e., we use multiple poloidal mode numbers but
    toroidal mode number is fixed),
    :math: \\Delta_c = \\cosh{\\tau_c} - \\cos{\\sigma_c}.

    Note: the factorial term \\frac{(2m+1)!!}{2^m m!} is equivalent to 1 if m = 0,
    otherwise \\prod_{i=0}^{m-1} \\left( 1 + \\frac{1}{2(m-i)}\\right)

    Parameters
    ----------
    input_coils:
        Bluemira CoilSet
    th_params:
        Dataclass holding necessary parameters for the TH approximation
    cos_m_chosen:
        Poloidal mode numbers (m) chosen to be used for the cos components
    sin_m_chosen:
        Poloidal mode numbers (m) chosen to be used for the sin components
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

    # [number of poloidal modes, number of coils]
    currents2harmonics_cos = np.zeros([len(cos_m_chosen), np.size(tau_c)])
    currents2harmonics_sin = np.zeros([len(sin_m_chosen), np.size(tau_c)])

    # TH coefficients from function of the current distribution
    # outside of the region containing the core plasma
    # TH coefficients = currents2harmonics @ coil currents
    factorial_term_cos = np.array([
        np.prod(1 + 0.5 / np.arange(1, m + 1)) for m in cos_m_chosen
    ])
    factorial_term_sin = np.array([
        np.prod(1 + 0.5 / np.arange(1, m + 1)) for m in sin_m_chosen
    ])

    cos_empty = len(cos_m_chosen) == 0
    sin_empty = len(sin_m_chosen) == 0

    if cos_empty:
        # cos_m_chosen is None
        Am_cos = []  # noqa: N806
    else:
        currents2harmonics_cos[:, :] = (
            (MU_0 * 1.0 / 2.0 ** (5.0 / 2.0))
            * factorial_term_cos[:, None]
            * (np.sinh(tau_c)[None, :] / np.sqrt(Deltac)[None, :])
            * legendre_p(
                cos_m_chosen[:, None] - 1 / 2, 1, np.cosh(tau_c)[None, :], n_max=30
            )
        )
        sigma_c_mult_mode_cos = [m * th_params.sigma_c for m in cos_m_chosen]
        Am_cos = currents2harmonics_cos * np.cos(sigma_c_mult_mode_cos)  # noqa: N806
        Am_cos = sig_fig_round(Am_cos, sig_figures)  # noqa: N806

    if sin_empty:
        # sin_m_chosen is None
        Am_sin = []  # noqa: N806
    else:
        currents2harmonics_sin[:, :] = (
            (MU_0 * 1.0 / 2.0 ** (5.0 / 2.0))
            * factorial_term_sin[:, None]
            * (np.sinh(tau_c)[None, :] / np.sqrt(Deltac)[None, :])
            * legendre_p(
                sin_m_chosen[:, None] - 1 / 2, 1, np.cosh(tau_c)[None, :], n_max=30
            )
        )
        sigma_c_mult_mode_sin = [m * th_params.sigma_c for m in sin_m_chosen]
        Am_sin = currents2harmonics_sin * np.sin(sigma_c_mult_mode_sin)  # noqa: N806
        Am_sin = sig_fig_round(Am_sin, sig_figures)  # noqa: N806
    return Am_cos, Am_sin


def toroidal_harmonic_approximate_psi(
    eq: Equilibrium,
    th_params: ToroidalHarmonicsParams,
    cos_m_chosen: np.ndarray | None = None,
    sin_m_chosen: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate psi using toroidal harmonic amplitudes for the specified cos and sin
    poloidal mode numbers (m) as calculated in coil_toroidal_harmonic_amplitude_matrix.

    coil_toroidal_harmonic_amplitude_matrix returns Am_cos and Am_sin, which we use here.

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
    cos_m_chosen:
        Poloidal mode numbers (m) chosen to be used for the cos components
    sin_m_chosen:
        Poloidal mode numbers (m) chosen to be used for the sin components

    Returns
    -------
    approx_coilset_psi:
        Matrix of coilset psi values approximated using TH
    Am_cos @ currents:
        TH cos coefficients for required number of poloidal mode numbers (m)
    Am_sin @ currents:
        TH sin coefficients for required number of poloidal mode numbers (m)

    """
    # Get coil positions and currents from equilibrium
    currents = np.array([eq.coilset[name].current for name in th_params.th_coil_names])

    cos_empty = len(cos_m_chosen) == 0
    sin_empty = len(sin_m_chosen) == 0

    # Initialise psi and A arrays
    approx_coilset_psi = np.zeros_like(th_params.R)
    A = np.zeros_like(th_params.R)
    # Useful combination
    Delta = np.cosh(th_params.tau) - np.cos(th_params.sigma)  # noqa: N806
    # Get sigma values for the grid
    sigma_mult_mode_cos = [m * th_params.sigma for m in cos_m_chosen]
    sigma_mult_mode_sin = [m * th_params.sigma for m in sin_m_chosen]

    factorial_m_cos = np.array([factorial(m) for m in cos_m_chosen])
    factorial_m_sin = np.array([factorial(m) for m in sin_m_chosen])

    # TH coefficient matrix
    Am_cos_current_function, Am_sin_current_function = (  # noqa: N806
        coil_toroidal_harmonic_amplitude_matrix(
            input_coils=eq.coilset,
            th_params=th_params,
            cos_m_chosen=cos_m_chosen,
            sin_m_chosen=sin_m_chosen,
        )
    )

    if cos_empty:
        A_cos = 0  # noqa: N806
    else:
        epsilon = 2 * np.ones(len(cos_m_chosen))
        epsilon[0] = 1
        Am_cos_matrix = (  # noqa: N806
            0
            if cos_empty
            else np.einsum(
                "ij, ikl, i, ikl -> ijkl",
                Am_cos_current_function,
                np.cos(sigma_mult_mode_cos),
                factorial_m_cos,
                legendre_q(
                    cos_m_chosen[:, None, None] - 1 / 2,
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
        epsilon = 2 * np.ones(len(sin_m_chosen))
        epsilon[0] = 1
        Am_sin_matrix = (  # noqa: N806
            0
            if sin_empty
            else np.einsum(
                "ij, ikl, i, ikl-> ijkl",
                Am_sin_current_function,
                np.sin(sigma_mult_mode_sin),
                factorial_m_sin,
                legendre_q(
                    sin_m_chosen[:, None, None] - 1 / 2,
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


def _separate_psi_contributions(
    eq: Equilibrium, th_params: ToroidalHarmonicsParams
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate the psi contributions from fixed sources (plasma + excluded coils) and from
    potentially variable sources (coilset).

    Excluded coils are any coils not being used in the toroidal harmonic approximation,
    e.g. they fall within the region over which we are approximating and so their
    contribution must be fixed.

    Parameters
    ----------
    eq:
        Bluemira Equilibrium
    th_params:
        Dataclass holding necessary parameters for the TH approximation

    Returns
    -------
    coilset_psi - excluded_coil_psi:
        The psi contribution from coils for which we allow the currents to vary.
        These coils are outside of our approximation region
    plasma_psi + excluded_coil_psi:
        The psi contribution from fixed sources
    """
    plasma_psi = eq.plasma.psi(th_params.R, th_params.Z)
    coilset_psi = eq.coilset.psi(th_params.R, th_params.Z)
    excluded_coils = list(set(eq.coilset.name) - set(th_params.th_coil_names))

    excluded_coil_psi = np.zeros_like(plasma_psi)
    for coil in excluded_coils:
        excluded_coil_psi += eq.coilset[coil].psi(th_params.R, th_params.Z)
    return coilset_psi - excluded_coil_psi, plasma_psi + excluded_coil_psi


def _set_n_degrees_of_freedom(
    n_dof: int | None, max_harmonic_mode: int, max_n_dof: int
) -> int:
    """
    Determine the number of degrees of freedom to use. This is limited by the number
    of coils and by the maximum poloidal mode number (m) of the harmonic functions.


    Parameters
    ----------
    n_dof:
        The number of harmonic functions (and amplitudes) to choose.
        If this is None, then n_dof is calculated as the minimum of max_n_dof and
        2 * max_harmonic_mode
    max_harmonic_mode:
        The maximum poloidal mode number of the harmonic functions to use
    max_n_dof:
        The maximum number of degrees of freedom that could be used.
        This is equal to the number of coils used in the approximation

    Returns
    -------
    n_dof:
        The number of harmonic functions (and amplitudes) to choose.
        If None, will default to the number of "free" coils.
        Will warn if input n_dof is inappropriate and will instead select an
        appropriate n_dof to return
    """
    if n_dof is None:
        n_dof = min(max_n_dof, 2 * max_harmonic_mode)
    elif not (1 < n_dof <= max_n_dof):
        bluemira_warn(
            "Number of DOFs must be between 1 and the number of control coils"
            f"but this is not the case: 1 < {n_dof} <= {max_n_dof}."
            "Clipping accordingly."
        )
        n_dof = np.clip(n_dof, 1, max_n_dof)

    if n_dof > 2 * max_harmonic_mode:
        bluemira_warn(
            "n_degrees_of_freedom cannot be greater than 2 * max_harmonic_mode"
        )
        n_dof = 2 * max_harmonic_mode

    return n_dof


def _get_plasma_mask(
    eq: Equilibrium,
    th_params: ToroidalHarmonicsParams,
    psi_norm: float,
    *,
    plasma_mask: bool,
) -> int | np.ndarray:
    """
    Get a plasma mask to apply to the psi field.

    Parameters
    ----------
    eq:
        Bluemira Equilibrium
    th_params:
        Dataclass holding necessary parameters for the TH approximation
    plasma_mask:
        Whether or not to apply a mask to the error metric (within the psi_norm flux
        surface)
    psi_norm:
        Normalised flux value of the surface of interest.

    Returns
    -------
    mask:
        The plasma mask to be applied to the psi field
    """
    if plasma_mask:
        # Want to mask to be able to calculate error in the plasma region only
        psi_norm = np.clip(psi_norm, 0.0, 1.0)
        # Get the original reference flux surface from the equilibrium
        mask_fs = (
            eq.get_LCFS() if np.isclose(psi_norm, 1.0) else eq.get_flux_surface(psi_norm)
        )
        mask_matrix = np.zeros_like(th_params.R)
        mask = _in_plasma(
            th_params.R, th_params.Z, mask_matrix, mask_fs.xz.T, include_edges=True
        )
    else:
        # Do not apply a mask to the error
        mask = 1

    return mask


@dataclass
class ToroidalHarmonicsSelectionResult:
    """
    Toroidal harmonic selection result dataclass
    """

    cos_m: np.ndarray
    """Selected cosine poloidal mode numbers"""
    sin_m: np.ndarray
    """Selected sine poloidal mode numbers"""
    cos_amplitudes: np.ndarray
    """Selected cosine toroidal harmonic amplitudes"""
    sin_amplitudes: np.ndarray
    """Selected sine toroidal harmonic amplitudes"""
    error: float
    """Error of L2 norm when comparing approximated coilset psi to desired coilset psi"""
    coilset_psi: np.ndarray
    """Approximated coilset psi"""
    fixed_psi: np.ndarray
    """Background (fixed) psi"""


def brute_force_toroidal_harmonic_approximation(
    eq: Equilibrium,
    th_params: ToroidalHarmonicsParams | None = None,
    psi_norm: float = 0.95,
    n_degrees_of_freedom: int | None = None,
    max_harmonic_mode: int = 5,
    *,
    plasma_mask: bool = False,
) -> ToroidalHarmonicsSelectionResult:
    """
    Calculate the toroidal harmonic (TH) amplitudes/coefficients for a given
    number of degrees of freedom, using TH functions up to a given maximum
    poloidal mode number.

    The optimal selection of harmonic functions is carried out by brute force
    for the different combinations, using an L2 norm of the error across the
    full psi map. If `plasma_mask` is specified the error is evaluated as the
    L2 norm of the psi map within the specified flux surface.

    Parameters
    ----------
    eq:
        Equilibrium to use as starting point for approximation.
        We will approximate psi using THs - the aim is to keep the
        core plasma contribution fixed (using TH amplitudes as constraints)
        while being able to vary the vacuum (coil) contribution, so that
        we do not need to re-solve for the equilibria during optimisation
    th_params:
        Dataclass containing necessary parameters for use in TH approximation.
        If th_params is None, then function defaults to finding the th_params by using
        toroidal_harmonic_grid_and_coil_setup with the focus point set to the effective
        centre.
    psi_norm:
        Normalised flux value of the surface of interest.
        None value will default to 0.95 flux surface.
    n_degrees_of_freedom:
        The number of harmonic functions (and amplitudes) to choose.
        If None, will default to the number of "free" coils
    max_harmonic_mode:
        The maximum poloidal mode number of the harmonic functions to use
    plasma_mask:
        Whether or not to apply a mask to the error metric (within the psi_norm flux
        surface)

    Returns
    -------
    result:
        ToroidalHarmonicsSelectionResult

    Raises
    ------
    EquilibriaError
        Problem not setup for harmonics
    ValueError
        Number of degrees of freedom inappropriate
    """
    if eq.grid is None or eq.plasma is None:
        raise EquilibriaError("Equilibrium has not been run yet.")

    if th_params is None:
        R_0, Z_0 = eq.effective_centre()
        th_params = toroidal_harmonic_grid_and_coil_setup(eq=eq, R_0=R_0, Z_0=Z_0)

    n_degrees_of_freedom = _set_n_degrees_of_freedom(
        n_degrees_of_freedom,
        max_harmonic_mode,
        len(th_params.th_coil_names),
    )

    true_coilset_psi, fixed_psi = _separate_psi_contributions(eq, th_params)

    mask = _get_plasma_mask(
        eq=eq, th_params=th_params, psi_norm=psi_norm, plasma_mask=plasma_mask
    )

    dof_id = np.arange(0, 2 * max_harmonic_mode)
    mode_values = np.tile(np.arange(max_harmonic_mode), 2)

    error = np.inf

    for c in combinations(dof_id, n_degrees_of_freedom):
        mode_id = np.array(c)
        cos_m_chosen = mode_values[mode_id[mode_id < max_harmonic_mode]]
        sin_m_chosen = mode_values[mode_id[mode_id >= max_harmonic_mode]]

        # Calculate psi using the combination of poloidal mode numbers (m) selected in
        # this iteration
        approximate_coilset_psi, cos_amps, sin_amps = toroidal_harmonic_approximate_psi(
            eq=eq,
            th_params=th_params,
            cos_m_chosen=cos_m_chosen,
            sin_m_chosen=sin_m_chosen,
        )
        # Calculate L2 norm of the error between the approximated coilset psi and the
        # true coilset psi
        error_new = np.linalg.norm(mask * (approximate_coilset_psi - true_coilset_psi))

        # If the new error is less than the previously lowest error, then select the
        # current combination of poloidal mode numbers (m), amplitudes and associated psi
        if error_new < error:
            error = error_new
            cos_m = cos_m_chosen
            sin_m = sin_m_chosen
            coilset_psi = approximate_coilset_psi
            cos_amplitudes = cos_amps
            sin_amplitudes = sin_amps

    return ToroidalHarmonicsSelectionResult(
        cos_m=cos_m,
        sin_m=sin_m,
        cos_amplitudes=cos_amplitudes,
        sin_amplitudes=sin_amplitudes,
        error=error,
        coilset_psi=coilset_psi,
        fixed_psi=fixed_psi,
    )


def plot_toroidal_harmonic_approximation(
    eq: Equilibrium,
    th_params: ToroidalHarmonicsParams,
    result: ToroidalHarmonicsSelectionResult,
    psi_norm: float = 0.95,
):
    """
    Plot the toroidal harmonic approximation of the coilset psi and the bluemira
    true coilset psi on the same graph to allow comparison.
    Also plot the psi_norm flux surfaces for the approximation psi and the equilibrium
    coilset psi

    Parameters
    ----------
    eq:
        Bluemira Equilibrium
    th_params:
        Dataclass holding necessary parameters for the TH approximation
    result:
        ToroidalHarmonicsSelectionResult object returned from the
        brute_force_toroidal_approximation function
    psi_norm:
        Normalised flux value of the surface of interest.

    Returns
    -------
    f, ax:
        The Matplotlib figure and axis
    """
    original_fs = (
        eq.get_LCFS() if np.isclose(psi_norm, 1.0) else eq.get_flux_surface(psi_norm)
    )
    approx_fs = find_flux_surf(
        th_params.R,
        th_params.Z,
        result.coilset_psi + result.fixed_psi,
        psi_norm,
        *eq.get_OX_points(),
    )
    approx_fs = Coordinates({"x": approx_fs[0], "z": approx_fs[1]})

    f, ax = plt.subplots()
    ax.contour(
        th_params.R,
        th_params.Z,
        eq.coilset.psi(th_params.R, th_params.Z),
        levels=PLOT_DEFAULTS["psi"]["nlevels"],
        colors="black",
        linewidths=1,
    )
    ax.contour(
        th_params.R,
        th_params.Z,
        result.coilset_psi,
        levels=PLOT_DEFAULTS["psi"]["nlevels"],
        colors="red",
        linewidths=1,
    )

    ax.plot(
        approx_fs.x,
        approx_fs.z,
        color="r",
        label="TH FS",
        linestyle="dashed",
        lw=5,
    )
    ax.plot(
        original_fs.x,
        original_fs.z,
        color="blue",
        label="BM FS",
        lw=5,
    )
    ax.legend(loc="upper right")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    return f, ax

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
from math import factorial

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma, poch

from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print, bluemira_warn
from bluemira.equilibria.coils._grouping import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.find import find_flux_surf
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    fs_fit_metric,
)
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
    R_coils, Z_coils = (  # noqa: N806
        eq.coilset.get_control_coils().x,
        eq.coilset.get_control_coils().z,
    )
    tau_c, sigma_c = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=R_coils, Z=Z_coils)

    return ToroidalHarmonicsParams(
        R_0, Z_0, R, Z, R_coils, Z_coils, tau, sigma, tau_c, sigma_c, th_coil_names
    )


def coil_toroidal_harmonic_amplitude_matrix(
    input_coils: CoilSet,
    th_params: ToroidalHarmonicsParams,
    max_degree: int | None = None,
    sig_figures: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct coefficient matrices from toroidal harmonic amplitudes at given coil
    locations.

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
    max_degree:
        Maximum number of degrees to calculate up to
    sig_figures:
        Number of significant figures for rounding currents2harmonics values

    Returns
    -------
    Am_cos:
        Cos component of matrix of harmonic amplitudes,
    Am_sin:
        Sin component of matrix of harmonic amplitudes

    """
    if max_degree is None:
        max_degree = len(th_params.th_coil_names) - 1

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
    currents2harmonics = np.zeros([max_degree, np.size(tau_c)])

    # TH coefficients from function of the current distribution
    # outside of the region containing the core plasma
    # TH coefficients = currents2harmonics @ coil currents
    degrees = np.arange(0, max_degree)[:, None]
    factorial_term = np.array([
        np.prod(1 + 0.5 / np.arange(1, m + 1)) for m in range(max_degree)
    ])

    currents2harmonics[:, :] = (
        (MU_0 * 1.0 / 2.0 ** (5.0 / 2.0))
        * factorial_term[:, None]
        * (np.sinh(tau_c)[None, :] / np.sqrt(Deltac)[None, :])
        * legendre_p(degrees - 1 / 2, 1, np.cosh(tau_c)[None, :], n_max=30)
    )
    sigma_c_mult_degree = [m * th_params.sigma_c for m in range(max_degree)]
    Am_cos = currents2harmonics * np.cos(sigma_c_mult_degree)  # noqa: N806
    Am_sin = currents2harmonics * np.sin(sigma_c_mult_degree)  # noqa: N806
    return sig_fig_round(Am_cos, sig_figures), sig_fig_round(Am_sin, sig_figures)


def toroidal_harmonic_approximate_psi(
    eq: Equilibrium,
    th_params: ToroidalHarmonicsParams,
    max_degree: int | None = None,
    # TODO @clmould: add different ways to set th grid size
    # e.g. limit_type: TH_GRID_LIMIT = TH_GRID_LIMIT.LCFS or TH_GRID_LIMIT.COILSET
    # 3870
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate psi using toroidal harmonic amplitudes calculated in
    coil_toroidal_harmonic_amplitude_matrix.

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
    max_degree:
        Maximum number of degrees to calculate up to

    Returns
    -------
    approx_coilset_psi:
        Matrix of coilset psi values approximated using TH
    Am_cos @ currents:
        TH cos coefficients for required number of degrees
    Am_sin @ currents:
        TH sin coefficients for required number of degrees

    """
    if max_degree is None:
        max_degree = len(th_params.th_coil_names) - 1

    # Get coil positions and currents from equilibrium
    currents = np.array([eq.coilset[name].current for name in th_params.th_coil_names])

    # Initialise psi and A arrays
    approx_coilset_psi = np.zeros_like(th_params.R)
    A = np.zeros_like(th_params.R)
    # Useful combination
    Delta = np.cosh(th_params.tau) - np.cos(th_params.sigma)  # noqa: N806
    # Get sigma values for the grid
    sigma_mult_degree = [m * th_params.sigma for m in range(max_degree)]

    epsilon = 2 * np.ones(max_degree)
    epsilon[0] = 1
    factorial_m = np.array([factorial(m) for m in range(max_degree)])
    degrees = np.arange(0, max_degree)[:, None, None]
    # TH coefficient matrix
    Am_cos, Am_sin = coil_toroidal_harmonic_amplitude_matrix(  # noqa: N806
        input_coils=eq.coilset, th_params=th_params, max_degree=max_degree
    )

    Am_cos_sin = np.einsum(  # noqa: N806
        "ij, ikl -> ijkl", Am_cos, np.cos(sigma_mult_degree)
    ) + np.einsum("ij, ikl -> ijkl", Am_sin, np.sin(sigma_mult_degree))
    A = np.sqrt(2 / np.pi) * (
        np.einsum(
            "ijkl, i, i, kl, ikl, j -> kl",
            Am_cos_sin,
            epsilon,
            factorial_m,
            np.sqrt(Delta),
            legendre_q(degrees - 1 / 2, 1, np.cosh(th_params.tau), n_max=30),
            currents,
        )
    )
    approx_coilset_psi = A * th_params.R

    return approx_coilset_psi, Am_cos @ currents, Am_sin @ currents


def toroidal_harmonic_approximation(
    eq: Equilibrium,
    th_params: ToroidalHarmonicsParams | None = None,
    acceptable_fit_metric: float = 0.01,
    psi_norm: float = 1.0,
    *,
    plot: bool = False,
) -> tuple[
    ToroidalHarmonicsParams, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray
]:
    """
    Calculate the toroidal harmonic (TH) amplitudes/coefficients.

    Use a FS fit metric to determine the required number of degrees.

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
    acceptable_fit_metric:
        The default flux surface (FS) used for this metric is the LCFS.
        (psi_norm value is used to select an alternative)
        If the FS found using the TH approximation method perfectly matches the
        FS of the input equilibria then the fit metric = 0.
        A fit metric of 1 means that they do not overlap at all.
        fit_metric_value = total area within one but not both FSs /
        (input FS area + approximation FS area)
    psi_norm:
        Normalised flux value of the surface of interest.
        None value will default to LCFS.
    plot:
        Whether or not to plot the results

    Returns
    -------
    th_params:
        Dataclass containing necessary parameters for use in TH approximation
    Am_cos:
        TH cos coefficients/amplitudes for required number of degrees
    Am_sin:
        TH sin coefficients/amplitudes for required number of degrees
    degree:
        Number of degrees required for a TH approx with the desired fit metric
    fit_metric_value:
        Fit metric achieved
    approx_total_psi:
        Total psi obtained using the TH approximation
    approx_coilset_psi:
        Coilset psi obtained using the TH approximation

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

    bluemira_total_psi = eq.psi(R_approx, Z_approx)
    # Non TH contribution to psi field
    non_th_contribution_psi = eq.plasma.psi(R_approx, Z_approx)
    excluded_coils = list(set(eq.coilset.name) - set(th_params.th_coil_names))

    for coil in excluded_coils:
        non_th_contribution_psi += eq.coilset[coil].psi(R_approx, Z_approx)

    # Set min degree to save some time
    min_degree = 2
    # Can't have more degrees than sampled psi
    max_degree = len(th_params.th_coil_names) - 1
    # Have cos and sin components so this must be half
    allowable_n_degrees = int(np.trunc(max_degree / 2))

    # Find LCFS from TH approx
    approx_eq = deepcopy(eq)
    approx_eq.coilset.control = th_params.th_coil_names
    o_points, x_points = approx_eq.get_OX_points()

    for degree in range(min_degree, allowable_n_degrees):
        # Construct matrix from harmonic amplitudes for the coils and approximate psi
        approx_coilset_psi, Am_cos, Am_sin = toroidal_harmonic_approximate_psi(  # noqa: N806
            eq=eq, th_params=th_params, max_degree=degree + 1
        )
        # Add the non TH coil contribution to the total
        approx_total_psi = approx_coilset_psi + non_th_contribution_psi

        # Find flux surface for our TH approximation equilibrium
        f_s = find_flux_surf(
            R_approx,
            Z_approx,
            approx_total_psi,
            psi_norm,
            o_points=o_points,
            x_points=x_points,
        )
        approx_fs = Coordinates({"x": f_s[0], "z": f_s[1]})

        # Compare staring equilibrium to new approximate equilibrium
        fit_metric_value = fs_fit_metric(original_fs, approx_fs)

        bluemira_print(
            f"Fit metric value = {fit_metric_value} using {degree + 1} degrees."
        )

        if fit_metric_value <= acceptable_fit_metric:
            break
        if degree + 1 == max_degree:
            bluemira_warn(
                "You may need to use more degrees for a fit metric of"
                f" {acceptable_fit_metric}!"
            )

    # Plot comparing original psi to the TH approximation
    if plot:
        nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
        cmap = PLOT_DEFAULTS["psi"]["cmap"]
        # Plot difference between approx total psi and bluemira total psi
        total_psi_diff = np.abs(approx_total_psi - bluemira_total_psi) / np.max(
            np.abs(bluemira_total_psi)
        )
        f, ax = plt.subplots()
        ax.plot(approx_fs.x, approx_fs.z, color="red", label="Approx FS from TH")
        ax.plot(
            original_fs.x,
            original_fs.z,
            color="c",
            linestyle="dashed",
            label="FS from Bluemira",
        )
        im = ax.contourf(R_approx, Z_approx, total_psi_diff, levels=nlevels, cmap=cmap)
        f.colorbar(mappable=im)
        # ax.set_title("|th_approx_psi - psi| / max(psi)")
        ax.legend(loc="upper right")
        eq.coilset.plot(ax=ax)
        plt.show()

    return (
        th_params,
        Am_cos,
        Am_sin,
        degree + 1,
        fit_metric_value,
        approx_total_psi,
        approx_coilset_psi,
    )

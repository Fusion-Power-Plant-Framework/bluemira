# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of functions used to approximate toroidal harmonics.
"""

from math import factorial

import numpy as np
from scipy.special import gamma, poch

from bluemira.base.constants import MU_0
from bluemira.equilibria.coils._grouping import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
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
        F = F + (poch(a, s) * poch(b, s)) / (gamma(c + s) * factorial(s)) * z**s  # noqa: PLR6104
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


def coil_toroidal_harmonic_amplitude_matrix(
    input_coils: CoilSet,
    R_0: float,
    Z_0: float,
    th_coil_names: list,
    max_degree: int = 5,
    sig_figures: int = 15,
) -> np.ndarray:
    """
    Construct matrix from toroidal harmonic amplitudes at given coil locations.

    To get an array of toroidal harmonic amplitudes/coefficients (A_m)
    which can be used in a toroidal harmonic approximation of the
    vacuum/coil contribution to the poloidal flux (psi) do:

    A_m = matrix harmonic amplitudes @ vector of coil currents

    A_m can be used as constraints in optimisation, see toroidal_harmonics_constraint.
    todo write toroidal_harmonics_constraint.

    N.B. for a single filament (coil):

    .. math::
        A_{m} = \\frac{\\mu_{0} I_{c}}{2^{5/2}} \\frac{(2m+1)!!}{2^m m!}
        \\frac{\\sinh{\\tau_{c}}}{\\Delta_{c}^{1/2}}
        P_{m-\\frac{1}{2}}^{-1}(\\cosh{\\tau_c})


    Where m = poloidal mode number, :math: P_{\\lambda}^{-\\mu} are the associated
    Legendre functions of the first kind of degree lambda and order minus mu, and :math:
    \\Delta_c = \\cosh{\\tau_c} - \\cos{\\sigma_c}.

    Parameters
    ----------
    input_coils:
        Bluemira CoilSet
    R_0:
        R coordinate of the focus point in cylindrical coordinates
    Z_0:
        Z coordinate of the focus point in cylindrical coordinates
    th_coil_names:
        Names of the coils to use with TH approximation
    max_degree:
        Maximum degree of harmonic to calculate up to
    sig_figures:
        Number of significant figures for rounding currents2harmonics values

    Returns
    -------
    currents2harmonics:
        Matrix of harmonic amplitudes

    """
    # Coils
    x_c = []
    z_c = []
    for n in th_coil_names:
        x_c.append(input_coils[n].x)
        z_c.append(input_coils[n].z)

    x_c = np.array(x_c)
    z_c = np.array(z_c)

    # Toroidal coords
    tau_c, sigma_c = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=x_c, Z=z_c)
    # Useful combination
    Deltac = np.cosh(tau_c) - np.cos(sigma_c)  # noqa: N806

    # [number of degrees, number of coils]
    currents2harmonics = np.zeros([max_degree + 1, np.size(tau_c)])

    # TH coefficients from function of the current distribution
    # outside of the region containing the core plamsa
    # TH coefficients = currents2harmonics @ coil currents
    degrees = np.arange(0, max_degree + 1)[:, None]
    factorial_term = np.array([
        np.prod(1 + 0.5 / np.arange(1, m + 1)) for m in range(max_degree + 1)
    ])

    currents2harmonics[:, :] = (
        (MU_0 * 1.0 / 2.0 ** (5.0 / 2.0))
        * factorial_term[:, None]
        * (np.sinh(tau_c)[None, :] / np.sqrt(Deltac)[None, :])
        * legendre_p(degrees - 1 / 2, 1, np.cosh(tau_c)[None, :], n_max=30)
    )

    return sig_fig_round(currents2harmonics, sig_figures)


def toroidal_harmonic_approximate_psi(
    eq: Equilibrium,
    R_0: float,
    Z_0: float,
    max_degree: int = 5,
    # TODO add different ways to set th grid size
    # e.g. limit_type: TH_GRID_LIMIT = TH_GRID_LIMIT.LCFS or TH_GRID_LIMIT.COILSET
):
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
    R_0:
        R coordinate of the focus point in cylindrical coordinates
    Z_0:
        Z coordinate of the focus point in cylindrical coordinates
    max_degree:
        Maximum degree of harmonic to calculate up to

    Returns
    -------
    psi_approx:
        Matrix of psi values aproximated using TH
    R:
        grid values used in the approximation
    Z:
        grid values used in the approximation
    """
    # Find region over which to approximate psi using TH
    lcfs = eq.get_LCFS()
    lcfs_tau, _ = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=lcfs.x, Z=lcfs.z)
    tau_lcfs_limit = np.min(lcfs_tau)

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

    # Get coil positions and currents from equilibrium
    currents = eq.coilset.current
    R_coils = eq.coilset.x  # noqa: N806
    Z_coils = eq.coilset.z  # noqa: N806
    # Initialise psi and A arrays
    psi_approx = np.zeros_like(R)
    A = np.zeros_like(R)
    # Useful combination
    Delta = np.cosh(tau) - np.cos(sigma)  # noqa: N806
    # Get sigma values for the coils
    _, sigma_c = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=R_coils, Z=Z_coils)
    sigma_c_mult_degree = [m * sigma_c for m in range(max_degree + 1)]
    sigma_mult_degree = [m * sigma for m in range(max_degree + 1)]

    epsilon = 2 * np.ones(max_degree + 1)
    epsilon[0] = 1
    factorial_m = np.array([factorial(m) for m in range(max_degree + 1)])
    degrees = np.arange(0, max_degree + 1)[:, None, None]
    # TH coefficient matrix
    A_m = coil_toroidal_harmonic_amplitude_matrix(  # noqa: N806
        input_coils=eq.coilset,
        R_0=R_0,
        Z_0=Z_0,
        th_coil_names=eq.coilset.name,
        max_degree=max_degree,
    )
    Am_cos = currents @ np.transpose(A_m * np.cos(sigma_c_mult_degree))  # noqa: N806
    Am_sin = currents @ np.transpose(A_m * np.sin(sigma_c_mult_degree))  # noqa: N806

    A_coil_matrix = Am_cos[:, None, None] * epsilon[:, None, None] * factorial_m[  # noqa: N806
        :, None, None
    ] * np.sqrt(2 / np.pi) * np.sqrt(Delta[None, :]) * legendre_q(
        degrees - 1 / 2, 1, np.cosh(tau), n_max=30
    ) * np.cos(sigma_mult_degree) + Am_sin[:, None, None] * epsilon[
        :, None, None
    ] * factorial_m[:, None, None] * np.sqrt(2 / np.pi) * np.sqrt(
        Delta[None, :]
    ) * legendre_q(degrees - 1 / 2, 1, np.cosh(tau), n_max=30) * np.sin(
        sigma_mult_degree
    )
    A = np.array(np.sum(A_coil_matrix, axis=0), dtype=float)
    psi_approx = A * R
    return psi_approx, R, Z

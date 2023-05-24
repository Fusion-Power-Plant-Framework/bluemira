# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
A collection of simple 0-D rules of thumb for tokamak plasmas.
"""

import numpy as np

from bluemira.base.constants import EV_TO_J, K_BOLTZMANN, MU_0
from bluemira.plasma_physics.collisions import coulomb_logarithm, spitzer_conductivity


def estimate_loop_voltage(
    R_0: float, B_t: float, Z_eff: float, T_e: float, n_e: float, q_0: float
) -> float:
    """
    A 0-D estimate of the loop voltage during burn

    Parameters
    ----------
    R_0:
        Major radius [m]
    B_t:
        Toroidal field on axis [T]
    Z_eff:
        Effective charge [a.m.u.]
    T_e:
        Electron temperature on axis [eV]
    n_e:
        Electron density [1/m^3]
    q_0:
        Safety factor on axis

    Returns
    -------
    Loop voltage during burn [V]

    Notes
    -----
    H. Zohm, W. Morris (2022)

    \t:math:`v_{loop}=2\\pi R_{0}\\dfrac{2\\pi B_{t}}{\\mu_{0}q_{0}\\sigma_{0}R_{0}}`

    where :math:`\\sigma_{0}` is the Spitzer conductivity on axis:
    \t:math:`\\sigma_{0} = 1.92e4 (2-Z_{eff}^{-1/3}) \\dfrac{T_{e}^{3/2}}{Z_{eff}ln\\Lambda}`

    Assumes no non-inductive current on axis

    Assumes a circular cross-section on axis

    There is no neo-classical resistivity on axis because there are no trapped particles
    """  # noqa: W505
    ln_lambda = coulomb_logarithm(T_e * EV_TO_J / K_BOLTZMANN, n_e)
    sigma = spitzer_conductivity(Z_eff, T_e, ln_lambda)

    # Current density on axis
    j_0 = 2 * B_t / (MU_0 * q_0 * R_0)
    v_loop = 2 * np.pi * R_0 * j_0 / sigma
    return v_loop


def estimate_Le(A: float, kappa: float) -> float:  # noqa: N802
    """
    Estimate the normalised external plasma self-inductance.

    Parameters
    ----------
    A:
        Last closed flux surface aspect ratio
    kappa:
        Last closed flux surface elongation

    Returns
    -------
    Normalised plasma external inductance

    Notes
    -----
    Hirshman and Neilson, 1986
    https://pubs.aip.org/aip/pfl/article/29/3/790/944223/External-inductance-of-an-axisymmetric-plasma
    Assuming a LCFS parameterisation as per:
    :py:func:`bluemira.equilibria.shapes.flux_surface_hirshman`
    """
    eps = 1 / A
    sqrt_eps = np.sqrt(eps)

    a = (
        (1 + 1.81 * sqrt_eps + 2.05 * eps) * np.log(8 * A)
        - 2.0
        - 9.25 * sqrt_eps
        + 1.21 * eps
    )
    b = 0.73 * sqrt_eps * (1 + 2 * eps**4 - 6 * eps**5 + 3.7 * eps**6)
    return a * (1 - eps) / (1 - eps + b * kappa)


def estimate_M(A: float, kappa: float) -> float:  # noqa: N802
    """
    Estimate the plasma mutual inductance.

    Parameters
    ----------
    A:
        Last closed flux surface aspect ratio
    kappa:
        Last closed flux surface elongation

    Returns
    -------
    Plasma mutual inductance

    Notes
    -----
    Hirshman and Neilson, 1986
    https://pubs.aip.org/aip/pfl/article/29/3/790/944223/External-inductance-of-an-axisymmetric-plasma
    Assuming a LCFS parameterisation as per:
    :py:func:`bluemira.equilibria.shapes.flux_surface_hirshman`
    """
    eps = 1 / A

    c = 1 + 0.98 * eps**2 + 0.49 * eps**4 + 1.47 * eps**6
    d = 0.25 * eps * (1 + 0.84 * eps - 1.44 * eps**2)
    return (1 - eps) ** 2 / ((1 - eps) ** 2 * c + d * np.sqrt(kappa))

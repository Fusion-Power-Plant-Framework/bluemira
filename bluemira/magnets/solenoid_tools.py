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
Tools for simple solenoid calculations.
"""

import numpy as np

from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_warn


def calculate_B_max(
    rho_j: float, r_inner: float, r_outer: float, height: float
) -> float:
    """
    Calculate the maximum field in a solenoid

    Parameters
    ----------
    rho_j:
        Current density across the solenoid winding pack
    r_inner:
        Solenoid inner radius
    r_outer:
        Solenoid outer radius
    height:
        Solenoid vertical extent

    Return
    ------
    Maximum field in a solenoid

    Notes
    -----
    M. Wilson, Superconducting Magnets, 1983, p22
    """
    half_height = 0.5 * height
    alpha = r_outer / r_inner
    beta = half_height / r_inner

    if not (1.0 < alpha < 2.0):
        bluemira_warn(
            f"Solenoid B_max calculation parameter alpha is not between 1.0 and 2.0: {alpha=:.2f}"
        )
    if beta <= 0.5:
        bluemira_warn(
            f"Solenoid B_max calculation parameter beta is not greater than 0.5: {beta=:.2f}"
        )

    b_0 = (
        rho_j
        * MU_0
        * half_height
        * np.log((alpha + np.hypot(alpha, beta)) / (1 + np.hypot(1, beta)))
    )

    tail = 0.0
    alpha_1 = alpha - 1.0
    if beta > 3.0:
        b_c = (3.0 / beta) ** 2
        factor = b_c * (1.007 + 0.0055 * alpha_1)
        tail = (1 - b_c) * MU_0 * rho_j * (r_outer - r_inner)

    elif beta > 2.0:
        b_c = beta - 2.0
        factor = 1.025 - 0.018 * b_c + (0.01 - 0.0045 * b_c) * alpha_1

    elif beta > 1.0:
        b_c = beta - 1.0
        factor = 1.117 - 0.092 * b_c + (0.01 * b_c) * alpha_1

    elif beta > 0.75:
        b_c = beta - 0.75
        factor = 1.3 - 0.732 * b_c + (-0.05 + 0.2 * b_c) * alpha_1

    else:
        b_c = beta - 0.5
        factor = 1.64 - 1.4 * b_c + (-0.2 + 0.6 * b_c) * alpha_1

    B_max = factor * b_0 + tail

    return B_max


def calculate_hoop_stress(
    B_in: float,
    B_out: float,
    rho_j: float,
    r_inner: float,
    r_outer: float,
    r: float,
    poisson_ratio: float = 0.3,
) -> float:
    """
    Calculate the hoop stress at a radial location in a solenoid

    Parameters
    ----------
    r:
        Radius at which to calculate

    Return
    ------
    Hoop stress at the radial location
    """
    alpha = r_outer / r_inner
    eps = r / r_inner
    nu = poisson_ratio

    K = (alpha * B_in - B_out) * rho_j * r_inner / (alpha - 1)  # noqa: N806
    M = (B_in - B_out) * rho_j * r_inner / (alpha - 1)  # noqa: N806
    a = K * (2 + nu) / (3 * (alpha + 1))
    b = (
        1.0
        + alpha
        + alpha**2 * (1 + 1 / eps**2)
        - eps * (1 + 2 * nu) * (alpha + 1) / (2 + nu)
    )
    c = M * (3 + nu) / 8
    d = 1.0 + alpha**2 * (1 + 1 / eps**2) - eps**2 * (1 + 3 * nu) / (3 + nu)
    hoop_stress = a * b - c * d

    return hoop_stress


def calculate_flux_max(B_max: float, r_inner: float, r_outer: float) -> float:
    """
    Calculate the maximum flux achievable from a solenoid

    Parameters
    ----------
    B_max:
        Maximum field in the solenoid [T]
    r_inner:
        Solenoid inner radius [m]
    r_outer:
        Solenoid outer radius [m]

    Returns
    -------
    Maximum flux achievable from a solenoid [V.s]
    """
    return np.pi / 3 * B_max * (r_outer**2 + r_inner**2 + r_outer * r_inner)

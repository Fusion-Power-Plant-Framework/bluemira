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
    rho_j: float, r_inner: float, r_outer: float, half_height: float
) -> float:
    """
    Parameters
    ----------

    Return
    ------
    Maximum field in a solenoid
    """
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
    if beta > 3.0:
        a = (3.0 / beta) ** 2
        factor = a * (1.007 + 0.0055 * (alpha - 1))
        tail = (1 - a) * MU_0 * rho_j * (r_outer - r_inner)

    elif beta > 2.0:
        factor = 1.025 - 0.018 * (beta - 2) + (0.01 - 0.0045 * (beta - 2)) * (alpha - 1)

    elif beta > 1.0:
        factor = 1.117 - 0.092 * (beta - 1) + (0.01 * (beta - 1)) * (alpha - 1)

    elif beta > 0.75:
        factor = (
            1.3 - 0.732 * (beta - 0.75) + (-0.05 + 0.2 * (beta - 0.75)) * (alpha - 1)
        )

    else:
        factor = 1.64 - 1.4 * (beta - 0.5) + (-0.2 + 0.6 * (beta - 0.5)) * (alpha - 1)

    B_max = factor * b_0 + tail

    return B_max

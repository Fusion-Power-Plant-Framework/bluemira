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
Fatigue model
"""

from dataclasses import dataclass


@dataclass
class ParisFatigueMaterial:
    """
    Material properties for the Paris fatigue model
    """

    C: float  # Paris law material constant
    m: float  # Paris law material exponent
    K_ic: float  # Fracture toughness


@dataclass
class ParisFatigueSafetyFactors:
    """
    Safety factors for the Paris fatigue model
    """

    sf_n_cycle: float
    sf_radial_crack: float
    sf_vertical_crack: float
    sf_fracture: float


@dataclass
class Crack:
    """
    Crack description for the Paris fatigue model
    """

    depth: float  # a
    width: float  # c


def calculate_n_pulses(
    tk_radial: float,
    tk_vertical: float,
    max_hoop_stress: float,
    residual_stress: float,
    walker_coeff: float,
    initial_crack: Crack,
    material: ParisFatigueMaterial,
    safety: ParisFatigueSafetyFactors,
) -> int:
    """
    Calculate the number of plasma pulses possible prior to fatigue.

    Parameters
    ----------

    Returns
    -------
    Number of plasma pulses

    Notes
    -----
    Assumes two stress cycles per pulse.
    Calculates using the cycle-by-cycle method.
    """
    mean_stress_ratio = residual_stress / (max_hoop_stress + residual_stress)

    C_r = material.C * (1 - mean_stress_ratio) ** (material.m * (walker_coeff - 1))

    a = initial_crack.depth
    c = initial_crack.width
    n_cycles = 0

    while True:
        a += 0
        c += 0
        n_cycles += 0

    return n_cycles // 2

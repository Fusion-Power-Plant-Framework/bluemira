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
FE stress interpolations
"""

from typing import List

import numpy as np


def hermite_displacement(n: int) -> np.ndarray:
    """
    \t:math:`v(x)`
    """
    x = np.linspace(0, 1, n)
    matrix = np.zeros((n, 4))
    matrix[:, 0] = 1 - 3 * x**2 + 2 * x**3
    matrix[:, 1] = x - 2 * x**2 + x**3
    matrix[:, 2] = 3 * x**2 - 2 * x**3
    matrix[:, 3] = -(x**2) + x**3
    return matrix


def hermite_curvature(n: int) -> np.ndarray:
    """
    \t:math:`M = EI\\dfrac{\\partial^2 v}{\\partial^2 x}`
    """
    x = np.linspace(0, 1, n)
    matrix = np.zeros((n, 4))
    matrix[:, 0] = -6 + 12 * x
    matrix[:, 1] = -4 + 6 * x
    matrix[:, 2] = 6 - 12 * x
    matrix[:, 3] = -2 + 6 * x
    return matrix


def hermite_shear(n: int) -> np.ndarray:
    """
    \t:math:`V = EI\\dfrac{\\partial^3 v}{\\partial^3 x}`
    """
    matrix = np.zeros((n, 4))
    matrix[:, 0] = 12 * np.ones(n)
    matrix[:, 1] = 6 * np.ones(n)
    matrix[:, 2] = -12 * np.ones(n)
    matrix[:, 3] = 6 * np.ones(n)
    return matrix


def hermite_polynomials(n: int) -> List[np.ndarray]:
    """
    Calculate all the base Hermite polynomials

    Parameters
    ----------
    n:
        The number of interpolation points
    """
    # NOTE: still need [:, 1] and [:, 3] to be multiplied by element length
    n_1 = hermite_displacement(n)
    n_2 = hermite_curvature(n)
    n_3 = hermite_shear(n)
    return [n_1, n_2, n_3]

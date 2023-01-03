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
Fitting tools
"""
from typing import List

import numpy as np
from scipy.linalg import lstsq
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def surface_fit(x, y, z, order: int = 2, n_grid: int = 30):
    """
    Fit a polynomial surface to a 3-D data set.

    Parameters
    ----------
    x: np.array(n)
        The x values of the data set
    y: np.array(n)
        The y values of the data set
    z: np.array(n)
        The z values of the data set
    order: int
        The order of the fitting polynomial
    n_grid: int
        The number of gridding points to use on the x and y data

    Returns
    -------
    x2d: np.array(n_grid, n_grid)
        The gridded x data
    y2d: np.array(n_grid, n_grid)
        The gridded y data
    zz: np.array(n_grid, n_grid)
        The gridded z fit data
    coeffs: list
        The list of polynomial coefficents
    r2: float
        The R^2 score of the fit

    Notes
    -----
    The coefficients are ordered by power, and by x and y. For an order = 2
    polynomial, the resultant equation would be:

    \t:math:`c_{1}x^{2}+c_{2}y^{2}+c_{3}xy+c_{4}x+c_{5}y+c_{6}`
    """
    x, y, z = np.array(x), np.array(y), np.array(z)
    if len(x) != len(y) != len(z):
        raise ValueError("x, y, and z must be of equal length.")

    n = len(x)
    poly = PolynomialFeatures(order)

    eq_matrix = poly.fit_transform(np.c_[x, y])

    index = powers_arange(poly.powers_)
    eq_matrix = eq_matrix[:, index]

    coeffs, _, _, _ = lstsq(eq_matrix, z)

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Grid the x and y arrays
    x2d, y2d = np.meshgrid(
        np.linspace(x_min, x_max, n_grid), np.linspace(y_min, y_max, n_grid)
    )
    xx = x2d.flatten()
    yy = y2d.flatten()

    zz = np.dot(poly.fit_transform(np.c_[xx, yy])[:, index], coeffs).reshape(x2d.shape)

    z_predicted = np.dot(eq_matrix, coeffs).reshape(n)

    return x2d, y2d, zz, coeffs, r2_score(z, z_predicted)


def powers_arange(powers: np.ndarray) -> List:
    """
    Reorder powers index to order by power from 1st to nth index.

    Parameters
    ----------
    powers: np.ndarray
        array of powers

    Returns
    -------
    index: list
        index to rearrange array

    """
    return sorted(
        range(powers.shape[0]),
        key=lambda x: (np.sum(powers[x]), np.max(powers[x])),
        reverse=True,
    )

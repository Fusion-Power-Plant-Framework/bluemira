# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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

    # Ordering the array is harder than the actual fit!
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


def powers_arange(powers):
    """
    Reorder powers index to order by power from 1st to nth index.

    Parameters
    ----------
    powers: np.ndarray
        array of powers (from `PolynomialFeatures().powers_`)

    Returns
    -------
    index: list
        index to rearrange array

    """

    def max_index(pl):
        index_order = []
        for mx in range(pl.shape[1]):
            index_order += [pl[:, mx].argmax()]
        return index_order

    def remain_list(p_len, index_order):
        return list(set(p_len) - set(index_order))

    reorder_list = []
    glob_list = []
    for i, p in enumerate(np.append(powers, np.atleast_2d(np.zeros(2)), axis=0)):
        if powers[i - 1].sum() == p.sum():
            # Add index of equal power sum to list
            if reorder_list == []:
                # Add first one to index
                reorder_list += [i - 1]
            reorder_list += [i]
        elif reorder_list != []:
            reorder_list = np.array(reorder_list)

            # Change order of max powers in reorder_list
            p_list = powers[reorder_list]
            p_len = np.arange(len(p_list))

            index_order = max_index(p_list)
            rem_list = np.array(remain_list(p_len, index_order))

            # Change order of < max power in reorder_list
            while len(rem_list) > 1:
                reo_l = reorder_list[rem_list]
                index_order += rem_list[max_index(powers[reo_l])].tolist()
                rem_list = remain_list(p_len, index_order)

            # Single value remaining in odd lengthed reorder_list
            if len(rem_list) > 0:
                index_order += [rem_list[0]]

            glob_list = reorder_list[index_order].tolist() + glob_list

            reorder_list = []

    # Add powers [0, 0] to index
    glob_list += [0]

    return glob_list

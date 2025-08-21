# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Utils for magnets"""

import numpy as np


def summation(arr: list | np.ndarray):
    """
    Compute the simple summation of the series

    Parameters
    ----------
    arr:
        list or numpy array containing the elements on which the serie shall
        be calculated

    Returns
    -------
    Result: float

    i.e.

    Y = sum(x1 + x2 + x3 ...)
    """
    return np.sum(arr)


def reciprocal_summation(arr: list | np.ndarray):
    """
    Compute the inverse of the summation of a reciprocal series

    Parameters
    ----------
    arr:
        list or numpy array containing the elements on which the serie shall
        be calculated

    Returns
    -------
    Result: float

    i.e.

    Y = [sum(1/x1 + 1/x2 + 1/x3 ...)]^-1
    """
    return (np.sum((1 / element) for element in arr)) ** -1


def delayed_exp_func(x0: float, tau: float, t_delay: float = 0):
    """
    Delayed Exponential function

    x = x0 * exp(-(t-t_delay)/tau)

    Parameters
    ----------
    x0: float
        initial value
    tau: float
        characteristic time constant
    t_delay: float
        delay time

    Returns
    -------
    A Callable - exponential function

    """

    def fun(t):
        x = x0
        if t > t_delay:
            x = x0 * np.exp(-(t - t_delay) / tau)
        return x

    return fun

# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Utils for magnets"""

import numpy as np


def serie_r(arr: list | np.ndarray):
    """
    Compute the serie (as for resistance)

    Parameters
    ----------
    arr:
        list or numpy array containing the elements on which the serie shall
        be calculated

    Returns
    -------
    Result: float
    """
    return np.sum(arr)


def parall_r(arr: list | np.ndarray):
    """
    Compute the parallel (as for resistance)

    Parameters
    ----------
    arr:
        list or numpy array containing the elements on which the parallel
        shall be calculated

    Returns
    -------
    Result: float
    """
    out = 0
    for i in range(len(arr)):
        out += 1 / arr[i]
    return out**-1


def serie_k(arr: list | np.ndarray):
    """
    Compute the serie (as for spring)

    Parameters
    ----------
    arr:
        list or numpy array containing the elements on which the serie
        shall be calculated

    Returns
    -------
    Result: float
    """
    out = 0
    for i in range(len(arr)):
        out += 1 / arr[i]
    return out**-1


def parall_k(arr: list | np.ndarray):
    """
    Compute the parallel (as for spring)

    Parameters
    ----------
    arr:
        list or numpy array containing the elements on which the parallel
        shall be calculated

    Returns
    -------
    Result: float
    """
    return np.sum(arr)


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

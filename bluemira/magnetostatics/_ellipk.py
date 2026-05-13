# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from typing import TypeVar

import numba as nb
import numpy as np
import numpy.typing as npt

_FloatOrArray = TypeVar("_FloatOrArray", npt.NDArray, float)


_P = (
    1.37982864606273237150e-4,
    2.28025724005875567385e-3,
    7.97404013220415179367e-3,
    9.85821379021226008714e-3,
    6.87489687449949877925e-3,
    6.18901033637687613229e-3,
    8.79078273952743772254e-3,
    1.49380448916805252718e-2,
    3.08851465246711995998e-2,
    9.65735902811690126535e-2,
    1.38629436111989062502e0,
)

_Q = (
    2.94078955048598507511e-5,
    9.14184723865917226571e-4,
    5.94058303753167793257e-3,
    1.54850516649762399335e-2,
    2.39089602715924892727e-2,
    3.01204715227604046988e-2,
    3.73774314173823228969e-2,
    4.88280347570998239232e-2,
    7.03124996963957469739e-2,
    1.24999999999870820058e-1,
    4.99999999999999999821e-1,
)

_C1 = 1.3862943611198906188e0  # log(4)


@nb.njit(cache=True)
def eval_polynomial(x: float, coefs: npt.NDArray) -> float:
    """
    Evaluate a polynomial via Horner's method.

    Returns
    -------
    :
        The evaluation of the polynomial.
    """
    result = 0.0
    for c in coefs:
        result = result * x + c
    return result


@nb.njit([nb.float64(nb.float64)], cache=True)
def _ellipk(m: float) -> float:
    """
    K(m) for m >= 0.

    Parameters
    ----------
    m:
        Positive scalar.

    Returns
    -------
    :
        Evaluation of K(m).
    """
    if np.isnan(m):
        return np.nan
    if m == 1.0:  # noqa: RUF069
        return np.inf
    if m > 1.0:
        return np.nan

    p = 1.0 - m
    return eval_polynomial(p, _P) - np.log(p) * eval_polynomial(p, _Q)


@nb.vectorize([nb.float64(nb.float64)], nopython=True, cache=True)
def ellipk_nb(m: _FloatOrArray) -> _FloatOrArray:
    """
    Complete elliptic integral of the first kind, K(m).

    Parameters
    ----------
    m:
        Parameter(s) of the elliptic integral. Values m > 1 return NaN. Can be a float or
        an NDArray. If an array, the function is executed elementwise.

    Returns
    -------
    :
        K(m).

    Notes
    -----
    .. math::
        K(m) = \\int_{0}^{\\tfrac{\\pi}{2}}{(1 - m \\sin^2(t)})^{-\\tfrac{1}{2}} dt

    [ellipk_1]_

    Implementation derived from the Cephes [ellipk_2]_ C library (MIT licensed; also used
    by SciPy) - with help from Claude to translate the C into Python.

    .. [ellipk_1] Abramowitz, M., and I. A. Stegun. Handbook of Mathematical Functions.
                  Dover Publications, 1965.
    .. [ellipk_2] Moshier, S. L. (2000). Cephes Math Library Release 2.8.
                  http://www.netlib.org/cephes
    """
    if m < 0.0:
        # Reflection identity: K(m) = K(m/(m-1)) / sqrt(1-m)
        return _ellipk(m / (m - 1.0)) / np.sqrt(1.0 - m)
    return _ellipk(m)

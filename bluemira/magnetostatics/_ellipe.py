# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numba as nb
import numpy as np

from bluemira.magnetostatics._ellipk import _FloatOrArray, eval_polynomial

_P = (
    1.53552577301013293365e-4,
    2.50888492163602239211e-3,
    8.68786816565889628429e-3,
    1.07350949056076193403e-2,
    7.77395492516787092951e-3,
    7.58395289413514708519e-3,
    1.15688436810574127319e-2,
    2.18317996015557253103e-2,
    5.68051945617860553470e-2,
    4.43147180560990850618e-1,
    1.00000000000000000299e0,
)

_Q = (
    3.27954898576485872656e-5,
    1.00962792679356715133e-3,
    6.50609489976927491433e-3,
    1.68862163993311317300e-2,
    2.61769742454493659583e-2,
    3.34833904888224918614e-2,
    4.27180926518931511717e-2,
    5.85936634471101055642e-2,
    9.37499997205916132169e-2,
    2.49999999999888314361e-1,
)


@nb.njit([nb.float64(nb.float64)], cache=True)
def _ellipe(m: float) -> float:
    """
    E(m) for m >= 0.

    Parameters
    ----------
    m:
        Positive scalar.

    Returns
    -------
    :
        E(m).
    """
    if np.isnan(m):
        return np.nan
    if m == 0.0:  # noqa: RUF069
        return np.pi / 2.0
    if m == 1.0:  # noqa: RUF069
        return 1.0
    if m > 1.0:
        return np.nan

    x = 1.0 - m
    return eval_polynomial(x, _P) - np.log(x) * (x * eval_polynomial(x, _Q))


@nb.vectorize([nb.float64(nb.float64)], nopython=True, cache=True)
def ellipe_nb(m: _FloatOrArray) -> _FloatOrArray:
    """
    Complete elliptic integral of the second kind, E(m).

    Parameters
    ----------
    m:
        Parameter(s) of the elliptic integral. Values m > 1 return NaN. Can be a float
        or an NDArray. If an array, the function is executed elementwise.

    Returns
    -------
    :
        E(m)

    Notes
    -----
    .. math::
        E(m) = \\int_{0}^{\\tfrac{\\pi}{2}}{(1 - m \\sin^2(t)})^{\\tfrac{1}{2}} dt

    [ellipe_1]_

    Implementation derived from the Cephes [ellipe_2]_ C library (MIT licensed; also used
    by SciPy) - with help from Claude to translate the C into Python.

    .. [ellipe_1] Abramowitz, M., and I. A. Stegun. Handbook of Mathematical Functions.
                  Dover Publications, 1965.
    .. [ellipe_2] Moshier, S. L. (2000). Cephes Math Library Release 2.8.
                  http://www.netlib.org/cephes
    """
    if m < 0.0:
        # Reflection identity: E(m) = sqrt(1-m) * E(m/(m-1))
        return np.sqrt(1.0 - m) * _ellipe(m / (m - 1.0))
    return _ellipe(m)

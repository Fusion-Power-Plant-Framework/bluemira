# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

"""
Just-in-time compilation utilities and LowLevelCallable speed-up utilities.
"""
import numba as nb
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable
from scipy.integrate import quad, nquad


__all__ = ["jit_llc3", "jit_llc4", "jit_llc5", "jit_llc7", "integrate", "n_integrate"]


# Integration decorators and utilities
# Significant speed gains by using LowLevelCallable (> x2)


def jit_llc7(f_integrand):
    """
    Decorator for 6-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand: callable
        The integrand function

    Returns
    -------
    low_level: LowLevelCallable
        The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: U100
        return f_jitted(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6])

    return LowLevelCallable(wrapped.ctypes)


def jit_llc5(f_integrand):
    """
    Decorator for 4-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand: callable
        The integrand function

    Returns
    -------
    low_level: LowLevelCallable
        The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: U100
        return f_jitted(xx[0], xx[1], xx[2], xx[3], xx[4])

    return LowLevelCallable(wrapped.ctypes)


def jit_llc4(f_integrand):
    """
    Decorator for 3-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand: callable
        The integrand function

    Returns
    -------
    low_level: LowLevelCallable
        The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: U100
        return f_jitted(xx[0], xx[1], xx[2], xx[3])

    return LowLevelCallable(wrapped.ctypes)


def jit_llc3(f_integrand):
    """
    Decorator for 2-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand: callable
        The integrand function

    Returns
    -------
    low_level: LowLevelCallable
        The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: U100
        return f_jitted(xx[0], xx[1], xx[2])

    return LowLevelCallable(wrapped.ctypes)


def integrate(func, args, bound1, bound2):
    """
    Utility for integration of a function between bounds. Easier to refactor
    integration methods.

    Parameters
    ----------
    func: callable
        The function to integrate. The integration variable should be the last
        argument of this function.
    args: Iterable
        The iterable of static arguments to the function.
    bound1: Union[float, int]
        The lower integration bound
    bound2: Union[float, int]
        The upper integration bound

    Returns
    -------
    value: float
        The value of the integral of the function between the bounds
    """
    return quad(func, bound1, bound2, args=args)[0]


def n_integrate(func, args, bounds):
    """
    Utility for n-dimensional integration of a function between bounds. Easier
    to refactor integration methods.

    Parameters
    ----------
    func: callable
        The function to integrate. The integration variable should be the last
        argument of this function.
    args: Iterable
        The iterable of static arguments to the function.
    bounds: List[Iterable[Union[int, float]]]
        The list of lower and upper integration bounds applied to x[0], x[1], ..

    Returns
    -------
    value: float
        The value of the integral of the function between the bounds
    """
    return nquad(func, bounds, args=args)[0]

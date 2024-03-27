# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Just-in-time compilation and LowLevelCallable speed-up tools.
"""

import warnings
from collections.abc import Callable, Iterable

import numba as nb
import numpy as np
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable
from scipy.integrate import IntegrationWarning, nquad, quad

from bluemira.geometry.coordinates import Coordinates
from bluemira.magnetostatics.error import (
    MagnetostaticsError,
    MagnetostaticsIntegrationError,
)

__all__ = [
    "integrate",
    "jit_llc3",
    "jit_llc4",
    "jit_llc5",
    "jit_llc7",
    "n_integrate",
    "process_coords_array",
    "process_xyz_array",
]


# Integration decorators and utilities
# Significant speed gains by using LowLevelCallable (> x2)


def process_xyz_array(func):
    """
    Decorator for coordinate input handling in array-return functions and methods.
    """

    def wrapper(cls, x, y, z):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        if not len(x) == len(y) == len(z):
            raise MagnetostaticsError("Coordinate vector lengths must be equal.")

        if len(x) == 1:
            # Float handling
            return func(cls, x[0], y[0], z[0])
        if len(x.shape) == 1:
            # 1-D array handling
            return np.array([
                func(cls, xi, yi, zi) for xi, yi, zi in zip(x, y, z, strict=False)
            ]).T
        if len(x.shape) == 2:  # noqa: PLR2004
            # 2-D array handling
            m, n = x.shape
            result = np.zeros((3, m, n))
            for i in range(m):
                for j in range(n):
                    result[:, i, j] = np.array([func(cls, x[i, j], y[i, j], z[i, j])])
            return result

        raise MagnetostaticsError(
            "This operation only supports floats and 1-D and 2-D arrays."
        )

    return wrapper


def process_coords_array(shape: np.ndarray | Coordinates) -> np.ndarray:
    """
    Parse Coordinates or array to an array.

    Parameters
    ----------
    shape:
        The Coordinates or array to make into a coordinate array

    Returns
    -------
    Array in proper dimensions
    """
    if isinstance(shape, np.ndarray):
        pass

    elif isinstance(shape, Coordinates):
        shape = shape.T

    else:
        raise MagnetostaticsError(
            f"Cannot make a CurrentSource from an object of type: {type(shape)}."
        )

    return shape


def process_to_coordinates(shape: np.ndarray | dict | Coordinates) -> Coordinates:
    """
    Parse input to Coordinates

    Raises
    ------
    CoordinatesError: if the type could not be parsed, or if the input was bad.
    """
    if isinstance(shape, Coordinates):
        return shape
    return Coordinates(shape)


def jit_llc7(f_integrand: Callable) -> LowLevelCallable:
    """
    Decorator for 6-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand:
        The integrand function

    Returns
    -------
    The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: ARG001
        return f_jitted(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6])

    return LowLevelCallable(wrapped.ctypes)


def jit_llc5(f_integrand: Callable) -> LowLevelCallable:
    """
    Decorator for 4-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand:
        The integrand function

    Returns
    -------
    The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: ARG001
        return f_jitted(xx[0], xx[1], xx[2], xx[3], xx[4])

    return LowLevelCallable(wrapped.ctypes)


def jit_llc4(f_integrand: Callable) -> LowLevelCallable:
    """
    Decorator for 3-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand:
        The integrand function

    Returns
    -------
    The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: ARG001
        return f_jitted(xx[0], xx[1], xx[2], xx[3])

    return LowLevelCallable(wrapped.ctypes)


def jit_llc3(f_integrand: Callable) -> LowLevelCallable:
    """
    Decorator for 2-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand:
        The integrand function

    Returns
    -------
    The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: ARG001
        return f_jitted(xx[0], xx[1], xx[2])

    return LowLevelCallable(wrapped.ctypes)


def integrate(func: Callable, args: Iterable, bound1: float, bound2: float) -> float:
    """
    Utility for integration of a function between bounds. Easier to refactor
    integration methods.

    Parameters
    ----------
    func:
        The function to integrate. The integration variable should be the last
        argument of this function.
    args:
        The iterable of static arguments to the function.
    bound1:
        The lower integration bound
    bound2:
        The upper integration bound

    Returns
    -------
    The value of the integral of the function between the bounds
    """
    warnings.filterwarnings("error", category=IntegrationWarning)
    try:
        result = quad(func, bound1, bound2, args=args)[0]
    except IntegrationWarning:
        # First attempt at fixing the integration problem
        points = [
            0.25 * (bound2 - bound1),
            0.5 * (bound2 - bound1),
            0.75 * (bound2 - bound1),
        ]
        try:
            result = quad(func, bound1, bound2, args=args, points=points, limit=200)[0]
        except IntegrationWarning as error:
            raise MagnetostaticsIntegrationError from error

    warnings.filterwarnings("default", category=IntegrationWarning)
    return result


def n_integrate(
    func: Callable, args: Iterable, bounds: list[Iterable[int | float]]
) -> float:
    """
    Utility for n-dimensional integration of a function between bounds. Easier
    to refactor integration methods.

    Parameters
    ----------
    func:
        The function to integrate. The integration variable should be the last
        argument of this function.
    args:
        The iterable of static arguments to the function.
    bounds:
        The list of lower and upper integration bounds applied to x[0], x[1], ..

    Returns
    -------
    The value of the integral of the function between the bounds
    """
    return nquad(func, bounds, args=args)[0]


def reduce_coordinates(coordinates: np.ndarray) -> Coordinates:
    """
    Function to reduce the number of discretised points for a shape
    for use in magnetostatics calculations where accuracy is dependent
    upon shape not discretisation.

    Parameters
    ----------
    points:
        The coordinates of the shape to reduce

    Returns
    -------
        A reduced array of points
    """
    points = coordinates.T
    p0 = points[:-2]
    p1 = points[1:-1]
    p2 = points[2:]
    l1 = (p1 - p0) / np.linalg.norm((p1 - p0), axis=1)[:, None]
    l2 = (p2 - p1) / np.linalg.norm((p2 - p1), axis=1)[:, None]
    mask = np.ones_like(points[:, 0], dtype=bool)
    diag_dot = np.hstack([0, np.einsum("ij, ij -> i", l1, l2), 0])
    mask[np.isclose(diag_dot, 1)] = False
    return Coordinates(points[mask, :].T)

# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Collection of utility functions for the module."""

from collections.abc import Callable, Iterable
from typing import Any, NoReturn

import numpy as np
from scipy.optimize._numdiff import (
    approx_derivative as _approx_derivative,
)
from scipy.optimize._optimize import OptimizeResult

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation.error import OptimisationError

_FloatOrArray = float | np.ndarray

NO_CODE = "Unknown termination code"


def approx_derivative(
    func: Callable[[np.ndarray], _FloatOrArray],
    x0: np.ndarray,
    method: str = "3-point",
    rel_step: _FloatOrArray | None = None,
    f0: _FloatOrArray | None = None,
    bounds: Iterable[_FloatOrArray] | None = (-np.inf, np.inf),
    args: tuple[Any, ...] | None = (),
) -> np.ndarray:
    """
    Approximate the gradient of a function about a point.

    Parameters
    ----------
    func:
        Function for which to calculate the gradient.
    x0:
        Point about which to calculate the gradient.
    method:
        Finite difference method to use.
    rel_step:
        Relative step size to use.
    f0:
        Result of func(x0). If None, this is recomputed.
    bounds:
        Lower and upper bounds on individual variables.
    args:
        Additional positional arguments to ``func``.

    Returns
    -------
    :
        approximate gradient of the function about a point.
    """
    return _approx_derivative(
        func, x0, method=method, rel_step=rel_step, f0=f0, bounds=bounds, args=args
    )


def _log_and_raise(alg: str, msg: str) -> NoReturn:
    bluemira_warn(f"{alg} failed: {msg}")
    raise OptimisationError(f"{alg} optimisation failed: {msg}")


def _process_slsqp(res: OptimizeResult) -> np.ndarray:
    codes = {
        -1: "Gradient evaluation required (g & a)",
        1: "Function evaluation required (f & c)",
        2: "More equality constraints than independent variables",
        3: "More than 3*n iterations in LSQ subproblem",
        4: "Inequality constraints incompatible",
        5: "Singular matrix E in LSQ subproblem",
        6: "Singular matrix C in LSQ subproblem",
        7: "Rank-deficient equality constraint subproblem HFTI",
        8: "Positive directional derivative for linesearch",
        9: "Iteration limit reached",
    }
    if res.status == 0:
        return res.x
    if res.status == 9:  # noqa: PLR2004
        bluemira_warn(f"{codes.get(res.status)}, returning inoptimal result.")
        return res.x
    _log_and_raise(
        "SLSQP",
        codes.get(res.status, NO_CODE) + f"\n\n{res.message}{res!s}",
    )


def _process_cobyla(res: OptimizeResult) -> np.ndarray:
    codes = {
        2: "Maximum number of function evaluations reached.",
        3: "Trust-region subproblem failed.",
        20: "Maximum number of trust-region iterations reached.",
        -1: "NaN/Inf encountered in x.",
        -2: "NaN/Inf encountered in f.",
        -3: "NaN/Inf encountered in model.",
        6: "No space between variable bounds.",
        7: "Numerical instability (damaging rounding).",
        8: "Zero linear constraint detected.",
        30: "Terminated by user callback.",
        100: "COBYLA internal error (invalid input).",
        101: "COBYLA internal error (assertion fails).",
        102: "COBYLA internal error (validation fails).",
        103: "COBYLA internal error (memory allocation fails).",
    }
    if res.status in {0, 1}:
        return res.x
    if res.status in {2, 20}:
        bluemira_warn(f"{codes.get(res.status)}, returning inoptimal result.")
        return res.x
    _log_and_raise(
        "COBYLA",
        codes.get(res.status, NO_CODE) + f"\n\n{res.message}{res!s}",
    )


def _process_cobyqa(res: OptimizeResult) -> np.ndarray:
    codes = {
        5: "Maximum function evaluations reached.",
        6: "Maximum iteration count reached.",
        -1: "Infeasible starting point or problem.",
        -2: "Linear algebra failure inside COBYQA.",
    }
    if res.status in {0, 1, 2, 3, 4}:
        return res.x
    if res.status in {5, 6}:
        bluemira_warn(f"{codes.get(res.status)}, returning inoptimal result.")
        return res.x
    _log_and_raise(
        "COBYQA",
        codes.get(res.status, NO_CODE) + f"\n\n{res.message}{res!s}",
    )


def process_scipy_result(res: OptimizeResult, alg: str) -> np.ndarray:
    """
    Handle a scipy.minimize OptimizeResult object. Process error codes, if any.

    Parameters
    ----------
    res:
        Scipy optimise result

    Returns
    -------
    x: np.array
        The optimal set of parameters (result of the optimisation)

    Raises
    ------
    OptimisationError
        if an error code returned without a usable result.
    """
    if res.success:
        return res.x

    if not hasattr(res, "status"):
        bluemira_warn("Scipy optimisation was not succesful. Failed without status.")
        raise OptimisationError(f"{res.message}\n{res!s}")

    # scipy uses different result status codes depending on the algorithm *sigh*
    if alg == "SLSQP":
        return _process_slsqp(res)
    if alg == "COBYLA":
        return _process_cobyla(res)
    if alg == "COBYQA":
        return _process_cobyqa(res)

    raise OptimisationError(f"{res.message}\n{res!s}")


def _check_bounds(n_dims: int, new_bounds: np.ndarray) -> None:
    """Validate that the bounds have the correct dimensions.

    Raises
    ------
    ValueError
        New bounds in not 1D and does not have a size of n_dims
    """
    if new_bounds.ndim != 1 or new_bounds.size != n_dims:
        raise ValueError(
            f"Cannot set bounds with shape '{new_bounds.shape}', "
            f"array must be one dimensional and have '{n_dims}' elements."
        )


def _initial_guess_from_bounds(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Derive an initial guess for the optimiser.

    Takes the center of the bounds for each parameter.

    Returns
    -------
    :
        Initial guess based on the midpoint of the provided bounds.
    """
    bounds = np.array([lower, upper])
    # bounds are +/- inf by default, change to real numbers so
    # we can take an average
    np.nan_to_num(
        bounds,
        posinf=np.finfo(np.float64).max,
        neginf=np.finfo(np.float64).min,
        copy=False,
    )
    return np.mean(bounds, axis=0)

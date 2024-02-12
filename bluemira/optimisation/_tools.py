# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Collection of utility functions for the module."""

from typing import Any, Callable, Iterable, Optional, Tuple, Union

import numpy as np
from scipy.optimize._numdiff import (
    approx_derivative as _approx_derivative,  # noqa: PLC2701
)

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation.error import OptimisationError

_FloatOrArray = Union[float, np.ndarray]


def approx_derivative(
    func: Callable[[np.ndarray], _FloatOrArray],
    x0: np.ndarray,
    method: str = "3-point",
    rel_step: Optional[_FloatOrArray] = None,
    f0: Optional[_FloatOrArray] = None,
    bounds: Optional[Iterable[_FloatOrArray]] = (-np.inf, np.inf),
    args: Optional[Tuple[Any, ...]] = (),
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
    """
    return _approx_derivative(
        func, x0, method=method, rel_step=rel_step, f0=f0, bounds=bounds, args=args
    )


def process_scipy_result(res):
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
    InternalOptError if an error code returned without a usable result.
    """
    if res.success:
        return res.x

    if not hasattr(res, "status"):
        bluemira_warn("Scipy optimisation was not succesful. Failed without status.")
        raise OptimisationError(f"{res.message}\n{res!s}")

    if res.status == 8:  # noqa: PLR2004
        # This can happen when scipy is not convinced that it has found a minimum.
        bluemira_warn(
            "\nOptimiser (scipy) found a positive directional derivative,\n"
            f"returning suboptimal result. \n\n{res.message}{res!s}"
        )
        return res.x

    if res.status == 9:  # noqa: PLR2004
        bluemira_warn(
            "\nOptimiser (scipy) exceeded number of iterations, returning"
            f"suboptimal result. \n\n{res.message}{res!s}"
        )
        return res.x

    raise OptimisationError(f"{res.message}\n{res!s}")

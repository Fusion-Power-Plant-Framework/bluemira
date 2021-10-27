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
Optimisation utilities
"""

import numpy as np
import nlopt
from bluemira.utilities.error import InternalOptError
from bluemira.base.look_and_feel import bluemira_warn


# =============================================================================
# Analytical objective functions
# =============================================================================


def tikhonov(A, b, gamma):
    """
    Tikhonov regularisation of Ax-b problem.

    \t:math:`\\textrm{minimise} || Ax - b ||^2 + ||{\\gamma} \\cdot x ||^2`\n
    \t:math:`x = (A^T A + {\\gamma}^2 I)^{-1}A^T b`

    Parameters
    ----------
    A: np.array(n, m)
        The 2-D A matrix of responses
    b: np.array(n)
        The 1-D b vector of values
    gamma: float
        The Tikhonov regularisation parameter

    Returns
    -------
    x: np.array(m)
        The result vector
    """
    try:
        return np.dot(
            np.linalg.inv(np.dot(A.T, A) + gamma ** 2 * np.eye(A.shape[1])),
            np.dot(A.T, b),
        )
    except np.linalg.LinAlgError:
        bluemira_warn("Tikhonov singular matrix..!")
        return np.dot(
            np.linalg.pinv(np.dot(A.T, A) + gamma ** 2 * np.eye(A.shape[1])),
            np.dot(A.T, b),
        )


def least_squares(A, b):
    """
    Least squares optimisation.

    \t:math:`\\textrm{minimise} || Ax - b ||^{2}_{2}`\n
    \t:math:`x = (A^T A)^{-1}A^T b`

    Parameters
    ----------
    A: np.array(n, m)
        The 2-D A matrix of responses
    b: np.array(n)
        The 1-D b vector of values

    Returns
    -------
    x: np.array(m)
        The result vector
    """
    return np.linalg.solve(A, b)


# =============================================================================
# Scipy interface
# =============================================================================


def process_scipy_result(res):
    """
    Handle a scipy.minimize OptimizeResult object. Process error codes, if any.

    Parameters
    ----------
    res: OptimizeResult

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
        raise InternalOptError("\n".join([res.message, res.__str__()]))

    elif res.status == 8:
        # This can happen when scipy is not convinced that it has found a minimum.
        bluemira_warn(
            "\nOptimiser (scipy) found a positive directional derivative,\n"
            "returning suboptimal result. \n"
            "\n".join([res.message, res.__str__()])
        )
        return res.x

    elif res.status == 9:
        bluemira_warn(
            "\nOptimiser (scipy) exceeded number of iterations, returning "
            "suboptimal result. \n"
            "\n".join([res.message, res.__str__()])
        )
        return res.x

    else:
        raise InternalOptError("\n".join([res.message, res.__str__()]))


# =============================================================================
# NLOpt interface
# =============================================================================


def process_NLOPT_result(opt):  # noqa (N802)
    """
    Handle a NLOPT optimiser and check results.

    Parameters
    ----------
    opt: NLOPT Optimize object
        The optimiser to check
    """
    # TODO: Check constraints
    result = opt.last_optimize_result()
    if result < 0:
        bluemira_warn(f"\nNLOPT Optimiser failed with internal error code: {result}")

    if result == nlopt.MAXEVAL_REACHED:
        bluemira_warn(
            "\nNlOPT Optimiser succeeded but stopped at the maximum number of evaulations\n"
        )

    elif result == nlopt.MAXTIME_REACHED:
        bluemira_warn(
            "\nNLOPT Optimiser succeeded but stopped at the maximum duration\n"
        )


def approx_fprime(xk, func, epsilon, bounds, *args, f0=None):
    """
    An altered version of a scipy function, but with the added feature
    of clipping the perturbed variables to be within their prescribed bounds

    Parameters
    ----------
    xk: array_like
        The state vector at which to compute the Jacobian matrix.
    func: callable f(x,*args)
        The vector-valued function.
    epsilon: float
        The perturbation used to determine the partial derivatives.
    bounds: array_like(len(xk), 2)
        The bounds the variables to respect
    args: sequence
        Additional arguments passed to func.
    f0: Union[float, None]
        The initial value of the function at x=xk. If None, will be calculated

    Returns
    -------
    grad: array_like(len(func), len(xk))
        The gradient of the func w.r.t to the perturbed variables

    Notes
    -----
    The approximation is done using forward differences.
    """
    if f0 is None:
        f0 = func(*((xk,) + args))

    grad = np.zeros((len(xk),), float)
    ei = np.zeros((len(xk),), float)
    for i in range(len(xk)):
        ei[i] = 1.0
        # The delta value to add the the variable vector
        d = epsilon * ei
        # Clip the perturbed variable vector with the variable bounds
        xk_d = np.clip(xk + d, bounds[0], bounds[1])

        # Get the clipped length of the perturbation
        delta = xk_d[i] - xk[i]

        if delta == 0:
            df = 0
        else:
            df = (func(*((xk_d,) + args)) - f0) / delta

        if not np.isscalar(df):
            try:
                df = df.item()
            except (ValueError, AttributeError):
                raise InternalOptError(
                    "The user-provided objective function must return a scalar value."
                )
        grad[i] = df
        ei[i] = 0.0
    return grad


def approx_jacobian(x, func, epsilon, bounds, *args, f0=None):
    """
    An altered version of a scipy function, but with the added feature
    of clipping the perturbed variables to be within their prescribed bounds

    Approximate the Jacobian matrix of a callable function.

    Parameters
    ----------
    x : array_like
        The state vector at which to compute the Jacobian matrix.
    func : callable f(x,*args)
        The vector-valued function.
    epsilon : float
        The perturbation used to determine the partial derivatives.
    bounds: array_like(len(xk), 2)
        The bounds the variables to respect.
    args : sequence
        Additional arguments passed to func.
    f0: Union[np.array, None]
        The initial value of the function at x=xk. If None, will be calculated

    Returns
    -------
    An array of dimensions ``(lenf, lenx)`` where ``lenf`` is the length
    of the outputs of `func`, and ``lenx`` is the number of elements in
    `x`.

    Notes
    -----
    The approximation is done using forward differences.
    """
    x0 = np.asfarray(x)

    if f0 is None:
        f0 = np.atleast_1d(func(*((x0,) + args)))

    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = 1.0
        # The delta value to add the the variable vector
        d = epsilon * dx
        # Clip the perturbed variable vector with the variable bounds
        x_dx = np.clip(x + d, bounds[0], bounds[1])

        # Get the clipped length of the perturbation
        delta = x_dx[i] - x[i]

        if delta == 0:
            jac[i] = 0
        else:
            jac[i] = (func(*((x_dx,) + args)) - f0) / delta

        dx[i] = 0.0

    return jac.transpose()

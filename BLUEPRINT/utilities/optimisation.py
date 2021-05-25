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
Optimisation utilities
"""
import numpy as np
from numpy import dot, eye
from numpy.linalg import inv, pinv, LinAlgError
from scipy.optimize._constraints import old_constraint_to_new
from BLUEPRINT.base.lookandfeel import bpwarn
from BLUEPRINT.geometry.constants import VERY_BIG
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.geomtools import distance_between_points, normal, get_intersect


class InternalOptError(Exception):
    """
    Error class for errors inside the optimisation algorithms.
    """

    pass


class ExternalOptError(Exception):
    """
    Error class for errors relating to the optimisation, but not originating
    inside the optimisers.
    """

    pass


class _NLOPTFunction:
    """
    Base class for an optimisation function where numerical estimations of the
    gradient or jacobian are required.

    Parameters
    ----------
    func: callable
        The function to calculate the objective or constraints
    bounds: np.array(n, 2)
        The array of lower and upper bounds
    """

    def __init__(self, func, bounds):
        self.func = func
        self.bounds = bounds


class NLOPTObjectiveFunction(_NLOPTFunction):
    """
    An objective function with numerical calculation of the gradient.
    """

    def __call__(self, x, grad, *args):
        """
        Calculate the objective functions and its gradient (numerically).

        Parameters
        ----------
        x: np.array
            The optimisation variable vector
        grad: np.array
            The array of the gradient in NLopt
        args: tuple
            The additional arguments used in the function evaluation

        Returns
        -------
        result: float
            The value of the objective function

        Notes
        -----
        Modifies `grad` in-place as per NLopt usage.
        """
        result = self.func(x, *args)

        if grad.size > 0:
            grad[:] = approx_fprime(x, self.func, 1e-6, self.bounds, *args, f0=result)

        return result


class NLOPTConstraintFunction(_NLOPTFunction):
    """
    A constraint function with numerical calculation of the Jacobian.
    """

    def __call__(self, constraint, x, grad, *args):
        """
        Calculate the objective functions and its gradient (numerically).

        Parameters
        ----------
        constraint: np.array
            The array of the constraint equations
        x: np.array
            The optimisation variable vector
        grad: np.array
            The array of the gradient in NLopt
        args: tuple
            The additional arguments used in the function evaluation

        Returns
        -------
        constraint: np.array
            The array of the constraint equations

        Notes
        -----
        Modifies `grad` and `constraint` in-place as per NLopt usage.
        """
        constraint[:] = self.func(x, *args)

        if grad.size > 0:
            grad[:] = approx_jacobian(
                x, self.func, 1e-6, self.bounds, *args, f0=constraint
            )


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
        return dot(inv(dot(A.T, A) + gamma ** 2 * eye(A.shape[1])), dot(A.T, b))
    except LinAlgError:
        bpwarn("utilities/optimisation.py: Tikhonov singular matrix..!")
        return dot(pinv(dot(A.T, A) + gamma ** 2 * eye(A.shape[1])), dot(A.T, b))


def leastsquares(A, b):
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
        bpwarn("Scipy optimisation was not succesful. Failed without status.")
        raise InternalOptError("\n".join([res.message, res.__str__()]))

    elif res.status == 8:
        # This can happen when scipy is not convinced that it has found a minimum.
        bpwarn(
            "\nOptimiser (scipy) found a positive directional derivative,\n"
            "returning suboptimal result. \n"
            "\n".join([res.message, res.__str__()])
        )
        return res.x

    elif res.status == 9:
        bpwarn(
            "\nOptimiser (scipy) exceeded number of iterations, returning "
            "suboptimal result. \n"
            "\n".join([res.message, res.__str__()])
        )
        return res.x

    else:
        raise InternalOptError("\n".join([res.message, res.__str__()]))


def process_NLOPT_result(opt):  # noqa (N802)
    """
    Handle a NLOPT optimiser and check results.

    Parameters
    ----------
    opt: NLOPT Optimize object
        The optimiser to check
    """
    result = opt.last_optimize_result()
    if result < 0:
        bpwarn(
            "\nNLOPT Optimiser failed with internal error code (see below)"
            "\nreturning last optimum value\n"
            "\n".join([str(result)])
        )


def convert_scipy_constraints(list_of_con_dicts):
    """
    Converts a list of old-style scipy constraint dicts into NonLinearConstraints

    Parameters
    ----------
    list_of_con_dicts

    Returns
    -------
    constraints: List[NonLinearConstraint]
    """
    new_constraints = []
    for i, con in enumerate(list_of_con_dicts):
        new = old_constraint_to_new(i, con)
        new_constraints.append(new)
    return new_constraints


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


def geometric_constraint(bound, loop, con_type="external"):
    """
    Geometric constraint function in 2-D.

    Parameters
    ----------
    bound: Loop
        The bounding loop constraint
    loop: Loop
        The shape being optimised
    con_type: str
        The type of constraint to apply ["internal", "external"]

    Returns
    -------
    constraint: np.array
        The geometric constraint array
    """

    def get_min_distance(point, vector_line):
        x_inter, z_inter = get_intersect(loop, vector_line)
        distances = []
        for xi, zi in zip(x_inter, z_inter):
            distances.append(distance_between_points(point, [xi, zi]))

        return np.min(distances)

    normals = normal(*bound.d2)
    constraint = np.zeros(len(bound))
    for i, b_point in enumerate(bound.d2.T):
        n_hat = np.array([normals[0][i], normals[1][i]])

        n_hat = VERY_BIG * n_hat
        x_con, z_con = b_point

        line = Loop(
            x=[x_con - n_hat[0], x_con + n_hat[0]],
            z=[z_con - n_hat[1], z_con + n_hat[1]],
        )
        distance = get_min_distance(b_point, line)
        constraint[i] = distance

    return constraint


def dot_difference(bound, loop, side="internal"):
    """
    Utility function for geometric constraints.
    """
    xloop, zloop = loop.d2
    switch = 1 if side == "internal" else -1
    n_xloop, n_zloop = normal(xloop, zloop)
    x_bound, z_bound = bound.d2
    dotp = np.zeros(len(x_bound))
    for j, (x, z) in enumerate(zip(x_bound, z_bound)):
        i = np.argmin((x - xloop) ** 2 + (z - zloop) ** 2)
        dl = [xloop[i] - x, zloop[i] - z]
        dn = [n_xloop[i], n_zloop[i]]
        dotp[j] = switch * np.dot(dl, dn)
    return dotp

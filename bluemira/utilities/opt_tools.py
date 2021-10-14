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

import abc
from copy import deepcopy
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

        if np.isclose(delta, 0.0):
            # Re-bound the bound in the other direction
            xk_d[i] = xk[i] - d[i]
            delta = xk[i] - xk_d[i]

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

        if np.isclose(delta, 0.0):
            jac[i] = 0.0
        else:
            jac[i] = (func(*((x_dx,) + args)) - f0) / delta

        dx[i] = 0.0

    return jac.transpose()


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

    def __init__(self, func, bounds, epsilon=1e-6):
        self.func = func
        self.bounds = bounds
        self.epsilon = epsilon


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
            grad[:] = approx_fprime(
                x, self.func, self.epsilon, self.bounds, *args, f0=result
            )

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
                x, self.func, self.epsilon, self.bounds, *args, f0=constraint
            )


# =============================================================================
# Generic optimisation interface
# =============================================================================


class Optimiser(abc.ABC):
    """
    Optimiser ABC for interfacing with different optimisation library APIs.

    Parameters
    ----------
    algorithm: Any
        Optimisation algorithm to use
    n_variables: int
        Size of the variable vector
    """

    def __init__(self, algorithm, n_variables, opt_parameters={}, opt_conditions={}):
        self.n_variables = n_variables
        self.set_algorithm(algorithm)
        self.x0 = np.zeros(n_variables)
        self.lower_bounds = np.zeros(n_variables)
        self.upper_bounds = np.ones(n_variables)
        self.constraints = []
        self.constraint_tols = []
        self.set_algorithm_parameters(opt_parameters)
        self.set_termination_conditions(opt_conditions)

    def set_initial_value(self, x0):
        """
        Set the optimiser initial solution.

        Parameters
        ----------
        x0: np.ndarray
            Initial solution vector
        """
        if len(x0) != self.n_variables:
            raise InternalOptError(
                "Initial solution must have the same dimension as the optimiser."
            )
        self.x0 = x0

    @abc.abstractmethod
    def set_algorithm(self, algorithm):
        """
        Set the optimisation algorithm to use.

        Parameters
        ----------
        algorithm: Any
            Optimisation algorithm to use
        """
        pass

    @abc.abstractmethod
    def set_algorithm_parameters(self, opt_parameters):
        """
        Set the optimisation algorithm parameters to use.

        Parameters
        ----------
        opt_parameters: dict
            Optimisation algorithm parameters to use
        """
        pass

    @abc.abstractmethod
    def set_termination_conditions(self, opt_conditions):
        """
        Set the optimisation algorithm termination condition(s) to use.

        Parameters
        ----------
        opt_conditions: dict
            Termination conditions for the optimisation algorithm
        """
        pass

    @abc.abstractmethod
    def set_objective_function(self, f_objective):
        """
        Set the objective function (minimisation).

        Parameters
        ----------
        f_objective: callable
            Objective function to minimise
        """
        pass

    @abc.abstractmethod
    def add_ineq_constraint(self, f_constraint, tolerance):
        """
        Add a single-valued inequality constraint.

        Parameters
        ----------
        f_constraint: callable
            Constraint function
        tolerance: float
            Tolerance with which to enforce the constraint
        """
        pass

    @abc.abstractmethod
    def add_ineq_constraints(self, f_constraint, tolerance):
        """
        Add a vector-valued inequality constraint.

        Parameters
        ----------
        f_constraint: callable
            Constraint function
        tolerance: np.ndarray
            Tolerance array with which to enforce the constraint
        """
        pass

    @abc.abstractmethod
    def add_eq_constraint(self, f_constraint, tolerance):
        """
        Add a single-valued equality constraint.

        Parameters
        ----------
        f_constraint: callable
            Constraint function
        tolerance: float
            Tolerance with which to enforce the constraint
        """
        pass

    @abc.abstractmethod
    def set_lower_bounds(self, lower_bounds):
        """
        Set the lower bounds.

        Parameters
        ----------
        lower_bounds: np.ndarray
            Lower bound vector
        """
        pass

    @abc.abstractmethod
    def set_upper_bounds(self, upper_bounds):
        """
        Set the upper bounds.

        Parameters
        ----------
        upper_bounds: np.ndarray
            Upper bound vector
        """
        pass

    @abc.abstractmethod
    def optimise(self):
        """
        Run the optimisation problem.

        Returns
        -------
        x_star: np.ndarray
            Optimal solution vector
        """
        pass

    def _append_constraint_tols(self, constraint, tolerance):
        """
        Append constraint function and tolerances.
        """
        self.constraints.append(constraint)
        self.constraint_tols.append(tolerance)

    @abc.abstractmethod
    def process_result(self):
        """
        Process the optimisation result.
        """
        pass

    @abc.abstractmethod
    def check_constraints(self):
        """
        Check that the constraints have been met.
        """
        pass

    def copy(self):
        """
        Get a deepcopy of the Optimiser.
        """
        return deepcopy(self)


class NLOptOptimiser(Optimiser):
    """
    NLOpt optimiser class.

    Parameters
    ----------
    algorithm: nlopt algorithm
        Optimisation algorithm to use
    n_variables: int
        Size of the variable vector
    """

    def __init__(self, algorithm, n_variables, opt_parameters={}, opt_conditions={}):
        super().__init__(algorithm, n_variables, opt_parameters, opt_conditions)

    def set_algorithm(self, algorithm):
        """
        Parameters
        ----------
        algorithm: nlopt algorithm
        Optimisation algorithm to use
        """
        self._opt = nlopt.opt(algorithm, self.n_variables)

    def set_algorithm_parameters(self, opt_parameters):
        """
        Set the optimisation algorithm parameters to use.

        Parameters
        ----------
        opt_parameters: dict
            Optimisation algorithm parameters to use
        """
        for k, v in opt_parameters.items():
            self._opt.set_param(k, v)

    def set_termination_conditions(self, opt_conditions):
        """
        Set the optimisation algorithm termination condition(s) to use.

        Parameters
        ----------
        opt_conditions: dict
            Termination conditions for the optimisation algorithm
        """
        if "ftol_abs" in opt_conditions:
            self._opt.set_ftol_abs(opt_conditions["ftol_abs"])
        if "ftol_rel" in opt_conditions:
            self._opt.set_ftol_rel(opt_conditions["ftol_rel"])
        if "xtol_abs" in opt_conditions:
            self._opt.set_xtol_abs(opt_conditions["xtol_abs"])
        if "xtol_rel" in opt_conditions:
            self._opt.set_xtol_rel(opt_conditions["xtol_rel"])
        if "max_time" in opt_conditions:
            self._opt.set_maxtime(opt_conditions["max_time"])
        if "max_eval" in opt_conditions:
            self._opt.set_maxeval(opt_conditions["max_eval"])
        if "stop_val" in opt_conditions:
            self._opt.set_stopval(opt_conditions["stop_val"])

    def set_objective_function(self, f_objective):
        """
        Set the objective function (minimisation).

        Parameters
        ----------
        f_objective: callable
            Objective function to minimise
        """
        if f_objective is None:
            return
        f_objective = NLOPTObjectiveFunction(
            f_objective, [self.lower_bounds, self.upper_bounds]
        )
        self._opt.set_min_objective(f_objective)

    def set_lower_bounds(self, lower_bounds):
        """
        Set the lower bounds.

        Parameters
        ----------
        lower_bounds: np.ndarray
            Lower bound vector
        """
        self.lower_bounds = lower_bounds
        self._opt.set_lower_bounds(lower_bounds)

    def set_upper_bounds(self, upper_bounds):
        """
        Set the upper bounds.

        Parameters
        ----------
        upper_bounds: np.ndarray
            Upper bound vector
        """
        self.upper_bounds = upper_bounds
        self._opt.set_upper_bounds(upper_bounds)

    def add_eq_constraint(self, f_constraint, tolerance):
        """
        Add a single-valued equality constraint.

        Parameters
        ----------
        f_constraint: callable
            Constraint function
        tolerance: float
            Tolerance with which to enforce the constraint
        """
        if f_constraint is None:
            return
        self._opt.add_equality_constraint(f_constraint, tolerance)
        self._append_constraint_tols(f_constraint, tolerance)

    def add_ineq_constraint(self, f_constraint, tolerance):
        """
        Add a single-valued inequality constraint.

        Parameters
        ----------
        f_constraint: callable
            Constraint function
        tolerance: float
            Tolerance with which to enforce the constraint
        """
        if f_constraint is None:
            return
        f_constraint = NLOPTConstraintFunction(
            f_constraint, [self.lower_bounds, self.upper_bounds]
        )
        self._opt.add_inequality_constraint(f_constraint, tolerance)
        self._append_constraint_tols(f_constraint, tolerance)

    def add_ineq_constraints(self, f_constraint, tolerance):
        """
        Add a vector-valued inequality constraint.

        Parameters
        ----------
        f_constraint: callable
            Constraint function
        tolerance: np.ndarray
            Tolerance array with which to enforce the constraint
        """
        if f_constraint is None:
            return
        f_constraint = NLOPTConstraintFunction(
            f_constraint, [self.lower_bounds, self.upper_bounds]
        )
        self._opt.add_inequality_mconstraint(f_constraint, tolerance)
        self._append_constraint_tols(f_constraint, tolerance)

    def optimise(self):
        """
        Run the optimisation problem.

        Returns
        -------
        x_star: np.ndarray
            Optimal solution vector
        """
        try:
            x_star = self._opt.optimize(self.x0)
        except nlopt.RoundoffLimited:
            self.rms = self._opt.last_optimum_value()

        self.rms = self._opt.last_optimum_value()
        self.process_result()
        self.check_constraints(x_star)
        return x_star

    def process_result(self):
        """
        Process the optimisation result.
        """
        process_NLOPT_result(self._opt)

    def check_constraints(self, x):
        """
        Check that the constraints have been met.
        """
        c_values = []
        tolerances = []
        for constraint, tolerance in zip(self.constraints, self.constraint_tols):
            c_values.extend(constraint.func(x))
            tolerances.extend(tolerance)

        c_values = np.array(c_values)
        tolerances = np.array(tolerances)

        if not np.all(c_values < tolerances):
            raise InternalOptError(
                "Some constraints have not been adequately satisfied."
            )

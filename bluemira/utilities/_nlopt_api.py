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
Thin wrapper API interface to optimisation library (NLOpt)
"""

import numpy as np
import inspect
import nlopt

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.error import OptUtilitiesError
from bluemira.utilities.opt_tools import approx_fprime, approx_jacobian


NLOPT_ALG_MAPPING = {
    "SLSQP": nlopt.LD_SLSQP,
    "COBYLA": nlopt.LN_COBYLA,
    "MMA": nlopt.LD_MMA,
    "BFGS": nlopt.LD_LBFGS,
    "DIRECT": nlopt.GN_DIRECT,
    "DIRECT-L": nlopt.GN_DIRECT_L,
    "CRS": nlopt.GN_CRS2_LM,
    "ISRES": nlopt.GN_ISRES,
}

GRAD_ALGS = ["SLSQP", "MMA", "BGFS"]

INEQ_CON_ALGS = ["SLSQP", "COBYLA", "ISRES"]

EQ_CON_ALGS = ["SLSQP", "COBYLA", "ISRES"]


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

    if result == nlopt.MAXEVAL_REACHED:
        bluemira_warn(
            "\nNlOPT Optimiser succeeded but stopped at the maximum number of evaulations.\n"
        )

    elif result == nlopt.MAXTIME_REACHED:
        bluemira_warn(
            "\nNLOPT Optimiser succeeded but stopped at the maximum duration.\n"
        )
    elif result == nlopt.ROUNDOFF_LIMITED:
        bluemira_warn(
            "\nNLOPT Optimiser was halted due to round-off errors. A useful result was probably found...\n"
        )
    elif result == nlopt.FAILURE:
        bluemira_warn(
            f"\nNLOPT Optimiser failed real hard.. internal error code: {nlopt.FAILURE}.\n"
        )
    elif result == nlopt.INVALID_ARGS:
        bluemira_warn(f"\nNLOPT Optimiser failed because of invalid arguments.\n")
    elif result == nlopt.OUT_OF_MEMORY:
        bluemira_warn(f"\nNLOPT Optimiser failed because it ran out of memory.\n")
    elif result == nlopt.FORCED_STOP:
        bluemira_warn(f"\nNLOPT Optimiser failed because of a forced stop.\n")


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


class NLOPTObjectiveFunction:
    def __init__(self, func):
        self.func = func
        self.last_x = None

    def __call__(self, x, grad, *args):
        self.last_x = x
        return self.func(x, grad, *args)


class NLOPTNumGradObjectiveFunction(_NLOPTFunction):
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
        self.last_x = x

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


class NLOPTOptimiser:
    """
    NLOpt optimiser API class.

    Parameters
    ----------
    algorithm_name: str
        Optimisation algorithm to use
    n_variables: int
        Size of the variable vector
    """

    def __init__(
        self, algorithm_name, n_variables, opt_parameters={}, opt_conditions={}
    ):
        self.n_variables = n_variables
        self.set_algorithm(algorithm_name)
        self.lower_bounds = np.zeros(n_variables)
        self.upper_bounds = np.ones(n_variables)
        self.constraints = []
        self.constraint_tols = []
        self.set_algorithm_parameters(opt_parameters)
        self.set_termination_conditions(opt_conditions)

    def _grad_alg_and_no_grad(self, func):
        return (
            self.algorithm_name in GRAD_ALGS
            and inspect.signature(func).parameters["grad"] is None
        )

    def _append_constraint_tols(self, constraint, tolerance):
        """
        Append constraint function and tolerances.
        """
        self.constraints.append(constraint)
        self.constraint_tols.append(tolerance)

    def set_algorithm(self, algorithm_name):
        """
        Parameters
        ----------
        algorithm: nlopt algorithm
        Optimisation algorithm to use
        """
        algorithm = NLOPT_ALG_MAPPING.get(algorithm_name, None)
        if algorithm is None:
            raise OptUtilitiesError(f"Unknown or unmapped algorithm: {algorithm_name}")

        self.algorithm_name = algorithm_name
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
        if self._grad_alg_and_no_grad(f_objective):
            # Gradient-based algorithm but grad is set to None: numerically calculate
            f_objective = NLOPTNumGradObjectiveFunction(
                f_objective, [self.lower_bounds, self.upper_bounds]
            )
        else:
            f_objective = NLOPTObjectiveFunction(f_objective)
        self._f_objective = f_objective
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
        if self.algorithm_name not in EQ_CON_ALGS:
            raise OptUtilitiesError(
                f"{self.algorithm_name} does not support equality constraints."
            )

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
        if self.algorithm_name not in INEQ_CON_ALGS:
            raise OptUtilitiesError(
                f"{self.algorithm_name} does not support inequality constraints."
            )
        if self._grad_alg_and_no_grad(f_constraint):
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
        if self.algorithm_name not in INEQ_CON_ALGS:
            raise OptUtilitiesError(
                f"{self.algorithm_name} does not support inequality constraints."
            )

        if self._grad_alg_and_no_grad(f_constraint):
            f_constraint = NLOPTConstraintFunction(
                f_constraint, [self.lower_bounds, self.upper_bounds]
            )

        self._opt.add_inequality_mconstraint(f_constraint, tolerance)
        self._append_constraint_tols(f_constraint, tolerance)

    def optimise(self, x0):
        """
        Run the optimiser.

        Returns
        -------
        x_star: np.ndarray
            Optimal solution vector
        """
        try:
            x_star = self._opt.optimize(x0)
        except nlopt.RoundoffLimited:
            # It's likely that the last call was still a reasonably good solution.
            self.rms = self._opt.last_optimum_value()
            x_star = self._f_objective.last_x
        self.rms = self._opt.last_optimum_value()
        process_NLOPT_result(self._opt)
        return x_star

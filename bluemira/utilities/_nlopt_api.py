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
import nlopt
import functools

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.error import (
    OptUtilitiesError,
    ExternalOptError,
    OptVariablesError,
)


EPS = np.finfo(np.longdouble).eps ** (1 / 3)

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
        bluemira_warn("\nNLOPT Optimiser failed real hard...\n")
    elif result == nlopt.INVALID_ARGS:
        bluemira_warn("\nNLOPT Optimiser failed because of invalid arguments.\n")
    elif result == nlopt.OUT_OF_MEMORY:
        bluemira_warn("\nNLOPT Optimiser failed because it ran out of memory.\n")
    elif result == nlopt.FORCED_STOP:
        bluemira_warn("\nNLOPT Optimiser failed because of a forced stop.\n")


class _NLOPTObjectiveFunction:
    """
    Wrapper to store x-vector in case of RoundOffLimited errors.
    """

    def __init__(self, func):
        self.func = func
        self.last_x = None

    def __call__(self, x, grad, *args):
        self.store_x(x)
        return self.func(x, grad, *args)

    def store_x(self, x):
        if not np.isnan(np.sum(x)):
            self.last_x = x


class NLOPTOptimiser:
    """
    NLOpt optimiser API class.

    Parameters
    ----------
    algorithm_name: str
        Optimisation algorithm to use
    n_variables: Optional[int]
        Size of the variable vector
    opt_conditions: dict
        Dictionary of algorithm termination criteria
    opt_parameters: dict
        Dictionary of algorithm parameters
    """

    def __init__(
        self,
        algorithm_name,
        n_variables=None,
        opt_conditions={},
        opt_parameters={},
    ):
        self.opt_conditions = opt_conditions
        self.opt_parameters = opt_parameters
        self.algorithm_name = algorithm_name
        self.n_variables = n_variables
        self.constraints = []
        self.constraint_tols = []

    def _opt_inputs_ready(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not isinstance(self._n_variables, int):
                raise OptUtilitiesError(
                    "You must specify the dimensionality of the optimisation problem before using ."
                )
            func(self, *args, **kwargs)

        return wrapper

    @property
    def algorithm_name(self):
        """
        Name of the optimisation algorithm.
        """
        return self._algorithm_name

    @algorithm_name.setter
    def algorithm_name(self, name):
        """
        Setter for the name of the optimisation algorithm.

        Parameters
        ----------
        name: str
            Name of the optimisation algorithm

        Raises
        ------
        OptUtilitiesError:
            If the algorithm is not recognised
        """
        if name not in NLOPT_ALG_MAPPING:
            raise OptUtilitiesError(f"Unknown or unmapped algorithm: {name}")
        self._algorithm_name = name

    @property
    def n_variables(self):
        """
        Dimension of the optimisater.
        """
        return self._n_variables

    @n_variables.setter
    def n_variables(self, value):
        """
        Setter for the dimenstion of the optimiser.
        """
        self._n_variables = value

        if value is not None:
            self.set_algorithm()
            self.set_termination_conditions(self.opt_conditions)
            self.set_algorithm_parameters(self.opt_parameters)
            self.lower_bounds = np.zeros(value)
            self.upper_bounds = np.ones(value)

    def _append_constraint_tols(self, constraint, tolerance):
        """
        Append constraint function and tolerances.
        """
        self.constraints.append(constraint)
        self.constraint_tols.append(tolerance)

    def set_algorithm(self):
        """
        Initialise the underlying NLOPT algorithm.
        """
        algorithm = NLOPT_ALG_MAPPING[self.algorithm_name]
        self._opt = nlopt.opt(algorithm, int(self._n_variables))

    @_opt_inputs_ready
    def set_algorithm_parameters(self, opt_parameters):
        """
        Set the optimisation algorithm parameters to use.

        Parameters
        ----------
        opt_parameters: dict
            Optimisation algorithm parameters to use
        """
        for k, v in opt_parameters.items():
            if self._opt.has_param(k):
                self._opt.set_param(k, v)
            else:
                bluemira_warn(f"Unrecognised algorithm parameter: {k}")

    @_opt_inputs_ready
    def set_termination_conditions(self, opt_conditions):
        """
        Set the optimisation algorithm termination condition(s) to use.

        Parameters
        ----------
        opt_conditions: dict
            Termination conditions for the optimisation algorithm
        """
        if not opt_conditions:
            raise OptUtilitiesError(
                "You must specify at least one termination criterion for the optimisation algorithm."
            )

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

    @_opt_inputs_ready
    def set_objective_function(self, f_objective):
        """
        Set the objective function (minimisation).

        Parameters
        ----------
        f_objective: callable
            Objective function to minimise
        """
        f_objective = _NLOPTObjectiveFunction(f_objective)
        self._f_objective = f_objective
        self._opt.set_min_objective(f_objective)

    @_opt_inputs_ready
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

    @_opt_inputs_ready
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

    @_opt_inputs_ready
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

    @_opt_inputs_ready
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

        self._opt.add_inequality_constraint(f_constraint, tolerance)
        self._append_constraint_tols(f_constraint, tolerance)

    @_opt_inputs_ready
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

        self._opt.add_inequality_mconstraint(f_constraint, tolerance)
        self._append_constraint_tols(f_constraint, tolerance)

    def optimise(self, x0):
        """
        Run the optimiser.

        Parameters
        ----------
        x0: np.ndarray
            Starting solution vector

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
        except OptVariablesError:
            # Probably still some rounding errors due to numerical gradients
            # It's likely that the last call was still a reasonably good solution.
            bluemira_warn("Badly behaved numerical gradients are causing trouble...")
            self.rms = self._opt.last_optimum_value()
            x_star = np.round(self._f_objective.last_x, 6)
        except RuntimeError:
            # Usually "more than iter SQP iterations"
            raise ExternalOptError("Usually more than iter SQP iterations")

        self.rms = self._opt.last_optimum_value()
        self.n_evals = self._opt.get_numevals()
        process_NLOPT_result(self._opt)
        return x_star

    _opt_inputs_ready = staticmethod(_opt_inputs_ready)

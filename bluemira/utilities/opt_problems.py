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
Generic OptimisationProblem, OptimisationConstraint and OptimisationObjective interfaces.
"""
import inspect
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from bluemira.utilities.optimiser import Optimiser

__all__ = ["OptimisationProblem", "OptimisationObjective", "OptimisationConstraint"]


class OptimisationConstraint:
    """
    Data class to store information needed to apply a constraint
    to an optimisation problem.

    Parameters
    ----------
    f_constraint: callable
        Constraint function to apply to problem.
        For NLOpt constraints, onstraint functions should be of the form
        f_constraint(cls, constraint, x, grad, f_constraint_args)
    f_constraint_args: dict (default = None)
        Additional arguments to pass to NLOpt constraint function when called.
    tolerance: array
        Array of tolerances to use when applying the optimisation constraint.
    constraint_type: string (default: "inequality")
        Type of constraint to apply, either "inequality" or "equality".
    """

    def __init__(
        self,
        f_constraint,
        f_constraint_args=None,
        tolerance=np.array([1e-6]),
        constraint_type="inequality",
    ):
        self._tolerance = tolerance
        self._constraint_type = constraint_type
        self._f_constraint = f_constraint
        self._args = f_constraint_args

    def __call__(self, constraint, vector, grad):
        """
        Call to constraint function used by NLOpt, passing in arguments.
        """
        return self._f_constraint(constraint, vector, grad, **self._args)

    def apply_constraint(self, opt_problem):
        """
        Apply constraint to a specified OptimisationProblem.
        """
        # Add optimisation problem to constraint arguments if needed by
        # constraint function
        if "opt_problem" in inspect.getargspec(self._f_constraint).args:
            self._args["opt_problem"] = opt_problem

        # Apply constraint to optimiser
        if self._constraint_type == "inequality":
            opt_problem.opt.add_ineq_constraints(self, self._tolerance)
        elif self._constraint_type == "equality":
            opt_problem.opt.add_eq_constraints(self, self._tolerance)


class OptimisationObjective:
    """
    Data class to store information needed to apply a constraint
    to an optimisation problem.

    Parameters
    ----------
    f_objective: callable
        Objective function to apply to problem.
        For NLOpt objectives, objective functions should be of the form
        f_objective(cls, x, grad, f_objective_args)
    _args: dict (default = None)
        Additional arguments to pass to NLOpt objective function when called.
    """

    def __init__(self, f_objective, f_objective_args=None):
        self._f_objective = f_objective
        self._args = f_objective_args

    def __call__(self, vector, grad):
        """
        Call to objective function used by NLOpt, passing in arguments.
        """
        return self._f_objective(vector, grad, **self._args)

    def apply_objective(self, opt_problem):
        """
        Apply objective to a specified OptimisationProblem.
        """
        # Add optimisation problem to objective arguments if needed by
        # objective function
        if "opt_problem" in inspect.getargspec(self._f_objective).args:
            self._args["opt_problem"] = opt_problem

        # Apply objective to optimiser
        opt_problem.opt.set_objective_function(self)


class OptimisationProblem(ABC):
    """
    Generic OptimisationProblem to be subclassed for defining optimisation
    routines in Bluemira.
    """

    def __init__(
        self,
        parameterisation,
        optimiser: Optimiser,
        objective: OptimisationObjective,
        constraints: List[OptimisationConstraint],
    ):
        self._parameterisation = parameterisation
        self.opt = optimiser
        self._objective = objective
        self._constraints = constraints

    def set_up_optimiser(self, dimension: int, bounds: np.array):
        """
        Set up NLOpt-based optimiser with algorithm,  bounds, tolerances, and
        constraint & objective functions.

        Parameters
        ----------
        dimension: int
            Number of independent variables in the state vector to be optimised.
        bounds: tuple
            Tuple containing lower and upper bounds on the state vector.
        """
        # Build NLOpt optimiser, with optimisation strategy and length
        # of state vector
        self.opt.build_optimiser(n_variables=dimension)

        # Apply objective function to optimiser
        self._objective.apply_objective(self)

        # Apply constraints
        self.apply_constraints(self.opt, self._constraints)

        # Set state vector bounds (current limits)
        self.opt.set_lower_bounds(bounds[0])
        self.opt.set_upper_bounds(bounds[1])

    def apply_constraints(
        self, opt: Optimiser, opt_constraints: List[OptimisationConstraint]
    ):
        """
        Updates the optimiser in-place to apply problem constraints.
        To be overidden by child classes to apply specific constraints.

        Parameters
        ----------
        opt: Optimiser
            Optimiser on which to apply the constraints. Updated in place.
        opt_constraints: iterable
            Iterable of OptimisationConstraint objects containing optimisation
            constraints to be applied to the Optimiser.

        Notes
        -----
        Lambda functions are used here to ensure the CoilsetPositionCOP is passed
        to the function, to allow any properties that are stored in the
        CoilsetPositionCOP to be accessed at runtime.
        """
        for _opt_constraint in opt_constraints:
            _opt_constraint.apply_constraint(self)
        return opt

    def initialise_state(self, parameterisation) -> np.array:
        """
        Initialises state vector to be passed to optimiser from object used
        in parameterisation, at each optimise() call.
        To be overridden as needed.
        """
        initial_state = parameterisation
        return initial_state

    def update_parametrisation(self, state: np.array):
        """
        Update parameterisation object using the state vector.
        To be overridden as needed.
        """
        parameterisation = state
        return parameterisation

    @abstractmethod
    def optimise(self):
        """
        Optimisation routine used to return an optimised parameterisation.

        Returns
        -------
        self._parameterisation:
            Optimised parameterisation object.
        """
        initial_state = self.initialise_state(self._parameterisation)
        opt_state = self._opt.optimise(initial_state)
        self._parameterisation = self.update_parametrisation(opt_state)
        return self._parameterisation

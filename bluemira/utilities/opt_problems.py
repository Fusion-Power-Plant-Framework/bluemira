# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Generic OptimisationProblem, OptimisationConstraint and OptimisationObjective interfaces.
"""
import inspect
import warnings
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from bluemira.utilities.optimiser import Optimiser

warnings.warn(
    f"The module '{__name__}' is deprecated and will be removed in v2.0.0.\n"
    "See "
    "https://bluemira.readthedocs.io/en/latest/optimisation/optimisation.html "
    "for documentation of the new optimisation module.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = ["OptimisationProblem", "OptimisationObjective", "OptimisationConstraint"]


class OptimisationConstraint:
    """
    Data class to store information needed to apply a constraint
    to an optimisation problem.

    Parameters
    ----------
    f_constraint: callable
        Constraint function to apply to problem.
        For NLOpt constraints, constraint functions should be of the form
        f_constraint(constraint, x, grad, f_constraint_args)
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
        if "opt_problem" in inspect.getfullargspec(self._f_constraint).args:
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
        f_objective(x, grad, f_objective_args)
    f_objective_args: dict (default = None)
        Additional arguments to pass to NLOpt objective function when called.
    """

    def __init__(self, f_objective, f_objective_args=None):
        self._f_objective = f_objective
        self._args = f_objective_args if f_objective_args is not None else {}

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
        if "opt_problem" in inspect.getfullargspec(self._f_objective).args:
            self._args["opt_problem"] = opt_problem

        # Apply objective to optimiser
        opt_problem.opt.set_objective_function(self)


class OptimisationProblem(ABC):
    """
    Abstract base class to be subclassed for defining optimisation
    routines in bluemira.

    Subclasses should provide an optimise() method that
    returns an optimised parameterisation object, optimised according
    to a specific objective function for that subclass.

    Parameters
    ----------
    parameterisation: any
        Object storing parameterisation data to be optimised.
    optimiser: bluemira.utilities.optimiser.Optimiser (default: None)
        Optimiser object to use for constrained optimisation.
        Does not need to be provided if not used by
        optimise(), such as for purely unconstrained
        optimisation.
    objective: OptimisationObjective (default: None)
        OptimisationObjective storing objective information to
        provide to the Optimiser.
    constraints: List[OptimisationConstraint]
        Optional list of OptimisationConstraint objects storing
        information about constraints that must be satisfied
        during the optimisation, to be provided to the
        Optimiser.
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

    def set_up_optimiser(self, dimension: int, bounds: np.ndarray):
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
        if self._constraints is not None:
            self.apply_constraints(self.opt, self._constraints)

        # Set state vector bounds
        self.opt.set_lower_bounds(bounds[0])
        self.opt.set_upper_bounds(bounds[1])

    def apply_constraints(
        self, opt: Optimiser, opt_constraints: List[OptimisationConstraint]
    ) -> Optimiser:
        """
        Updates the optimiser in-place to apply problem constraints.
        To be overridden by child classes to apply specific constraints.

        Parameters
        ----------
        opt: bluemira.utilities.optimiser.Optimiser
            Optimiser on which to apply the constraints. Updated in place.
        opt_constraints: iterable
            Iterable of OptimisationConstraint objects containing optimisation
            constraints to be applied to the Optimiser.
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
        return parameterisation

    def update_parametrisation(self, state: np.ndarray) -> np.ndarray:
        """
        Update parameterisation object using the state vector.
        To be overridden as needed.
        """
        return state

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
        opt_state = self.opt.optimise(initial_state)
        self._parameterisation = self.update_parametrisation(opt_state)
        return self._parameterisation

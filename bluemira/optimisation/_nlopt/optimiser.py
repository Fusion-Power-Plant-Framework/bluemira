# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Union

import nlopt
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation._algorithm import Algorithm
from bluemira.optimisation._nlopt.conditions import NLOptConditions
from bluemira.optimisation._nlopt.functions import (
    Constraint,
    ConstraintType,
    ObjectiveFunction,
)
from bluemira.optimisation._optimiser import Optimiser, OptimiserResult
from bluemira.optimisation._typing import ObjectiveCallable, OptimiserCallable
from bluemira.optimisation.error import OptimisationError, OptimisationParametersError

_NLOPT_ALG_MAPPING = {
    Algorithm.SLSQP: nlopt.LD_SLSQP,
    Algorithm.COBYLA: nlopt.LN_COBYLA,
    Algorithm.SBPLX: nlopt.LN_SBPLX,
    Algorithm.MMA: nlopt.LD_MMA,
    Algorithm.BFGS: nlopt.LD_LBFGS,
    Algorithm.DIRECT: nlopt.GN_DIRECT,
    Algorithm.DIRECT_L: nlopt.GN_DIRECT_L,
    Algorithm.CRS: nlopt.GN_CRS2_LM,
    Algorithm.ISRES: nlopt.GN_ISRES,
}


class NloptOptimiser(Optimiser):
    r"""
    Optimiser implementation using NLOpt as the backend.

    Parameters
    ----------
    algorithm:
        The optimisation algorithm to use. Available algorithms are:

            * SLSQP
            * COBYLA
            * SBPLX
            * MMA
            * BFGS
            * DIRECT
            * DIRECT_L
            * CRS
            * ISRES

    n_variables:
        The number of optimisation parameters.
    f_objective:
        The objective function to minimise. This function must take one
        argument (a numpy array), and return a numpy array or float.
    df_objective:
        The derivative of the objective function. This must take the
        form: `f(x) -> y` where `x` is a numpy array containing the
        optimization parameters, and `y` is a numpy array where each
        element `i` is the partial derivative `\partialf/\partialdx_{i}`.
        If not given, a numerical approximation of the gradient is used.
    opt_conditions:
        The stopping conditions for the optimiser. At least one stopping
        condition is required. Supported conditions are:

            * ftol_abs: float
            * ftol_rel: float
            * xtol_abs: float
            * xtol_rel: float
            * max_eval: int
            * max_time: float
            * stop_val: float

    opt_parameters:
        Parameters specific to the algorithm being used. Consult NLopt
        documentation for these.
    keep_history:
        Whether to record the history of each step of the optimisation.
        (default: False)
    """

    SLSQP = "SLSQP"
    COBYLA = "COBYLA"
    SBPLX = "SBPLX"
    MMA = "MMA"
    BFGS = "BFGS"
    DIRECT = "DIRECT"
    DIRECT_L = "DIRECT_L"
    CRS = "CRS"
    ISRES = "ISRES"

    def __init__(
        self,
        algorithm: Union[str, Algorithm],
        n_variables: int,
        f_objective: ObjectiveCallable,
        df_objective: Optional[OptimiserCallable] = None,
        opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
        opt_parameters: Optional[Mapping[str, Any]] = None,
        keep_history: bool = False,
    ):
        opt_conditions = {} if opt_conditions is None else opt_conditions
        opt_parameters = {} if opt_parameters is None else opt_parameters
        self._keep_history = keep_history

        self._set_algorithm(algorithm)
        self._opt = nlopt.opt(_NLOPT_ALG_MAPPING[self.algorithm], n_variables)
        self._set_objective_function(f_objective, df_objective)
        self._set_termination_conditions(opt_conditions)
        self._set_algorithm_parameters(opt_parameters)
        self._eq_constraints: List[Constraint] = []
        self._ineq_constraints: List[Constraint] = []

    @property
    def algorithm(self) -> Algorithm:
        """Return the optimiser's algorithm."""
        return self._algorithm

    @property
    def opt_conditions(self) -> Dict[str, float]:
        """Return the optimiser's stopping conditions."""
        return self._opt_conditions.to_dict()

    @property
    def opt_parameters(self) -> Mapping:
        """Return the optimiser algorithms's parameters."""
        return self._opt_parameters

    @property
    def lower_bounds(self) -> np.ndarray:
        """Return the lower bounds for the optimisation parameters."""
        return self._opt.get_lower_bounds().copy()

    @property
    def upper_bounds(self) -> np.ndarray:
        """Return the upper bounds for the optimisation parameters."""
        return self._opt.get_upper_bounds().copy()

    def add_eq_constraint(
        self,
        f_constraint: OptimiserCallable,
        tolerance: np.ndarray,
        df_constraint: Optional[OptimiserCallable] = None,
    ) -> None:
        """
        Add an equality constraint to the optimiser.

        See docs of
        :obj:`optimisation._optimiser.Optimiser.add_eq_constraint` for
        details of the form the arguments should take.
        """
        if self.algorithm not in [Algorithm.SLSQP, Algorithm.COBYLA, Algorithm.ISRES]:
            raise OptimisationError(
                f"Algorithm '{self.algorithm.name}' does not support equality "
                f"constraints."
            )
        constraint = Constraint(
            ConstraintType.EQUALITY,
            f_constraint,
            tolerance,
            df_constraint,
            bounds=(self.lower_bounds, self.upper_bounds),
        )
        self._opt.add_equality_mconstraint(constraint.call, constraint.tolerance)
        self._eq_constraints.append(constraint)

    def add_ineq_constraint(
        self,
        f_constraint: OptimiserCallable,
        tolerance: np.ndarray,
        df_constraint: Optional[OptimiserCallable] = None,
    ) -> None:
        r"""
        Add an inequality constraint to the optimiser.

        See docs of
        :obj:`optimisation._optimiser.Optimiser.add_ineq_constraint` for
        details of the form the arguments should take.
        """
        if self.algorithm not in [Algorithm.SLSQP, Algorithm.COBYLA, Algorithm.ISRES]:
            raise OptimisationError(
                f"Algorithm '{self.algorithm.name}' does not support inequality "
                f"constraints."
            )
        constraint = Constraint(
            ConstraintType.INEQUALITY,
            f_constraint,
            tolerance,
            df_constraint,
            bounds=(self.lower_bounds, self.upper_bounds),
        )
        self._opt.add_inequality_mconstraint(constraint.call, constraint.tolerance)
        self._ineq_constraints.append(constraint)

    def optimise(self, x0: Optional[np.ndarray] = None) -> OptimiserResult:
        """
        Run the optimiser.

        See docs of
        :obj:`optimisation._optimiser.Optimiser.optimise` for details of
        the form the arguments should take.
        """
        if x0 is None:
            x0 = _initial_guess_from_bounds(self.lower_bounds, self.upper_bounds)

        try:
            x_star = self._opt.optimize(x0)
        except nlopt.RoundoffLimited:
            # It's likely that the last call was still a reasonably good solution.
            bluemira_warn(
                "optimisation: round-off error occurred, returning last optimisation "
                "parameterisation."
            )
            if self._objective.history:
                fx_values = np.array(self._objective.history).T[1]
                f_x = np.min(fx_values)
                arg_min_fx = np.argmin(fx_values)
                x_star = self._objective.history[arg_min_fx][0]
            else:
                x_star = x0
                f_x = np.infty

        return OptimiserResult(
            x_star,
            f_x=f_x,
            n_evals=self._opt.get_numevals(),
            history=self._objective.history,
        )

    def set_lower_bounds(self, bounds: np.ndarray) -> None:
        """
        Set the lower bound for each optimisation parameter.

        Set to `-np.inf` to unbound the parameter's minimum.
        """
        bounds = np.array(bounds)
        _check_bounds(self._opt.get_dimension(), bounds)
        self._opt.set_lower_bounds(bounds)
        # As we use the optimisation variable bounds when calculating an
        # approximate derivative, we must set the new bounds on the
        # objective function and constraints.
        self._objective.set_approx_derivative_lower_bound(bounds)
        for constraint in self._eq_constraints + self._ineq_constraints:
            constraint.set_approx_derivative_lower_bound(bounds)

    def set_upper_bounds(self, bounds: Union[np.ndarray, float]) -> None:
        """
        Set the upper bound for each optimisation parameter.

        Set to `np.inf` to unbound the parameter's minimum.
        """
        bounds = np.array(bounds)
        _check_bounds(self._opt.get_dimension(), bounds)
        self._opt.set_upper_bounds(bounds)
        # As we use the optimisation variable bounds when calculating an
        # approximate derivative, we must set the new bounds on the
        # objective function and constraints.
        self._objective.set_approx_derivative_upper_bound(bounds)
        for constraint in self._eq_constraints + self._ineq_constraints:
            constraint.set_approx_derivative_upper_bound(bounds)

    def _set_algorithm(self, alg: Union[str, Algorithm]) -> None:
        """Set the optimiser's algorithm."""
        self._algorithm = _check_algorithm(alg)

    def _set_objective_function(
        self, func: ObjectiveCallable, df: Union[None, OptimiserCallable]
    ) -> None:
        """Wrap and set the objective function."""
        self._objective = ObjectiveFunction(
            func, df, bounds=(self.lower_bounds, self.upper_bounds)
        )
        if self._keep_history:
            self._opt.set_min_objective(self._objective.call_with_history)
        else:
            self._opt.set_min_objective(self._objective.call)

    def _set_termination_conditions(
        self, opt_conditions: Mapping[str, Union[int, float]]
    ) -> None:
        """Validate and set the termination conditions."""
        self._opt_conditions = NLOptConditions(**opt_conditions)
        if self._opt_conditions.ftol_abs:
            self._opt.set_ftol_abs(self._opt_conditions.ftol_abs)
        if self._opt_conditions.ftol_rel:
            self._opt.set_ftol_rel(self._opt_conditions.ftol_rel)
        if self._opt_conditions.xtol_abs:
            self._opt.set_xtol_abs(self._opt_conditions.xtol_abs)
        if self._opt_conditions.xtol_rel:
            self._opt.set_xtol_rel(self._opt_conditions.xtol_rel)
        if self._opt_conditions.max_time:
            self._opt.set_maxtime(self._opt_conditions.max_time)
        if self._opt_conditions.max_eval:
            self._opt.set_maxeval(self._opt_conditions.max_eval)
        if self._opt_conditions.stop_val:
            self._opt.set_stopval(self._opt_conditions.stop_val)

    def _set_algorithm_parameters(self, opt_parameters: Mapping) -> None:
        self._opt_parameters = opt_parameters
        unrecognised = []
        for k, v in self.opt_parameters.items():
            if self._opt.has_param(k):
                self._opt.set_param(k, v)
            elif k == "initial_step":
                self._opt.set_initial_step(v)
            else:
                unrecognised.append(k)

        if unrecognised:
            raise OptimisationParametersError(
                f"Unrecognised algorithm parameter(s): {str(unrecognised)[1:-1]}"
            )


def _check_algorithm(algorithm: Union[str, Algorithm]) -> Algorithm:
    """Validate, and convert, the given algorithm."""
    if isinstance(algorithm, str):
        return Algorithm[algorithm]
    elif isinstance(algorithm, Algorithm):
        return algorithm
    raise TypeError(f"Cannot set algorithm with object of type '{type(algorithm)}'.")


def _check_bounds(n_dims: int, new_bounds: np.ndarray) -> None:
    """Validate that the bounds have the correct dimensions."""
    if new_bounds.ndim != 1 or new_bounds.size != n_dims:
        raise ValueError(
            f"Cannot set bounds with shape '{new_bounds.shape}', "
            f"array must be one dimensional and have '{n_dims}' elements."
        )


def _initial_guess_from_bounds(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Derive an initial guess for the optimiser.

    Takes the center of the bounds for each parameter.
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

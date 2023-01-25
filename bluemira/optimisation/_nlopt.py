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
"""Implementation of optimiser using NLOpt as a backend."""

from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import nlopt
import numpy as np

from bluemira.optimisation._algorithm import Algorithm
from bluemira.optimisation._optimiser import Optimiser, OptimiserResult
from bluemira.optimisation._tools import approx_derivative
from bluemira.optimisation._typing import OptimiserCallable
from bluemira.optimisation.error import (
    OptimisationConditionsError,
    OptimisationError,
    OptimisationParametersError,
)

EPS = np.finfo(np.float64).eps
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
    algorithm: Union[str, Algorithm]
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

    n_variables: int
        The number of optimisation parameters.
    f_objective: Callable[[Arg(np.ndarray, 'x')], np.ndarray]
        The objective function to minimise. This function must take one
        argument (a numpy array), and return a numpy array or float.
    df_objective: Optional[Callable[[Arg(np.ndarray, 'x')], np.ndarray]]
        The derivative of the objective function. This must take the
        form: `f(x) -> y` where `x` is a numpy array containing the
        optimization parameters, and `y` is a numpy array where each
        element `i` is the partial derivative `\partialf/\partialdx_{i}`.
        If not given, a numerical approximation of the gradient is used.
    opt_conditions: Mapping[str, Union[int, float]]
        The stopping conditions for the optimiser. At least one stopping
        condition is required. Supported conditions are:

            * ftol_abs: float
            * ftol_rel: float
            * xtol_abs: float
            * xtol_rel: float
            * max_eval: int
            * max_time: float
            * stop_val: float

    opt_parameters: Optional[Mapping[str, Any]]
        Parameters specific to the algorithm being used. Consult NLopt
        documentation for these.
    keep_history: bool
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
        f_objective: OptimiserCallable,
        df_objective: Optional[OptimiserCallable] = None,
        opt_conditions: Mapping[str, Union[int, float]] = None,
        opt_parameters: Mapping[str, Any] = None,
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
        self._eq_constraints: List[_Constraint] = []
        self._ineq_constraints: List[_Constraint] = []

    @property
    def algorithm(self) -> Algorithm:
        """Return the optimiser's algorithm."""
        return self._algorithm

    @property
    def opt_conditions(self) -> _Conditions:
        """Return the optimiser's stopping conditions."""
        return self._opt_conditions

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
    ):
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
        constraint = _Constraint(
            _ConstraintType.EQUALITY, f_constraint, tolerance, df_constraint
        )
        self._opt.add_equality_mconstraint(constraint.f, constraint.tolerance)
        self._eq_constraints.append(constraint)

    # TODO(hsaunders1904): resolve tolerance typing conflict here and in base class.
    #  Should we be allowing a float? How do we get the dimensionality of the constraint?
    def add_ineq_constraint(
        self,
        f_constraint: OptimiserCallable,
        tolerance: np.ndarray,
        df_constraint: Optional[OptimiserCallable] = None,
    ):
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
        constraint = _Constraint(
            _ConstraintType.INEQUALITY, f_constraint, tolerance, df_constraint
        )
        self._opt.add_inequality_mconstraint(constraint.nlopt_call, constraint.tolerance)
        self._ineq_constraints.append(constraint)

    def optimise(self, x0: Optional[np.ndarray] = None) -> OptimiserResult:
        """
        Run the optimiser.

        See docs of
        :obj:`optimisation._optimiser.Optimiser.optimise` for details of
        the form the arguments should take.
        """
        if x0 is None:
            # TODO(hsaunders1904): deal with the case where only one
            #  set of bounds are finite
            bounds = np.array([self.lower_bounds, self.upper_bounds])
            # bounds are +/- inf by default, change to real numbers so
            # we can take an average
            np.nan_to_num(
                bounds,
                posinf=np.finfo(np.float64).max,
                neginf=np.finfo(np.float64).min,
                copy=False,
            )
            x0 = np.mean(bounds, axis=0)

        try:
            x_star = self._opt.optimize(x0)
        except nlopt.RoundoffLimited:
            # It's likely that the last call was still a reasonably good solution.
            warnings.warn(
                "optimisation: round-off error occurred, returning last optimisation "
                "parameterisation."
            )
            if self._objective.history:
                x_star = self._objective.history[-1]
            else:
                x_star = x0
        return OptimiserResult(
            x_star, n_evals=self._opt.get_numevals(), history=self._objective.history
        )

    def set_lower_bounds(self, bounds: np.ndarray):
        """
        Set the lower bound for each optimisation parameter.

        Set to `-np.inf` to unbound the parameter's minimum.
        """
        self._opt.set_lower_bounds(bounds)

    def set_upper_bounds(self, bounds: np.ndarray):
        """
        Set the upper bound for each optimisation parameter.

        Set to `np.inf` to unbound the parameter's minimum.
        """
        # TODO(hsaunders1904): validate bounds.size
        self._opt.set_upper_bounds(bounds)

    def _set_algorithm(self, alg: Union[str, Algorithm]):
        """Set the optimiser's algorithm."""
        self._algorithm = _check_algorithm(alg)

    def _set_objective_function(self, func: Callable, df: Union[None, Callable]):
        """Wrap and set the objective function."""
        self._objective = _NloptObjectiveFunction(func, df)
        if self._keep_history:
            self._opt.set_min_objective(self._objective.call_with_history)
        else:
            self._opt.set_min_objective(self._objective.call)

    def _set_termination_conditions(self, opt_conditions: Mapping):
        """Validate and set the termination conditions."""
        self._opt_conditions = _Conditions(**opt_conditions)
        if self.opt_conditions.ftol_abs:
            self._opt.set_ftol_abs(self.opt_conditions.ftol_abs)
        if self.opt_conditions.ftol_rel:
            self._opt.set_ftol_rel(self.opt_conditions.ftol_rel)
        if self.opt_conditions.xtol_abs:
            self._opt.set_xtol_abs(self.opt_conditions.xtol_abs)
        if self.opt_conditions.xtol_rel:
            self._opt.set_xtol_rel(self.opt_conditions.xtol_rel)
        if self.opt_conditions.max_time:
            self._opt.set_maxtime(self.opt_conditions.max_time)
        if self.opt_conditions.max_eval:
            self._opt.set_maxeval(self.opt_conditions.max_eval)
        if self.opt_conditions.stop_val:
            self._opt.set_stopval(self.opt_conditions.stop_val)

    def _set_algorithm_parameters(self, opt_parameters: Mapping):
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


class _ConstraintType(enum.Enum):
    EQUALITY = enum.auto()
    INEQUALITY = enum.auto()


class _Constraint:
    """Holder for NLOpt constraint functions."""

    def __init__(
        self,
        constraint_type: _ConstraintType,
        f: OptimiserCallable,
        tolerance: np.ndarray,
        df: Optional[OptimiserCallable] = None,
    ):
        self.constraint_type = constraint_type
        self.f = f
        self.tolerance = tolerance
        self.df = df if df is not None else self._approx_derivative
        self.f0: Optional[np.ndarray] = None

    def nlopt_call(self, result: np.ndarray, x: np.ndarray, grad: np.ndarray):
        """
        Execute the constraint function in the form required by NLOpt.

        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#vector-valued-constraints
        """
        if grad.size > 0:
            grad[:] = self.df(x)
        result[:] = self.f(x)
        self.f0 = result

    def _approx_derivative(self, x: np.ndarray) -> np.ndarray:
        return approx_derivative(self.f, x, f0=self.f0)


@dataclass
class _Conditions:
    """Hold and validate optimiser stopping conditions."""

    ftol_abs: Optional[float] = None
    ftol_rel: Optional[float] = None
    xtol_abs: Optional[float] = None
    xtol_rel: Optional[float] = None
    max_eval: Optional[int] = None
    max_time: Optional[float] = None
    stop_val: Optional[float] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for condition in [
            self.ftol_abs,
            self.ftol_rel,
            self.xtol_abs,
            self.xtol_rel,
        ]:
            if condition and condition < EPS:
                warnings.warn(
                    "optimisation: Setting stopping condition to less than machine "
                    "precision. This condition may never be met."
                )
        if self._no_stopping_condition_set():
            raise OptimisationConditionsError(
                "Must specify at least one stopping condition for the optimiser."
            )

    def _no_stopping_condition_set(self):
        return all(
            condition is None
            for condition in [
                self.ftol_abs,
                self.ftol_rel,
                self.xtol_abs,
                self.xtol_rel,
                self.max_eval,
                self.max_time,
                self.stop_val,
            ]
        )


class _NloptObjectiveFunction:
    """
    Holds an objective function for an NLOpt optimiser.

    Adapts the given objective function, and optional derivative, to a
    form understood by NLOpt.

    If no optimiser derivative is given, and the algorithm is gradient
    based, a numerical approximation of the gradient is calculated.
    """

    def __init__(
        self,
        f: OptimiserCallable,
        df: Optional[OptimiserCallable] = None,
        bounds: Tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
    ):
        self.f = f
        self.f0 = None
        self.df = df if df is not None else self._approx_derivative
        self.bounds = bounds
        self.history: List[np.ndarray] = []

    def call(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Execute the NLOpt objective function."""
        if grad.size > 0:
            grad[:] = self.df(x)
        self.f0 = self.f(x)
        return self.f0

    def call_with_history(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Execute the NLOpt objective function, recording the iteration history"""
        self.history.append(np.copy(x))
        if grad.size > 0:
            grad[:] = self.df(x)
        self.f0 = self.f(x)
        return self.f0

    def _approx_derivative(self, x: np.ndarray) -> np.ndarray:
        return approx_derivative(self.f, x, f0=self.f0, bounds=self.bounds)


def _check_algorithm(algorithm: Union[str, Algorithm]) -> Algorithm:
    """Validate, and convert, the given algorithm."""
    if isinstance(algorithm, str):
        return Algorithm.from_string(algorithm)
    elif isinstance(algorithm, Algorithm):
        return algorithm
    raise TypeError(f"Cannot set algorithm with object of type '{type(algorithm)}'.")

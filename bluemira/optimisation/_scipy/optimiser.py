# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Scipy optimisation interface"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation._algorithm import Algorithm, AlgorithmType
from bluemira.optimisation._optimiser import Optimiser, OptimiserResult
from bluemira.optimisation._scipy.conditions import ScipyConditions, _convert_to_scipy
from bluemira.optimisation._scipy.parameters import _make_alg_params
from bluemira.optimisation._tools import (
    _check_bounds,
    _initial_guess_from_bounds,
    process_scipy_result,
)
from bluemira.optimisation.error import OptimisationError
from bluemira.utilities.error import OptVariablesError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bluemira.optimisation.typing import ObjectiveCallable, OptimiserCallable


SCIPY_ALG_MAPPING = {
    Algorithm.COBYLA_SCIPY: "COBYLA",
    Algorithm.COBYQA: "COBYQA",
    Algorithm.L_BFGS_B: "L_BFGS_B",
    Algorithm.NELDER_MEAD: "NELDER_MEAD",
    Algorithm.POWELL: "POWELL",
    Algorithm.SLSQP_SCIPY: "SLSQP",
    Algorithm.TNC: "TNC",
    Algorithm.TRUST_CONSTR: "TRUST_CONSTR",
}

DF_SUPPORTED = {
    SCIPY_ALG_MAPPING[Algorithm.NELDER_MEAD],
    SCIPY_ALG_MAPPING[Algorithm.POWELL],
    SCIPY_ALG_MAPPING[Algorithm.COBYLA_SCIPY],
}


class ScipyOptimiser(Optimiser):
    def __init__(
        self,
        algorithm: AlgorithmType,
        n_variables: int,
        f_objective: ObjectiveCallable,
        df_objective: OptimiserCallable | None = None,
        opt_conditions: Mapping[str, int | float] | None = None,
        opt_parameters: Mapping[str, Any] | None = None,
        *,
        keep_history: bool = False,
    ):
        self._set_algorithm(algorithm)
        self.n_variables = n_variables
        self.f_objective = f_objective
        self.df_objective = df_objective
        self._set_conditions(opt_conditions or {})
        self._set_parameters(opt_parameters or {})
        self.set_lower_bounds(np.ones(n_variables) * -np.inf)
        self.set_upper_bounds(np.ones(n_variables) * np.inf)
        self.keep_history = keep_history
        self._eq_constraints = []
        self._ineq_constraints = []

    @property
    def algorithm(self) -> AlgorithmType:
        """
        Returns
        -------
        :
            the optimiser's algorithm.
        """
        return self._algorithm

    @property
    def opt_conditions(self) -> dict[str, float]:
        """
        Returns
        -------
        :
            the optimiser's stopping conditions.
        """
        return self._opt_conditions.to_dict()

    @property
    def opt_parameters(self) -> Mapping[str, int | float]:
        """
        Returns
        -------
        :
            the optimiser algorithms's parameters.
        """
        return self._opt_parameters

    @property
    def lower_bounds(self) -> np.ndarray:
        """
        Returns
        -------
        :
            the lower bounds for the optimisation parameters.
        """
        return self._lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        """
        Returns
        -------
        :
            the upper bounds for the optimisation parameters.
        """
        return self._upper_bounds

    def _set_algorithm(self, alg: AlgorithmType) -> None:
        """Set the optimiser's algorithm."""
        self._algorithm = Algorithm(alg)

    def _set_conditions(self, opt_conditions: Mapping[str, int | float]) -> None:
        """Initialise the optimiser's conditions."""
        self._opt_conditions = ScipyConditions(
            **_convert_to_scipy((opt_conditions), SCIPY_ALG_MAPPING[self.algorithm])
        )

    def _set_parameters(self, opt_parameters: Mapping[str, int | float]) -> None:
        """Initialise the optimiser's parameters."""
        self._opt_parameters = _make_alg_params(
            opt_parameters, SCIPY_ALG_MAPPING[self.algorithm]
        )

    def add_eq_constraint(
        self,
        f_constraint: OptimiserCallable,
        tolerance: np.ndarray,
        df_constraint: OptimiserCallable | None = None,
    ) -> None:
        r"""
        Add an equality constraint to the optimiser.

        The constraint is a vector-valued, non-linear, equality
        constraint of the form :math:`f_{c}(x) = 0`.

        The constraint function should have the form
        :math:`f(x) \rightarrow y`, where:

            * :math:`x` is a numpy array of the optimisation parameters.
            * :math:`y` is a numpy array containing the values of the
              constraint at :math:`x`, with size :math:`m`, where
              :math:`m` is the dimensionality of the constraint.

        Parameters
        ----------
        f_constraint:
            The constraint function, with form as described above.
        tolerance:
            The tolerances for each optimisation parameter.
        df_constraint:
            The gradient of the constraint function. This should have
            the same form as the constraint function, however its output
            array should have dimensions :math:`m \times n` where
            :math`m` is the dimensionality of the constraint, and
            :math:`n` is the number of optimisation parameters.

        Raises
        ------
        OptimisationError
            Algorithm does not support equality constraints.

        Notes
        -----
        Equality constraints are only supported by algorithms:

            * SLSQP
            * COBYQA
            * TRUST_CONSTR

            However, equality constraints can be converted to pairs
            of inequality constraints to work with other algorithms,
            such as COBYLA.

        """
        if self.algorithm == Algorithm.SLSQP_SCIPY:
            self._eq_constraints.append({
                "type": "eq",
                "fun": f_constraint,
                "jac": df_constraint,
            })
        elif self.algorithm in {
            Algorithm.COBYLA_SCIPY,
            Algorithm.COBYQA,
            Algorithm.TRUST_CONSTR,
        }:
            self._eq_constraints.append(
                NonlinearConstraint(
                    fun=f_constraint,
                    lb=-tolerance,
                    ub=tolerance,
                    jac=df_constraint,
                )
            )
        else:
            raise OptimisationError(
                f"Algorithm '{self.algorithm}' does not support equality constraints."
            )

    def add_ineq_constraint(
        self,
        f_constraint: OptimiserCallable,
        tolerance: np.ndarray,
        df_constraint: OptimiserCallable | None = None,
    ) -> None:
        r"""
        Add an inequality constrain to the optimiser.

        The constraint is a vector-valued, non-linear, inequality
        constraint of the form :math:`f_{c}(x) \le 0`.

        The constraint function should have the form
        :math:`f(x) \rightarrow y`, where:

            * :math:`x` is a numpy array of the optimisation parameters.
            * :math:`y` is a numpy array containing the values of the
              constraint at :math:`x`, with size :math:`m`, where
              :math:`m` is the dimensionality of the constraint.

        Parameters
        ----------
        f_constraint:
            The constraint function, with form as described above.
        tolerance:
            The tolerances for each optimisation parameter.
        df_constraint:
            The gradient of the constraint function. This should have
            the same form as the constraint function, however its output
            array should have dimensions :math:`m \times n` where
            :math`m` is the dimensionality of the constraint, and
            :math:`n` is the number of optimisation parameters.

        Raises
        ------
        OptimisationError
            Algorithm does not support inequality constraints.

        Notes
        -----
        Inequality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * COBYQA

        """
        if self.algorithm == Algorithm.SLSQP_SCIPY:
            self._ineq_constraints.append({
                "type": "ineq",
                "fun": lambda x, f=f_constraint: -f(x),
                "jac": (lambda x, df=df_constraint: -df(x)) if df_constraint else None,
            })
        elif self.algorithm in {
            Algorithm.COBYLA_SCIPY,
            Algorithm.COBYQA,
            Algorithm.TRUST_CONSTR,
        }:
            self._ineq_constraints.append(
                NonlinearConstraint(  # lb <= fun(x) <= ub
                    fun=lambda x, f=f_constraint: f(x),  # no need to flip
                    lb=-np.inf * np.ones_like(tolerance),
                    ub=tolerance * np.ones_like(tolerance),
                    jac=(lambda x, df=df_constraint: -df(x)) if df_constraint else None,
                )
            )
        else:
            raise OptimisationError(
                f"Algorithm '{self.algorithm}' does not support inequality constraints."
            )

    def optimise(self, x0: np.ndarray | None = None) -> OptimiserResult:
        """
        Run the optimiser.

        Parameters
        ----------
        x0:
            The initial guess for each of the optimisation parameters.
            If not given, each parameter is set to the average of its
            lower and upper bound. If no bounds exist, the initial guess
            will be all zeros.

        Returns
        -------
        The result of the optimisation, containing the optimised
        parameters ``x``, as well as other information about the
        optimisation.

        Raises
        ------
        OptimisationError
            Low-level optimisation error.
        KeyboardInterrupt
            Optimisation halted by user.
        """
        if x0 is None:
            x0 = _initial_guess_from_bounds(self._lower_bounds, self._upper_bounds)

        try:
            result = minimize(
                fun=self.f_objective,
                x0=x0,
                args=(),
                method=SCIPY_ALG_MAPPING[self.algorithm],
                jac=self.df_objective if self.algorithm in DF_SUPPORTED else None,
                hess=None,  # algorithms that use this are not yet implemented
                constraints=self._eq_constraints + self._ineq_constraints,
                bounds=Bounds(lb=self.lower_bounds, ub=self.upper_bounds),
                tol=None,  # ignore - provide specific tolerance in opt_conditions
                options={**self.opt_parameters, **self.opt_conditions},
            )
        except OptVariablesError as err:
            bluemira_warn("Badly behaved numerical gradients are causing trouble...")
            raise OptimisationError(f"SciPy {self.algorithm} failed: {err}") from None
        except RuntimeError as err:
            bluemira_warn(f"RuntimeError during optimisation: {err}")
            raise OptimisationError(f"SciPy {self.algorithm} failed: {err}") from None
        except KeyboardInterrupt:
            raise KeyboardInterrupt(
                "The optimisation was halted by the user. Please check "
                "your optimisation problem and termination conditions."
            ) from None

        process_scipy_result(result, SCIPY_ALG_MAPPING[self.algorithm])
        return OptimiserResult(
            f_x=result.fun,
            x=result.x,
            n_evals=result.nit if hasattr(result, "nit") else result.nfev,
            history=None,
            constraint_history=None,
        )

    def set_lower_bounds(self, bounds: np.ndarray) -> None:
        """
        Set the lower bound for each optimisation parameter.

        Set to `-np.inf` to unbound the parameter's minimum.

        Raises
        ------
        ValueError
            Incorrect bounds dimensions.
        """
        _check_bounds(self.n_variables, bounds)
        self._lower_bounds = bounds

    def set_upper_bounds(self, bounds: np.ndarray) -> None:
        """
        Set the upper bound for each optimisation parameter.

        Set to `np.inf` to unbound the parameter's minimum.

        Raises
        ------
        ValueError
            Incorrect bounds dimensions.
        """
        _check_bounds(self.n_variables, bounds)
        self._upper_bounds = bounds

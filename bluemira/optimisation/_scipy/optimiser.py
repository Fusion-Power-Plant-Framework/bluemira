# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Scipy optimisation interface"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from numpy import clip
from scipy.optimize import Bounds, minimize

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation._algorithm import Algorithm, AlgorithmType
from bluemira.optimisation._optimiser import Optimiser, OptimiserResult
from bluemira.optimisation._scipy.conditions import ScipyConditions
from bluemira.optimisation._tools import _initial_guess_from_bounds, process_scipy_result
from bluemira.optimisation.error import OptimisationError
from bluemira.utilities.error import OptVariablesError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from bluemira.optimisation.typing import ObjectiveCallable, OptimiserCallable


SCIPY_ALG_MAPPING = {
    Algorithm.BFGS_SCIPY: "BFGS",
    Algorithm.CG: "CG",
    Algorithm.COBYLA_SCIPY: "COBYLA",
    Algorithm.DOGLEG: "DOGLEG",
    Algorithm.L_BFGS_B: "L_BFGS_B",
    Algorithm.NELDER_MEAD: "NELDER_MEAD",
    Algorithm.NEWTON_CG: "NEWTON_CG",
    Algorithm.POWELL: "POWELL",
    Algorithm.SLSQP_SCIPY: "SLSQP",
    Algorithm.TNC: "TNC",
    Algorithm.TRUST_CONSTR: "TRUST_CONSTR",
    Algorithm.TRUST_EXACT: "TRUST_EXACT",
    Algorithm.TRUST_KRYLOV: "TRUST_KRYLOV",
    Algorithm.TRUST_NCG: "TRUST_NCG",
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
        self._set_conditions(algorithm, opt_conditions)
        self.opt_parameters = opt_parameters or {}
        self.keep_history = keep_history
        self.eq_constraint = []
        self.ineq_constraint = []

    @property
    def algorithm(self) -> str:
        """
        Returns
        -------
        :
            the optimiser's algorithm.
        """
        return self._algorithm

    def _set_algorithm(self, alg: AlgorithmType) -> None:
        """Set the optimiser's algorithm."""
        self._algorithm = SCIPY_ALG_MAPPING[Algorithm(alg)]

    def _set_conditions(
        self, alg: AlgorithmType, opt_conditions: Mapping[str, int | float] | None
    ) -> None:
        self._opt_conditions = ScipyConditions(Algorithm(alg), (opt_conditions or {}))

    def _add_constraint(
        self,
        constraint_list: list[dict],
        f_constraint: OptimiserCallable,
        tolerance: np.ndarray,
        df_constraint: OptimiserCallable | None = None,
        ctype: str = "eq",
    ):
        if self.algorithm not in {
            SCIPY_ALG_MAPPING[Algorithm.COBYLA_SCIPY],
            SCIPY_ALG_MAPPING[Algorithm.SLSQP_SCIPY],
            SCIPY_ALG_MAPPING[Algorithm.TRUST_CONSTR],
        }:
            raise OptimisationError(
                f"""Algorithm '{self.algorithm}' does not support {ctype} constraints."""
            )
        constraint_list.append({
            "type": ctype,
            "fun": lambda x, f=f_constraint: -f(x),
            "jac": (lambda x, df=df_constraint: -df(x)) if df_constraint else None,
            "tolerance": tolerance,
        })
        # TO DO: introduce NonlinearConstraint for trust-constr

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

        Notes
        -----
        Inequality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

        """
        self._add_constraint(
            self.eq_constraint, f_constraint, tolerance, df_constraint, "eq"
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

        Notes
        -----
        Inequality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

        """
        self._add_constraint(
            self.ineq_constraint, f_constraint, tolerance, df_constraint, "ineq"
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
        """
        if x0 is None:
            x0 = _initial_guess_from_bounds(self._lower_bounds, self._upper_bounds)

        def safe_obj(x):
            # COBYLA sometimes steps slightly outside normalized range
            x_safe = clip(x, 0.0, 1.0)
            return self.f_objective(x_safe)

        def wrap_constraint(c):
            def safe_constraint(x):
                x_safe = clip(x, 0.0, 1.0)
                return c["fun"](x_safe)

            new_c = c.copy()
            new_c["fun"] = safe_constraint
            return new_c

        try:
            result = minimize(
                fun=safe_obj
                if self.algorithm == SCIPY_ALG_MAPPING[Algorithm.COBYLA_SCIPY]
                else self.f_objective,
                x0=x0,
                args=(),
                method=self.algorithm,
                jac=self.df_objective if self.algorithm in DF_SUPPORTED else None,
                hess=None,
                constraints=[
                    wrap_constraint(c)
                    if self.algorithm == SCIPY_ALG_MAPPING[Algorithm.COBYLA_SCIPY]
                    else c
                    for c in self.ineq_constraint
                ],
                bounds=Bounds(lb=self._lower_bounds, ub=self._upper_bounds),
                tol=None,
                options={**self.opt_parameters, **self._opt_conditions.to_dict()},
            )
        except OptVariablesError:  # TO DO: add specific exceptions and messages
            bluemira_warn("Badly behaved numerical gradients are causing trouble...")

        process_scipy_result(result)
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
        """
        self._lower_bounds = bounds

    def set_upper_bounds(self, bounds: np.ndarray) -> None:
        """
        Set the upper bound for each optimisation parameter.

        Set to `np.inf` to unbound the parameter's minimum.
        """
        self._upper_bounds = bounds

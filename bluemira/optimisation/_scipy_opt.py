# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Scipy optimisation interface"""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum, auto
from inspect import signature
from types import DynamicClassAttribute
from typing import TYPE_CHECKING, Any

from eqdsk.file import dataclass
from scipy.optimize import Bounds, minimize

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation._algorithm import Algorithm, AlgorithmType
from bluemira.optimisation._optimiser import Optimiser, OptimiserResult
from bluemira.optimisation._tools import _initial_guess_from_bounds, process_scipy_result
from bluemira.optimisation.error import OptimisationError
from bluemira.utilities.error import OptVariablesError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from bluemira.optimisation.typing import ObjectiveCallable, OptimiserCallable


class ScipyAlgorithm(Enum):
    NELDER_MEAD = auto()
    POWELL = auto()
    CG = auto()
    BFGS = auto()
    NEWTON_CG = auto()
    L_BFGS_B = auto()
    TNC = auto()
    COBYLA = auto()
    SLSQP = auto()
    TRUST_CONSTR = auto()
    DOGLEG = auto()
    TRUST_NCG = auto()
    TRUST_EXACT = auto()
    TRUST_KRYLOV = auto()

    @DynamicClassAttribute
    def scipy_name(self) -> str:
        return self.name.replace("_", "-").lower()

    @DynamicClassAttribute
    def df_supported(self):
        return self not in {self.NELDER_MEAD, self.POWELL, self.COBYLA}

    @classmethod
    def _missing_(cls, value: str) -> ScipyAlgorithm:
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            raise ValueError(f"No such Algorithm value '{value}'.") from None

    @classmethod
    def from_algorithm(cls, algorithm: AlgorithmType):
        return {
            Algorithm.BFGS: cls.BFGS,
            Algorithm.CG: cls.CG,
            Algorithm.COBYLA: cls.COBYLA,
            Algorithm.DOGLEG: cls.DOGLEG,
            Algorithm.L_BFGS_B: cls.L_BFGS_B,
            Algorithm.NELDER_MEAD: cls.NELDER_MEAD,
            Algorithm.NEWTON_CG: cls.NEWTON_CG,
            Algorithm.POWELL: cls.POWELL,
            Algorithm.SLSQP: cls.SLSQP,
            Algorithm.TNC: cls.TNC,
            Algorithm.TRUST_CONSTR: cls.TRUST_CONSTR,
            Algorithm.TRUST_EXACT: cls.TRUST_EXACT,
            Algorithm.TRUST_KRYLOV: cls.TRUST_KRYLOV,
            Algorithm.TRUST_NCG: cls.TRUST_NCG,
        }[Algorithm(algorithm)]


COBYLA_OPS = ["catol"]
SLSQP_OPS = ["eps", "finite_diff_rel_step"]


@dataclass
class ScipyOptConditions:
    ftol: float | None = None
    maxiter: int | None = None

    def to_dict(self) -> dict[str, float]:
        """
        Return the data in dictionary form.

        Returns
        -------
        :
            A dictionary of optimiser conditions.
        """
        dct = asdict(self)
        if self.ftol is None:
            del dct["ftol"]
            bluemira_warn("Scipy default tolerance in use")
        if self.maxiter is None:
            del dct["maxiter"]
            bluemira_warn("Scipy default maxiter in use")
        return dct

    @classmethod
    def from_kwargs(cls, **kwargs):
        params = set(signature(cls).parameters)

        native_args, new_args = {}, {}
        for name, val in kwargs.items():
            if name in params:
                native_args[name] = val
            else:
                new_args[name] = val

        inst = cls(**native_args)
        for new_name, new_val in new_args.items():
            setattr(inst, new_name, new_val)
        return inst


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
        self.algorithm = ScipyAlgorithm.from_algorithm(algorithm)
        self.n_variables = n_variables
        self.f_objective = f_objective
        self.df_objective = df_objective
        self.opt_conditions = ScipyOptConditions.from_kwargs(**(opt_conditions or {}))
        self.opt_parameters = opt_parameters or {}
        self.keep_history = keep_history
        self.eq_constraint = []
        self.ineq_constraint = []

    def _add_constraint(
        self,
        constraint_list: list[dict],
        f_constraint: OptimiserCallable,
        tolerance: np.ndarray,
        df_constraint: OptimiserCallable | None = None,
        ctype: str = "eq",
    ):
        if self.algorithm not in {
            ScipyAlgorithm.COBYLA,
            ScipyAlgorithm.SLSQP,
            ScipyAlgorithm.TRUST_CONSTR,
        }:
            raise OptimisationError(
                f"Algorithm '{self.algorithm.name}' does not support {
                    ctype
                } constraints."
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

        try:
            result = minimize(
                fun=self.f_objective,
                x0=x0,
                args=(),
                method=self.algorithm.scipy_name,
                jac=self.df_objective if self.algorithm.df_supported else None,
                hess=None,
                constraints=[
                    *(
                        constr.as_dict() if not isinstance(constr, dict) else constr
                        for constr in self.eq_constraint
                    ),
                    *(
                        constr.as_dict() if not isinstance(constr, dict) else constr
                        for constr in self.ineq_constraint
                    ),
                ],
                bounds=Bounds(lb=self._lower_bounds, ub=self._upper_bounds),
                tol=None,
                options={**self.opt_parameters, **self.opt_conditions.to_dict()},
            )
        except OptVariablesError:  # TO DO: add specific exceptions and messages
            bluemira_warn("Badly behaved numerical gradients are causing trouble...")

        process_scipy_result(result)
        return OptimiserResult(
            f_x=result.fun,
            x=result.x,
            n_evals=result.nit,
            history=None,
            constraint_history=None,
        )

    def set_lower_bounds(self, bounds: np.ndarray) -> None:
        """
        Set the lower bound for each optimisation parameter.

        Set to `-np.inf` to unbound the parameter's minimum.
        """
        # TO DO: convert to ineq constraints if bounds not supported for give algorithm
        self._lower_bounds = bounds

    def set_upper_bounds(self, bounds: np.ndarray) -> None:
        """
        Set the upper bound for each optimisation parameter.

        Set to `np.inf` to unbound the parameter's minimum.
        """
        # TO DO: convert to ineq constraints if bounds not supported for give algorithm
        self._upper_bounds = bounds

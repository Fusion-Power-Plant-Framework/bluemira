from __future__ import annotations

from enum import Enum, auto
from types import DynamicClassAttribute
from typing import TYPE_CHECKING, Any

from eqdsk.file import dataclass
from scipy.optimize import Bounds, minimize, show_options

from bluemira.optimisation._algorithm import Algorithm, AlgorithmType
from bluemira.optimisation._optimiser import Optimiser, OptimiserResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from bluemira.optimisation.typing import ObjectiveCallable, OptimiserCallable


@dataclass
class Constraint:
    f_constraint: OptimiserCallable
    tolerance: np.ndarray
    type: str
    df_constraint: OptimiserCallable | None = None


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

    @classmethod
    def _missing_(cls, value: str) -> ScipyAlgorithm:
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            raise ValueError(f"No such Algorithm value '{value}'.") from None


SCIPY_ALG_MAPPING = {
    Algorithm.NELDER_MEAD: ScipyAlgorithm.NELDER_MEAD,
    Algorithm.POWELL: ScipyAlgorithm.POWELL,
    Algorithm.CG: ScipyAlgorithm.CG,
    Algorithm.BFGS_SCIPY: ScipyAlgorithm.BFGS,
    Algorithm.NEWTON_CG: ScipyAlgorithm.NEWTON_CG,
    Algorithm.L_BFGS_B: ScipyAlgorithm.L_BFGS_B,
    Algorithm.TNC: ScipyAlgorithm.TNC,
    Algorithm.COBYLA_SCIPY: ScipyAlgorithm.COBYLA,
    Algorithm.SLSQP_SCIPY: ScipyAlgorithm.SLSQP,
    Algorithm.TRUST_CONSTR: ScipyAlgorithm.TRUST_CONSTR,
    Algorithm.DOGLEG: ScipyAlgorithm.DOGLEG,
    Algorithm.TRUST_NCG: ScipyAlgorithm.TRUST_NCG,
    Algorithm.TRUST_EXACT: ScipyAlgorithm.TRUST_EXACT,
    Algorithm.TRUST_KRYLOV: ScipyAlgorithm.TRUST_KRYLOV,
}


COBYLA_OPS = ["tol", "catol", "maxiter"]
SLSQP_OPS = ["ftol", "eps", "maxiter", "finite_diff_rel_step"]

show_options


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
        self.algorithm = ScipyAlgorithm(SCIPY_ALG_MAPPING[algorithm])
        self.n_variables = n_variables
        self.f_objective = f_objective
        self.df_objective = df_objective
        self.opt_conditions = opt_conditions or {}
        self.opt_parameters = opt_parameters or {}
        self.keep_history = keep_history
        self.eq_constraint = []
        self.ineq_constraint = []

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
        self.eq_constraint.append(
            Constraint(f_constraint, tolerance, "eq", df_constraint)
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
        self.ineq_constraint.append(
            Constraint(f_constraint, tolerance, "ineq", df_constraint)
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
        result = minimize(
            fun=self.f_objective,
            x0=x0,
            method=self.algorithm.scipy_name,
            jac=self.df_objective,
            bounds=Bounds(lb=self._lower_bounds, ub=self._upper_bounds),
            constraints=[
                *(constr.as_dict() for constr in self.eq_constraint),
                *(constr.as_dict() for constr in self.ineq_constraint),
            ],
            options={**self.opt_parameters, **self.opt_conditions},
        )
        return OptimiserResult(
            f_x=result.fun,
            x=result.x,
            n_evals=result.nit,
            history=None,
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

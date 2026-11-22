# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Debugging optimisation interface"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eqdsk.file import dataclass

from bluemira.optimisation._algorithm import Algorithm, AlgorithmType
from bluemira.optimisation._nlopt.optimiser import NLOPT_ALG_MAPPING, NloptOptimiser
from bluemira.optimisation._optimiser import Optimiser, OptimiserResult
from bluemira.optimisation._scipy_opt import ScipyAlgorithm, ScipyOptimiser
from bluemira.optimisation.error import OptimisationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from bluemira.optimisation.typing import ObjectiveCallable, OptimiserCallable


@dataclass
class Constraint:
    f_constraint: OptimiserCallable
    tolerance: np.ndarray
    df_constraint: OptimiserCallable | None = None


class DebugOptimiser(Optimiser):
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
        self.algorithm = Algorithm(algorithm)
        self.n_variables = n_variables
        self.f_objective = f_objective
        self.df_objective = df_objective
        self.opt_conditions = opt_conditions
        self.opt_parameters = opt_parameters
        self.keep_history = keep_history
        self.eq_constraint = []
        self.ineq_constraint = []

        if self.algorithm in NLOPT_ALG_MAPPING:
            optimiser = NloptOptimiser
        else:
            try:
                ScipyAlgorithm(self.algorithm)
            except KeyError:
                raise OptimisationError("Unknown algorithm") from None
            else:
                optimiser = ScipyOptimiser

        self.opt = optimiser(
            self.algorithm,
            self.n_variables,
            self.f_objective,
            self.df_objective,
            self.opt_conditions,
            self.opt_parameters,
            keep_history=self.keep_history,
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

        Notes
        -----
        Inequality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

        """
        self.opt.add_eq_constraint(f_constraint, tolerance, df_constraint)
        self.eq_constraint.append(Constraint(f_constraint, tolerance, df_constraint))

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
        self.opt.add_ineq_constraint(f_constraint, tolerance, df_constraint)
        self.ineq_constraint.append(Constraint(f_constraint, tolerance, df_constraint))

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
        self.opt_in = OptimiserResult(
            f_x=0, x=x0, n_evals=0, history=[], constraint_history=[]
        )
        self.opt_result = self.opt.optimise(x0)
        import pprint

        pprint.pprint(self.__dict__)

        # ipdb.set_trace()
        return self.opt_result

    def set_lower_bounds(self, bounds: np.ndarray) -> None:
        """
        Set the lower bound for each optimisation parameter.

        Set to `-np.inf` to unbound the parameter's minimum.
        """
        self.opt.set_lower_bounds(bounds)
        self.lower_bounds = bounds

    def set_upper_bounds(self, bounds: np.ndarray) -> None:
        """
        Set the upper bound for each optimisation parameter.

        Set to `np.inf` to unbound the parameter's minimum.
        """
        self.opt.set_upper_bounds(bounds)
        self.upper_bounds = bounds

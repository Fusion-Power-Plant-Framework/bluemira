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
"""Defines the interface for an Optimiser."""

import abc
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from bluemira.optimisation._typing import OptimiserCallable


@dataclass
class OptimiserResult:
    """Container for optimiser results."""

    x: np.ndarray
    n_evals: int
    history: List[np.ndarray] = field(repr=False)
    # TODO(hsaunders1904): add 'converged' property that's true if converged within tol


class Optimiser(abc.ABC):
    """Interface for an optimiser supporting bounds and constraints."""

    @abc.abstractmethod
    def add_eq_constraint(
        self,
        f_constraint: OptimiserCallable,
        tolerance: np.ndarray,
        df_constraint: Optional[OptimiserCallable] = None,
    ) -> None:
        """
        Add an equality constraint to the optimiser.

        The constraint is a vector-valued, non-linear, equality
        constraint of the form $h(x) = 0$.

        The constraint function must have the form `f(x) -> y`,
        where:

            * `x` is a numpy array of the optimisation parameters.
            * `y` is a numpy array containing the values of the
              constraint at `x`, with size $m$, where $m$ is the
              dimensionality of the constraint.

        Parameters
        ----------
        f_constraint: Callable[[Arg(np.ndarray, 'x')], np.ndarray]
            The constraint function, with form as described above.
        tolerance: np.ndarray
            The tolerances for each optimisation parameter.
        df_constraint: Optional[Callable[[Arg(np.ndarray, 'x')], np.ndarray]]
            The gradient of the constraint function. This should have the
            same form as the constraint function, however its output
            array should have dimensions `m x n_variables` where `m` is
            the dimensionality of the constraint.

        Notes
        -----
        Equality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

        """

    @abc.abstractmethod
    def add_ineq_constraint(
        self,
        f_constraint: OptimiserCallable,
        tolerance: np.ndarray,
        df_constraint: Optional[OptimiserCallable] = None,
    ) -> None:
        r"""
        Add an inequality constrain to the optimiser.

        The constraint is a vector-valued, non-linear, inequality
        constraint of the form $f_{c}(x) \le 0$.

        The constraint function should have the form `f(x) -> y`,
        where:

            * `x` is a numpy array of the optimisation parameters.
            * `y` is a numpy array containing the values of the
              constraint at `x`, with size $m$, where $m$ is the
              dimensionality of the constraint.

        Parameters
        ----------
        f_constraint: Callable[[Arg(np.ndarray, 'x')], np.ndarray]
            The constraint function, with form as described above.
        tolerance: Union[float, np.ndarray]
            The tolerances for each optimisation parameter.
        df_constraint: Optional[Callable[[Arg(np.ndarray, 'x')], np.ndarray]]
            The gradient of the constraint function. This should have the
            same form as the constraint function, however its output
            array should have dimensions `m x n_variables` where `m` is
            the dimensionality of the constraint.

        Notes
        -----
        Inequality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

        """

    @abc.abstractmethod
    def optimise(self, x0: Optional[np.ndarray] = None) -> OptimiserResult:
        """
        Run the optimiser.

        Parameters
        ----------
        x0: Optional[np.ndarray]
            The initial guess for each of the optimisation parameters.
            If not given, each parameter is set to the average of its
            lower and upper bound. If no bounds exist, the initial guess
            will be all zeros.

        Returns
        -------
        result: OptimiserResult
            The result of the optimisation, containing the optimised
            parameters `x`, as well as other information about the
            optimisation.
        """

    @abc.abstractmethod
    def set_lower_bounds(self, bounds: np.ndarray) -> None:
        """
        Set the lower bound for each optimisation parameter.

        Set to `-np.inf` to unbound the parameter's minimum.
        """

    @abc.abstractmethod
    def set_upper_bounds(self, bounds: np.ndarray) -> None:
        """
        Set the upper bound for each optimisation parameter.

        Set to `np.inf` to unbound the parameter's minimum.
        """

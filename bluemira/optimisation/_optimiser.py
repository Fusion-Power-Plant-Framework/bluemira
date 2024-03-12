# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Defines the interface for an Optimiser."""

import abc
from dataclasses import dataclass, field

import numpy as np

from bluemira.optimisation.typing import OptimiserCallable


@dataclass
class OptimiserResult:
    """Container for optimiser results."""

    f_x: float
    """The evaluation of the optimised parameterisation."""
    x: np.ndarray
    """The optimised parameterisation."""
    n_evals: int
    """The number of evaluations of the objective function in the optimisation."""
    history: list[tuple[np.ndarray, float]] = field(repr=False)
    """
    The history of the parametrisation at each iteration.

    The first element of each tuple is the parameterisation (x), the
    second is the evaluation of the objective function at x (f(x)).
    """
    constraints_satisfied: bool | None = None
    """
    Whether all constraints have been satisfied to within the required tolerance.

    Is ``None`` if constraints have not been checked.
    """


class Optimiser(abc.ABC):
    """Interface for an optimiser supporting bounds and constraints."""

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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

# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import enum
from collections.abc import Callable

import numpy as np

from bluemira.optimisation._tools import approx_derivative
from bluemira.optimisation.typing import ObjectiveCallable, OptimiserCallable

_FloatOrArrayT = np.ndarray | float


class ConstraintType(enum.Enum):
    """Enumeration of constraint types."""

    EQUALITY = enum.auto()
    INEQUALITY = enum.auto()


class _NloptFunction:
    """
    Base class for NLOpt objective/constraint functions.

    Implements the ability to calculate a numerical gradient, if an
    analytical one is not provided.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], _FloatOrArrayT],
        bounds: tuple[_FloatOrArrayT, _FloatOrArrayT],
    ):
        self.f = f
        self.f0: float | np.ndarray = 0
        self.bounds = bounds

    def _approx_derivative(self, x: np.ndarray) -> np.ndarray:
        return approx_derivative(self.f, x, bounds=self.bounds, f0=self.f0)

    def set_approx_derivative_lower_bound(self, lower_bound: np.ndarray) -> None:
        """Set the lower bounds for use in derivative approximation."""
        self.bounds = (lower_bound, self.bounds[1])

    def set_approx_derivative_upper_bound(self, upper_bound: np.ndarray) -> None:
        """Set the upper bounds for use in derivative approximation."""
        self.bounds = (self.bounds[0], upper_bound)


class ObjectiveFunction(_NloptFunction):
    """
    Holds an objective function for an NLOpt optimiser.

    Adapts the given objective function, and optional derivative, to a
    form understood by NLOpt.

    If no optimiser derivative is given, and the algorithm is gradient
    based, a numerical approximation of the gradient is calculated.
    """

    f: ObjectiveCallable
    f0: float

    def __init__(
        self,
        f: ObjectiveCallable,
        df: OptimiserCallable | None,
        n_variables: int,
        bounds: tuple[_FloatOrArrayT, _FloatOrArrayT] = (-np.inf, np.inf),
    ):
        super().__init__(f, bounds)
        self.df = df if df is not None else self._approx_derivative
        self.history: list[tuple[np.ndarray, float]] = []
        self.prev_iter = np.zeros(n_variables, dtype=float)

    def call(self, x: np.ndarray, grad: np.ndarray) -> float:
        """Execute the NLOpt objective function."""
        if not np.any(np.isnan(x)):
            self._store_x(x)
        return self._call_inner(x, grad)

    def call_with_history(self, x: np.ndarray, grad: np.ndarray) -> float:
        """Execute the NLOpt objective function, recording the iteration history."""
        f_x = self._call_inner(x, grad)
        self.history.append((np.copy(x), f_x))
        self.prev_iter = self.history[-1][0]
        return f_x

    def _call_inner(self, x: np.ndarray, grad: np.ndarray) -> float:
        """Execute the objective function in the form required by NLOpt."""
        # Cache f(x) so we do not need to recalculate it if we're using
        # an approximate gradient
        self.f0 = self.f(x)
        if grad.size > 0:
            grad[:] = self.df(x)
        return self.f0

    def _store_x(self, x: np.ndarray) -> None:
        """Store ``x`` in ``self.prev_iter``."""
        # Assign in place to avoid lots of allocations.
        # Not benchmarked, but may be more efficient...
        self.prev_iter[:] = x


class Constraint(_NloptFunction):
    """Holder for NLOpt constraint functions."""

    f: OptimiserCallable
    f0: np.ndarray

    def __init__(
        self,
        constraint_type: ConstraintType,
        f: OptimiserCallable,
        tolerance: np.ndarray,
        df: OptimiserCallable | None = None,
        bounds: tuple[_FloatOrArrayT, _FloatOrArrayT] = (-np.inf, np.inf),
        reflection_matrix: np.ndarray | None = None,
    ):
        super().__init__(f, bounds)
        self.constraint_type = constraint_type
        self.tolerance = tolerance
        self.df = df if df is not None else self._approx_derivative
        self.reflection_matrix = reflection_matrix

    def call(self, result: np.ndarray, x: np.ndarray, grad: np.ndarray) -> None:
        """
        Execute the constraint function in the form required by NLOpt.

        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#vector-valued-constraints
        """
        # Cache f(x) so we do not need to recalculate it if we're using
        # an approximate gradient
        # TODO: here
        if self.reflection_matrix is not None:
            # x = self.reflection_matrix @ x
            pass +
        result[:] = self.f(x)
        self.f0 = result
        if grad.size > 0:
            grad[:] = self.df(x)

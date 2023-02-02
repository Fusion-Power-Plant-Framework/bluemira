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

from typing import List, Optional, Tuple, Union

import numpy as np

from bluemira.optimisation._tools import approx_derivative
from bluemira.optimisation._typing import ObjectiveCallable, OptimiserCallable


class NloptObjectiveFunction:
    """
    Holds an objective function for an NLOpt optimiser.

    Adapts the given objective function, and optional derivative, to a
    form understood by NLOpt.

    If no optimiser derivative is given, and the algorithm is gradient
    based, a numerical approximation of the gradient is calculated.
    """

    def __init__(
        self,
        f: ObjectiveCallable,
        df: Optional[OptimiserCallable] = None,
        bounds: Tuple[Union[np.ndarray, float], Union[np.ndarray, float]] = (
            -np.inf,
            np.inf,
        ),
    ):
        self.f = f
        self.df = df if df is not None else self._approx_derivative
        self.bounds = bounds
        self.history: List[np.ndarray] = []

    def call(self, x: np.ndarray, grad: np.ndarray) -> float:
        """Execute the NLOpt objective function."""
        if grad.size > 0:
            grad[:] = self.df(x)
        return self.f(x)

    def call_with_history(self, x: np.ndarray, grad: np.ndarray) -> float:
        """Execute the NLOpt objective function, recording the iteration history."""
        self.history.append(np.copy(x))
        if grad.size > 0:
            grad[:] = self.df(x)
        return self.f(x)

    def _approx_derivative(self, x: np.ndarray) -> np.ndarray:
        return approx_derivative(self.f, x, bounds=self.bounds)

    def set_approx_derivative_lower_bound(self, lower_bound: np.ndarray) -> None:
        """Set the lower bounds for use in derivative approximation."""
        self.bounds = (lower_bound, self.bounds[1])

    def set_approx_derivative_upper_bound(self, upper_bound: np.ndarray) -> None:
        """Set the upper bounds for use in derivative approximation."""
        self.bounds = (self.bounds[0], upper_bound)

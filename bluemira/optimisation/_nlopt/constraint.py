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
import enum
from typing import Optional, Tuple, Union

import numpy as np

from bluemira.optimisation._tools import approx_derivative
from bluemira.optimisation._typing import OptimiserCallable


class ConstraintType(enum.Enum):
    """Enumeration of constraint types."""

    EQUALITY = enum.auto()
    INEQUALITY = enum.auto()


class Constraint:
    """Holder for NLOpt constraint functions."""

    def __init__(
        self,
        constraint_type: ConstraintType,
        f: OptimiserCallable,
        tolerance: np.ndarray,
        df: Optional[OptimiserCallable] = None,
        bounds: Tuple[Union[np.ndarray, float], Union[np.ndarray, float]] = (
            -np.inf,
            np.inf,
        ),
    ):
        self.constraint_type = constraint_type
        self.f = f
        self.tolerance = tolerance
        self.df = df if df is not None else self._approx_derivative
        self.bounds = bounds

    def nlopt_call(self, result: np.ndarray, x: np.ndarray, grad: np.ndarray) -> None:
        """
        Execute the constraint function in the form required by NLOpt.

        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#vector-valued-constraints
        """
        if grad.size > 0:
            grad[:] = self.df(x)
        result[:] = self.f(x)

    def _approx_derivative(self, x: np.ndarray) -> np.ndarray:
        return approx_derivative(self.f, x, bounds=self.bounds)

    def set_approx_derivative_lower_bound(self, lower_bound: np.ndarray) -> None:
        """Set the lower bounds for use in derivative approximation."""
        self.bounds = (lower_bound, self.bounds[1])

    def set_approx_derivative_upper_bound(self, upper_bound: np.ndarray) -> None:
        """Set the upper bounds for use in derivative approximation."""
        self.bounds = (self.bounds[0], upper_bound)

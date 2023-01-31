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
"""Collection of utility functions for the module."""

from typing import Callable, Iterable, Optional, Union

import numpy as np
from scipy.optimize._numdiff import approx_derivative as _approx_derivative


def approx_derivative(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    method: str = "3-point",
    rel_step: Optional[Union[float, np.ndarray]] = None,
    f0: Optional[Union[float, np.ndarray]] = None,
    bounds: Optional[Iterable[Union[float, np.ndarray]]] = (-np.inf, np.inf),
    args=(),
) -> np.ndarray:
    """
    Approximate the gradient of a function about a point.

    Parameters
    ----------
    func: Callable
        Function for which to calculate the gradient.
    x0: np.ndarray
        Point about which to calculate the gradient.
    method: str
        Finite difference method to use.
    rel_step: Optional[float, np.ndarray]
        Relative step size to use.
    f0: Optional[float, np.ndarray]
        Result of func(x0). If None, this is recomputed.
    bounds: Optional[Iterable]
        Lower and upper bounds on individual variables.
    args: tuple
        Additional arguments to func.
    """
    return _approx_derivative(
        func, x0, method=method, rel_step=rel_step, f0=f0, bounds=bounds, args=args
    )

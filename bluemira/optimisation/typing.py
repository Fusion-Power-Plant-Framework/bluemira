# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Types for the optimisation module."""

from typing import Optional, Protocol, TypedDict

import numpy as np
from typing_extensions import NotRequired


class ObjectiveCallable(Protocol):
    """Form for an optimiser objective function."""

    def __call__(self, x: np.ndarray) -> float:
        """
        Call the objective function.

        Parameters
        ----------
        x:
            The optimisation parameters.
        """
        ...


class OptimiserCallable(Protocol):
    """
    Form for an non-objective optimiser function.

    This is the form for a gradient, constraint, or constraint gradient.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Call the optimiser function.

        Parameters
        ----------
        x:
            The optimisation parameters.
        """
        ...


class ConstraintT(TypedDict):
    """Typing for definition of a constraint."""

    f_constraint: OptimiserCallable
    tolerance: np.ndarray
    df_constraint: NotRequired[Optional[OptimiserCallable]]

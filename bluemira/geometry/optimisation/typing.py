# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Typing for the geometry optimisation module"""

from typing import Optional, Protocol, TypedDict

import numpy as np
from typing_extensions import NotRequired

from bluemira.geometry.parameterisations import GeometryParameterisation


class GeomOptimiserObjective(Protocol):
    """Form for a geometry optimisation objective function."""

    def __call__(self, geom: GeometryParameterisation) -> float:
        """Call the geometry optimiser objective function."""
        ...


class GeomOptimiserCallable(Protocol):
    """Form for a geometry optimiser function (derivative, constraint, etc.)."""

    def __call__(self, geom: GeometryParameterisation) -> np.ndarray:
        """Call the geometry optimiser function."""
        ...


class GeomClsOptimiserCallable(Protocol):
    """Form for a geometry optimiser function (derivative, constraint, etc.)."""

    def __call__(self) -> np.ndarray:
        """Call the geometry optimiser function."""
        ...


class GeomConstraintT(TypedDict):
    """Typing for definition of a constraint."""

    f_constraint: GeomOptimiserCallable
    tolerance: np.ndarray
    df_constraint: NotRequired[Optional[GeomOptimiserCallable]]

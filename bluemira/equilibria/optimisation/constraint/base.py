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
import abc
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium


# TODO(hsaunders1904): should probably move this to optimisation module
class Constraint(abc.ABC):
    @abc.abstractmethod
    def f_constraint(self, x: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def df_constraint(self, x: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    @abc.abstractmethod
    def tolerance(self) -> npt.NDArray:
        raise NotImplementedError


class CoilSetConstraint(abc.ABC):
    @abc.abstractmethod
    def control_response(self, coilset: CoilSet) -> npt.NDArray:
        pass

    @abc.abstractmethod
    def evaluate(self) -> npt.NDArray:
        pass

    @abc.abstractmethod
    def constraint(self, coilset: CoilSet) -> Constraint:
        """The constraint object to pass to the optimiser."""

    @abc.abstractproperty
    def length(self) -> int:
        """The length of the constraint vector."""

    @abc.abstractproperty
    def weights(self) -> npt.NDArray:
        """The weights for each element in the constraint array."""

    @property
    def constraint_target(self) -> float:
        """The target value for the constraint."""
        return 0.0


class CoilSetConstraintSet:
    r"""Wrapper around a list of :class:`.CoilSetConstraint`\s"""

    def __init__(self, constraints: List[CoilSetConstraint]) -> None:
        self._constraints = constraints

    def get_weighted_arrays(
        self, coilset: CoilSet, eq: Equilibrium
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        weights = self.weight_matrix()
        weighted_a = weights[:, np.newaxis] * self.control_matrix(coilset)
        weighted_b = weights * self.b(eq)
        return weights, weighted_a, weighted_b

    def b(self, eq: Equilibrium) -> npt.NDArray:
        return self.target(eq) - self.background(eq)

    @property
    def constraint_length(self) -> int:
        """The cumulative size of the constraint set."""
        return sum(c.length for c in self._constraints)

    def weight_matrix(self) -> npt.NDArray:
        """
        Build the weight matrix used in an optimisation.

        This is assumed to be diagonal.
        """
        if not self._constraints:
            return np.array([])
        # TODO(hsaunders1904): how can this be diagonal if it's 1D?
        return np.concatenate([c.weights for c in self._constraints])

    def control_matrix(self, coilset: CoilSet) -> npt.NDArray:
        """Build the control response matrix used in optimisation."""
        if not self._constraints:
            return np.array([])
        return np.vstack([c.control_response(coilset) for c in self._constraints])

    def target(self, eq: Equilibrium) -> npt.NDArray:
        """The constraint target value vector."""
        if not self._constraints:
            return np.array([])
        return np.concatenate(
            [np.full(c.length, c.constraint_target) for c in self._constraints]
        )

    def background(self, eq: Equilibrium) -> npt.NDArray:
        """The background value vector."""
        if not self._constraints:
            return np.array([])
        return np.concatenate([c.evaluate() for c in self._constraints])

    def is_empty(self) -> bool:
        """Return ``True`` if the constraint set contains no constraints."""
        return len(self._constraints) == 0

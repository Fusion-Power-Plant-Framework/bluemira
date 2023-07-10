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
from typing import List

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


class L2NormConstraint(Constraint):
    def __init__(
        self,
        tolerance: npt.NDArray,
        a_mat: npt.NDArray,
        b_vec: npt.NDArray,
        scale: float,
        target_value: float,
    ) -> None:
        self._tolerance = tolerance
        self.a = a_mat
        self.b = b_vec
        self.scale = scale
        self.target_value = target_value

    def f_constraint(self, x: npt.NDArray) -> npt.NDArray:
        vector = self.scale * x
        residual = self.a @ vector - self.b
        constraint = residual.T @ residual - self.target_value
        return constraint

    def df_constraint(self, x: npt.NDArray) -> npt.NDArray:
        df_c = 2 * self.scale * (self.a.T @ self.a @ x - self.a.T @ self.b)
        return df_c

    def tolerance(self) -> npt.NDArray:
        return self._tolerance


class CoilSetConstraint(abc.ABC):
    @abc.abstractmethod
    def control_response(self, coilset: CoilSet):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def constraint_target(self) -> float:
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


class CoilSetConstraintSet:
    r"""Wrapper around a list of :class:`.CoilSetConstraint`\s"""

    def __init__(self, constraints: List[CoilSetConstraint]) -> None:
        self._constraints = constraints

    def get_weighted_arrays(self, coilset: CoilSet, eq: Equilibrium):
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
        # TODO(hsaunders1904): how can this be diagonal if it's 1D?
        return np.concatenate([c.weights for c in self._constraints])

    def control_matrix(self, coilset: CoilSet) -> npt.NDArray:
        """Build the control response matrix used in optimisation."""
        return np.vstack([c.control_response(coilset) for c in self._constraints])

    def target(self, eq: Equilibrium) -> npt.NDArray:
        """The constraint target value vector."""
        return np.concatenate(
            [np.full(c.length, c.constraint_target()) for c in self._constraints]
        )

    def background(self, eq: Equilibrium) -> npt.NDArray:
        """The background value vector."""
        return np.concatenate([c.evaluate() for c in self._constraints])


class IsofluxConstraint(CoilSetConstraint):
    def __init__(
        self,
        x: npt.ArrayLike,
        z: npt.ArrayLike,
        ref_x: float,
        ref_z: float,
        eq: Equilibrium,
        constraint_value: float = 0.0,
        weights: npt.ArrayLike = 1.0,
        tolerance: float = 1e-6,
    ):
        self.x = np.atleast_1d(x)
        self.z = np.atleast_1d(z)
        self.ref_x = ref_x
        self.ref_z = ref_z
        self.eq = eq
        self.constraint_value = constraint_value
        self._weights = (
            np.full_like(self.x, weights)
            if np.isscalar(weights)
            else np.atleast_1d(weights)
        )
        self.tolerance = np.atleast_1d(tolerance)
        # TODO: validate x, z and weights have equal length

    def control_response(self, coilset: CoilSet):
        return coilset.psi_response(self.x, self.z, control=True) - coilset.psi_response(
            self.ref_x, self.ref_z, control=True
        )

    def evaluate(self, I_not_dI: bool = True) -> npt.NDArray:
        if I_not_dI:
            return np.atleast_1d(self.eq.plasma.psi(self.x, self.z))
        return np.atleast_1d(self.eq.psi(self.x, self.z))

    def constraint_target(self, I_not_dI: bool = True) -> float:
        if I_not_dI:
            return float(self.eq.plasma.psi(self.ref_x, self.ref_z))
        return float(self.eq.psi(self.ref_x, self.ref_z))

    @property
    def length(self) -> int:
        """The size of the constraint vector."""
        return len(self.x) if hasattr(self.x, "__len__") else 1

    @property
    def weights(self) -> npt.NDArray:
        return self._weights

    def constraint(self, coilset: CoilSet) -> Constraint:
        a_mat = self.control_response(coilset)
        b_vec = self.constraint_target() - self.evaluate()
        return L2NormConstraint(
            a_mat=a_mat,
            b_vec=b_vec,
            target_value=0,
            tolerance=self.tolerance,
            scale=1,
        )

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
import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraint.base import (
    CoilSetConstraint,
    Constraint,
)
from bluemira.equilibria.optimisation.constraint.functions import L2NormConstraint


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
        self.tolerance = (
            np.full_like(self.x, tolerance)
            if np.isscalar(tolerance)
            else np.atleast_1d(tolerance)
        )
        if shapes := self._arrays_not_1d_and_equal():
            raise ValueError(
                "x, z, weights, and tolerance must be 1D and have equal length. "
                f"Found shapes {str(shapes)[1:-1]}."
            )

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

    def _arrays_not_1d_and_equal(self):
        shapes = [a.shape for a in [self.x, self.z, self.weights, self.tolerance]]
        shapes_equal = shapes.count(shapes[0]) == len(shapes)
        if shapes_equal and np.ndim(self.x) == 1:
            return []
        return shapes

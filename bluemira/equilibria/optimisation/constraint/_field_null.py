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
from typing import Union

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraint.base import (
    CoilSetConstraint,
    Constraint,
)
from bluemira.equilibria.optimisation.constraint.functions import (
    L2NormConstraint,
    make_constraint,
)


class FieldNullConstraint(CoilSetConstraint):
    """
    Magnetic field null constraint.

    In practice, this sets the Bx and Bz field components to be 0 at the
    specified location.
    """

    def __init__(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        eq: Equilibrium,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance: float = 1e-6,
        constraint_cls=L2NormConstraint,
    ) -> None:
        self.x = np.atleast_1d(x)
        self.z = np.atleast_1d(z)
        self.eq = eq
        self._weights = (
            np.full(2, weights) if np.isscalar(weights) else np.atleast_1d(weights)
        )
        self.tolerance = (
            np.full_like(self.x, tolerance)
            if np.isscalar(tolerance)
            else np.atleast_1d(tolerance)
        )
        self._constraint_cls = constraint_cls

    def control_response(self, coilset: CoilSet) -> npt.NDArray:
        """Calculate control response of a CoilSet to the constraint."""
        return np.vstack(
            [
                coilset.Bx_response(self.x, self.z, control=True),
                coilset.Bz_response(self.x, self.z, control=True),
            ]
        )

    def evaluate(self, I_not_dI: bool = True) -> np.ndarray:
        """Calculate the value of the constraint in an Equilibrium."""
        if I_not_dI:
            return np.concatenate(
                [self.eq.plasma.Bx(self.x, self.z), self.eq.plasma.Bz(self.x, self.z)]
            )
        return np.concatenate([self.eq.Bx(self.x, self.z), self.eq.Bz(self.x, self.z)])

    def constraint(self, coilset: CoilSet) -> Constraint:
        a_mat = self.control_response(coilset)
        b_vec = self.constraint_target() - self.evaluate()
        return make_constraint(
            constraint_cls=self._constraint_cls,
            tolerance=self.tolerance,
            a_mat=a_mat,
            b_vec=b_vec,
            scale=1.0,  # TODO
            target_value=self.constraint_target(),
        )

    @property
    def weights(self) -> npt.NDArray:
        return self._weights

    @property
    def length(self) -> int:
        return 2

    def constraint_target(self) -> float:
        return 0.0

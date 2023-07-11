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
from typing import Type, TypeVar

import numpy.typing as npt

from bluemira.equilibria.optimisation.constraint.base import Constraint

_ConstraintT = TypeVar("_ConstraintT", bound=Constraint)


def make_constraint(
    constraint_cls: Type[_ConstraintT],
    tolerance: npt.NDArray,
    a_mat: npt.NDArray,
    b_vec: npt.NDArray,
    scale: float,
    target_value: float,
) -> _ConstraintT:
    """Factory function to initialise a numerical constraint function class."""
    return constraint_cls(
        tolerance=tolerance,
        a_mat=a_mat,
        b_vec=b_vec,
        scale=scale,
        target_value=target_value,
    )


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

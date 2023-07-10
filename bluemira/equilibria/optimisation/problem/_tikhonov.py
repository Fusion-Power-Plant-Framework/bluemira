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
from typing import List

import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraint import (
    CoilSetConstraint,
    CoilSetConstraintSet,
)
from bluemira.equilibria.optimisation.problem.base import CoilSetOptimisationProblem


class TikhonovCurrentCOP(CoilSetOptimisationProblem):
    hack_ctr = 0

    def __init__(
        self,
        eq: Equilibrium,
        coilset: CoilSet,
        targets: List[CoilSetConstraint],
        gamma: float,
    ):
        self.gamma = gamma
        self.eq = eq
        self.coilset = coilset
        self._targets = targets
        self.a_mat, self.b_vec = self.get_a_mat_b_vec()
        print(f"{self.a_mat=}, {self.b_vec=}")
        self.hack_ctr += 1

    def pre_optimise(self):
        self.a_mat, self.b_vec = self.get_a_mat_b_vec()

    def objective(self, coilset) -> float:
        from bluemira.equilibria.opt_objectives import regularised_lsq_fom

        x = self.read_state(coilset)[-11:]  # TODO(hsaunders1904): normalize/scaling
        print(f"{x=}")
        a_mat, b_vec = self.a_mat, self.b_vec
        fom = regularised_lsq_fom(x, a_mat, b_vec, self.gamma)[0]
        print(f"{fom=}")
        return fom

    def df_objective(self, coilset) -> npt.NDArray:
        x = self.read_state(coilset)[-11:]
        a_mat, b_vec = self.a_mat, self.b_vec
        jac = 2 * a_mat.T @ a_mat @ x / len(b_vec)
        jac -= 2 * a_mat.T @ b_vec / len(b_vec)
        jac += 2 * self.gamma * self.gamma * x
        jac *= 1e6
        print(f"{jac=}")
        return jac

    def lower_bounds(self, coilset: CoilSet) -> npt.NDArray:
        return -self.upper_bounds(coilset)

    def upper_bounds(self, coilset: CoilSet) -> npt.NDArray:
        return self.bounds_of_currents(coilset, coilset.get_max_current())

    def get_a_mat_b_vec(self):
        constraint_set = self.constraints()
        return constraint_set.get_weighted_arrays(self.coilset, self.eq)[1:]

    def constraints(self) -> CoilSetConstraintSet:
        return CoilSetConstraintSet(self._targets)

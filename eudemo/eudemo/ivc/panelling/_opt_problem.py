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

from bluemira.utilities.opt_problems import (
    OptimisationConstraint,
    OptimisationObjective,
    OptimisationProblem,
    Optimiser,
)
from bluemira.utilities.optimiser import approx_derivative
from eudemo.ivc.panelling._paneller import Paneller


class PanellingOptProblem(OptimisationProblem):
    def __init__(self, paneller: Paneller, optimiser: Optimiser):
        self.paneller = paneller
        self.bounds = (np.zeros_like(self.paneller.x0), np.ones_like(self.paneller.x0))
        objective = OptimisationObjective(self.objective)
        constraint = OptimisationConstraint(
            self.constrain_min_length_and_angles,
            f_constraint_args={},
            tolerance=np.full(self.paneller.n_constraints, 1e-5),
        )
        super().__init__(self.paneller.x0, optimiser, objective, [constraint])
        self.set_up_optimiser(self.paneller.n_opts, bounds=self.bounds)

    def optimise(self):
        self.paneller.x0 = self.opt.optimise(self.paneller.x0)
        return self.paneller.x0

    def objective(self, x: np.ndarray, grad: np.ndarray) -> float:
        length = self.paneller.length(x)
        if grad.size > 0:
            grad[:] = approx_derivative(
                self.paneller.length, x, bounds=self.bounds, f0=length
            )
        return length

    def constrain_min_length_and_angles(
        self, constraint: np.ndarray, x: np.ndarray, grad: np.ndarray
    ) -> np.ndarray:
        # Constrain minimum length
        lengths = self.paneller.panel_lengths(x)
        constraint[: len(lengths)] = self.paneller.dx_min - lengths
        if grad.size > 0:
            # TODO(hsaunders1904): work out what BLUEPRINT was doing to
            #  get this gradient
            grad[: len(lengths)] = approx_derivative(
                lambda x_opt: -self.paneller.panel_lengths(x_opt),
                x0=x,
                f0=constraint[: len(lengths)],
                bounds=self.bounds,
            )

        # Constrain joint angles
        constraint[len(lengths) :] = self.paneller.angles(x) - self.paneller.max_angle
        if grad.size > 0:
            # TODO(hsaunders): I'm sure we can be smarter about this gradient
            grad[len(lengths) :, :] = approx_derivative(
                lambda x_opt: self.paneller.angles(x_opt),
                x0=x,
                f0=constraint[len(lengths) :],
                bounds=self.bounds,
            )
        return constraint

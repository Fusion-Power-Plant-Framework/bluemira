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
"""Definition of panelling optimisation problem for EUDEMO."""

import numpy as np

from bluemira.utilities.opt_problems import (
    OptimisationConstraint,
    OptimisationObjective,
    OptimisationProblem,
    Optimiser,
)
from bluemira.utilities.optimiser import approx_derivative
from eudemo.blanket.panelling._paneller import Paneller


class PanellingOptProblem(OptimisationProblem):
    """
    Optimisation problem to minimise the cumulative length of first wall panels.

    The optimisation parameters are the normalised lengths along the
    panelling boundary at which the panels and boundary touch.

    The objective is to minimise the cumulative length of the panels,
    subject to a maximum tail-to-tail angle between panels and a minimum
    panel length. Note that the parameters for these constraints are
    properties of the ``paneller``.

    Parameters
    ----------
    paneller:
        The :class:`.Paneller` to optimise the parameters of. Note that
        these parameters are the normalised length along the paneller's
        boundary where the panels and boundary meet.
    optimiser:
        The :class:`.Optimiser` to perform the optimisation with.
    """

    def __init__(self, paneller: Paneller, optimiser: Optimiser):
        self.paneller = paneller
        self.bounds = (np.zeros_like(self.paneller.x0), np.ones_like(self.paneller.x0))
        objective = OptimisationObjective(self.objective)
        constraint = OptimisationConstraint(
            self.constrain_min_length_and_angles,
            f_constraint_args={},
            tolerance=np.full(self.n_constraints, 1e-8),
        )
        super().__init__(self.paneller.x0, optimiser, objective, [constraint])
        self.set_up_optimiser(self.n_opts, bounds=self.bounds)

    @property
    def n_opts(self) -> int:
        """
        The number of optimisation parameters.

        The optimisation parameters are how far along the boundary's
        length each panel tangents the boundary (normalized to between
        0 and 1). We exclude the panel's start and end points, which are
        fixed.
        """
        # exclude start and end points; hence 'N - 2'
        return self.paneller.n_panels - 2

    @property
    def n_constraints(self) -> int:
        """
        The number of optimisation constraints.

        We constrain:

            - the minimum length of each panel
            - the angle between each panel
              (no. of angles = no. of panels - 1)

        Note that we exclude the start and end touch points which are
        fixed.
        """
        return 2 * self.paneller.n_panels - 1

    def optimise(self, check_constraints: bool = False) -> np.ndarray:
        """Perform the optimisation."""
        self.paneller.x0 = self.opt.optimise(self.paneller.x0, check_constraints)
        return self.paneller.x0

    def objective(self, x: np.ndarray, grad: np.ndarray) -> float:
        """Objective function to pass to ``nlopt``."""
        length = self.paneller.length(x)
        if grad.size > 0:
            grad[:] = approx_derivative(
                self.paneller.length, x, bounds=self.bounds, f0=length
            )
        return length

    def constrain_min_length_and_angles(
        self, constraint: np.ndarray, x: np.ndarray, grad: np.ndarray
    ) -> np.ndarray:
        """Constraint function to pass to ``nlopt``."""
        n_panels = self.paneller.n_panels
        self._constrain_min_length(constraint[:n_panels], x, grad[:n_panels])
        self._constrain_max_angle(constraint[n_panels:], x, grad[n_panels:])
        return constraint

    def _constrain_min_length(
        self, constraint: np.ndarray, x: np.ndarray, grad: np.ndarray
    ) -> None:
        lengths = self.paneller.panel_lengths(x)
        constraint[:] = self.paneller.dx_min - lengths
        if grad.size > 0:
            grad[:] = approx_derivative(
                lambda x_opt: -self.paneller.panel_lengths(x_opt),
                x0=x,
                f0=constraint,
                bounds=self.bounds,
            )

    def _constrain_max_angle(
        self, constraint: np.ndarray, x: np.ndarray, grad: np.ndarray
    ) -> None:
        constraint[:] = self.paneller.angles(x) - self.paneller.max_angle
        if grad.size > 0:
            grad[:] = approx_derivative(
                self.paneller.angles,
                x0=x,
                f0=constraint,
                bounds=self.bounds,
            )

    def constraints_violated_by(self, x: np.ndarray, atol: float) -> bool:
        """Return True if any constraints are violated by more that ``atol``."""
        constraint = np.empty(self.n_constraints)
        self.constrain_min_length_and_angles(constraint, x, np.empty(0))
        return np.any(constraint > atol)

# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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

"""
Geometry optimisation classes and tools
"""

import abc

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import signed_distance_2D_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.optimiser import Optimiser


class GeometryOptimisationProblem(abc.ABC):
    """
    Geometry optimisation problem class.

    Parameters
    ----------
    parameterisation: GeometryParameterisation
        Geometry parameterisation instance to use in the optimisation problem
    optimiser: Optimiser
        Optimiser instance to use in the optimisation problem
    """

    def __init__(self, parameterisation: GeometryParameterisation, optimiser: Optimiser):
        self.parameterisation = parameterisation
        self.optimiser = optimiser
        self.optimiser.build_optimiser(parameterisation.variables.n_free_variables)
        self.optimiser.set_lower_bounds(np.zeros(optimiser.n_variables))
        self.optimiser.set_upper_bounds(np.ones(optimiser.n_variables))
        self.optimiser.set_objective_function(self.f_objective)

    def apply_shape_constraints(self):
        """
        Add shape constraints to the geometry parameterisation, if they exist.
        """
        n_shape_ineq_cons = self.parameterisation.n_ineq_constraints
        if n_shape_ineq_cons > 0:
            self.optimiser.add_ineq_constraints(
                self.parameterisation.shape_ineq_constraints, np.zeros(n_shape_ineq_cons)
            )
        else:
            bluemira_warn(
                f"GeometryParameterisation {self.parameterisation.__class.__name__} does"
                "not have any shape constraints."
            )

    def update_parameterisation(self, x):
        """
        Update the GeometryParameterisation.
        """
        self.parameterisation.variables.set_values_from_norm(x)

    f_objective = None

    def solve(self, x0=None):
        """
        Solve the GeometryOptimisationProblem.
        """
        if x0 is None:
            x0 = self.parameterisation.variables.get_normalised_values()
        x_star = self.optimiser.optimise(x0)
        self.update_parameterisation(x_star)


class MinimiseLength(GeometryOptimisationProblem):
    """
    Optimiser to minimise the length of a geometry 2D parameterisation
    in the xz-plane, with optional constraints in the form of a
    "keep-out zone".
    """

    def __init__(
        self,
        parameterisation: GeometryParameterisation,
        optimiser: Optimiser,
        keep_out_zone: BluemiraWire = None,
        n_koz_points: int = 100,
        koz_con_tol: float = 1e-3,
    ):
        super().__init__(parameterisation, optimiser)

        self.n_koz_points = n_koz_points
        self.keep_out_zone = keep_out_zone
        if self.keep_out_zone is not None:
            self.koz_points = self._make_koz_points(
                self.keep_out_zone, self.n_koz_points
            )
            self.optimiser.add_ineq_constraints(
                self.f_constrain_koz,
                koz_con_tol * np.ones(self.n_koz_points),
            )

    def f_objective(self, x, grad):
        """The objective function for the optimiser."""
        length = self.calculate_length(x)
        if grad.size > 0:
            grad[:] = self.optimiser.approx_derivative(
                self.calculate_length, x, f0=length
            )
        return length

    def calculate_length(self, x):
        """Calculate the length of the shape being optimised."""
        self.update_parameterisation(x)
        return self.parameterisation.create_shape().length

    def f_constrain_koz(self, constraint, x, grad):
        """
        Geometry constraint function to the keep-out-zone.
        """
        constraint[:] = self.calculate_signed_distance(x)
        if grad.size > 0:
            grad[:] = self.optimiser.approx_derivative(
                self.calculate_signed_distance, x, constraint
            )
        return constraint

    def calculate_signed_distance(self, x):
        """
        Calculate the signed distances from the parameterised shape to
        the keep-out zone.
        """
        self.update_parameterisation(x)

        shape = self.parameterisation.create_shape()
        s = shape.discretize(ndiscr=self.n_koz_points).xz
        return signed_distance_2D_polygon(s.T, self.koz_points.T).T

    def _make_koz_points(self, keep_out_zone: BluemiraWire, n_points: int) -> np.ndarray:
        """
        Generate a set of points that combine the given keep-out-zones,
        to use as constraints in the optimisation problem.

        Returns
        -------
        coords: np.ndarray[2, n_points]
            Coordinates of the keep-out-zone points in the xz plane.
        """
        return keep_out_zone.discretize(byedges=True, ndiscr=n_points).xz

# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Some crude EU-DEMO remote maintenance considerations
"""

import numpy as np

from bluemira.base.constants import EPS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.geometry.tools import make_polygon, slice_shape
from bluemira.utilities.opt_problems import (
    OptimisationConstraint,
    OptimisationObjective,
    OptimisationProblem,
)
from bluemira.utilities.optimiser import Optimiser, approx_derivative


class UpperPortOP(OptimisationProblem):
    """
    Reduced model optimisation problem for the vertical upper port size optimisation.

    Parameters
    ----------
    """

    def __init__(
        self, parameterisation, optimiser: Optimiser, breeding_blanket_xz: BluemiraFace
    ):

        objective = OptimisationObjective(self.minimise_port_size, f_objective_args={})

        box = breeding_blanket_xz.bounding_box
        r_ib_min = box.x_min
        r_ob_max = box.x_max

        constraints = [
            OptimisationConstraint(
                self.constrain_blanket_cut,
                f_constraint_args={
                    "bb": breeding_blanket_xz,
                    "c_rm": 0.02,
                    "r_ib_min": r_ib_min,
                    "r_ob_max": r_ob_max,
                },
                tolerance=1e-6 * np.ones(3),
            )
        ]
        super().__init__(parameterisation, optimiser, objective, constraints)
        lower_bounds = [r_ib_min + 0.5, 9, r_ib_min + 1, 0]
        upper_bounds = [9, r_ob_max - 0.5, r_ob_max - 1, 30]
        self.set_up_optimiser(4, bounds=[lower_bounds, upper_bounds])

    @staticmethod
    def minimise_port_size(vector, grad):
        """
        Minimise the size of the port whilst maximising its outboard radius
        """
        ri, ro, ci, gamma = vector
        # Dual objective: minimise port (ro - ri) and maximise outer radius (-ro)
        # ro - ri - ro
        value = -ri
        if grad.size > 0:
            grad[0] = -1
            grad[1] = 0
            grad[2] = 0
            grad[3] = 0
        return value

    @staticmethod
    def constrain_blanket_cut(constraint, vector, grad, bb, c_rm, r_ib_min, r_ob_max):
        """
        Calculate the constraint and grad matrices.
        """
        constraint[:] = UpperPortOP.calculate_constraints(
            vector, bb, c_rm, r_ib_min, r_ob_max
        )

        if grad.size > 0:
            grad[:] = approx_derivative(
                UpperPortOP.calculate_constraints,
                vector,
                f0=constraint,
                args=(bb, c_rm, r_ib_min, r_ob_max),
            )

        return constraint

    @staticmethod
    def calculate_constraints(vector, bb, c_rm, r_ib_min, r_ob_max):
        """
        Calculate the constraints on the upper port size.
        """
        ri, ro, ci, gamma = vector

        co = UpperPortOP.get_outer_cut_point(bb, ci, gamma)

        return [
            (r_ob_max - ci + 2 * c_rm) - (ro - co),
            (ci - r_ib_min + 2 * c_rm) - (ro - ri),
            (ro - ci + 2 * c_rm) - (ci - r_ib_min),
        ]

    @staticmethod
    def get_outer_cut_point(bb, ci, gamma):
        """
        Get the outer cut point radius of the cutting plane with the breeding blanket
        geometry.
        """
        cut_plane = BluemiraPlacement.from_3_points([ci, 0, 0], [ci, 0, 1], [ci, 1, 1])
        # Get the first intersection with the vertical inner cut plane
        intersections = slice_shape(bb.boundary[0], cut_plane)
        intersections = intersections[intersections[:, -1] > 0.0]
        intersection = sorted(intersections, key=lambda x: x[-1])[0]

        x, y, z = intersection
        x2 = x - np.sin(np.deg2rad(gamma))
        y2 = y
        z2 = z + np.cos(np.deg2rad(gamma))
        angled_cut_plane = BluemiraPlacement.from_3_points(
            intersection, [x2, y2, z2], [x, y + 1, z]
        )
        # Get the last intersection with the angled cut plane and the outer
        intersections = slice_shape(bb.boundary[0], angled_cut_plane)
        intersections = intersections[intersections[:, -1] > z + EPS]
        intersection = sorted(intersections, key=lambda x: x[-1])[0]
        co, _, _ = intersection
        return co

    def optimise(self, x0):
        """
        Solve the optimisation problem.
        """
        return self.opt.optimise(x0)


if __name__ == "__main__":

    bb = make_polygon(
        {"x": [5, 6, 6, 11, 11, 12, 12, 5], "y": 0, "z": [-5, -5, 5, 5, -5, -5, 6, 6]},
        closed=True,
    )
    bb = BluemiraFace(bb)
    optimiser = Optimiser("SLSQP", opt_conditions={"max_eval": 1000, "ftol_rel": 1e-8})

    parameterisation = np.array([])
    design_problem = UpperPortOP(parameterisation, optimiser, bb)

    result = design_problem.optimise([7, 10, 9, 0])

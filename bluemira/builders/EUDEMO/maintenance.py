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
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import boolean_cut, make_polygon, slice_shape
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
    optimiser: Optimiser
        Optimiser object to use when solving this problem
    breeding_blanket_xz: BluemiraFace
        Unsegmented breeding blanket x-z geometry
    """

    def __init__(self, params, optimiser: Optimiser, breeding_blanket_xz: BluemiraFace):

        objective = OptimisationObjective(self.minimise_port_size, f_objective_args={})

        box = breeding_blanket_xz.bounding_box
        r_ib_min = box.x_min
        r_ob_max = box.x_max

        constraints = [
            OptimisationConstraint(
                self.constrain_blanket_cut,
                f_constraint_args={
                    "bb": breeding_blanket_xz,
                    "c_rm": params.c_rm.value,
                    "r_ib_min": r_ib_min,
                    "r_ob_max": r_ob_max,
                },
                tolerance=1e-6 * np.ones(3),
            )
        ]
        super().__init__(np.array([]), optimiser, objective, constraints)
        lower_bounds = [r_ib_min - params.c_rm.value, 9, r_ib_min + 1, 0]
        upper_bounds = [9, r_ob_max + params.c_rm.value, r_ob_max - 1, 30]
        self.set_up_optimiser(4, bounds=[lower_bounds, upper_bounds])
        self.bb_xz = breeding_blanket_xz
        self.params = params

    @staticmethod
    def minimise_port_size(vector, grad):
        """
        Minimise the size of the port whilst maximising its outboard radius
        """
        ri, ro, ci, gamma = vector
        # Dual objective: minimise port (ro - ri) and maximise outer radius (-ro)
        # ro - ri - ro
        value = ro - ri
        if grad.size > 0:
            grad[0] = -1
            grad[1] = 1
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

        co = UpperPortOP.get_outer_cut_point(bb, ci, gamma)[0]

        return [
            # The outboard blanket must fit through the port
            (r_ob_max - co + c_rm) - (ro - co),
            # The inboard blanket must fit through the port (dominated by below)
            # (ci - r_ib_min + c_rm) - (ro - ri),
            # The inboard blanket must squeeze past the other inboard blanket
            (ci - r_ib_min) - (ro - ci + c_rm),
            # There should be enough vertically accessible space on the inboard blanket
            (ri + 0.5 * abs(ci - r_ib_min)) - (ci),
        ]

    @staticmethod
    def get_inner_cut_point(bb, ci):
        """
        Get the inner cut point of the breeding blanket geometry.
        """
        cut_plane = BluemiraPlane.from_3_points([ci, 0, 0], [ci, 0, 1], [ci, 1, 1])
        # Get the first intersection with the vertical inner cut plane
        intersections = slice_shape(bb.boundary[0], cut_plane)
        intersections = intersections[intersections[:, -1] > 0.0]
        intersection = sorted(intersections, key=lambda x: x[-1])[0]
        return intersection

    @staticmethod
    def get_outer_cut_point(bb, ci, gamma):
        """
        Get the outer cut point radius of the cutting plane with the breeding blanket
        geometry.
        """
        intersection = UpperPortOP.get_inner_cut_point(bb, ci)
        x, y, z = intersection
        x2 = x - np.sin(np.deg2rad(gamma))
        y2 = y
        z2 = z + np.cos(np.deg2rad(gamma))
        angled_cut_plane = BluemiraPlane.from_3_points(
            intersection, [x2, y2, z2], [x, y + 1, z]
        )
        # Get the last intersection with the angled cut plane and the outer
        intersections = slice_shape(bb.boundary[0], angled_cut_plane)
        intersections = intersections[intersections[:, -1] > z + EPS]
        intersection = sorted(intersections, key=lambda x: x[-1])[0]
        return intersection

    def optimise(self, x0):
        """
        Solve the optimisation problem.
        """
        x0 = self.opt.optimise(x0)
        bb_cut = self._make_bb_cut_geometry(x0)
        up_zone = self._make_upper_port_zone(x0)
        cut_result = boolean_cut(self.bb_xz, bb_cut)
        if len(cut_result) != 2:
            bluemira_warn(
                "Something strange is going on with the BB poloidal segmentation"
            )
        ib_silhouette, ob_silhouette = sorted(
            cut_result, key=lambda x: x.center_of_mass[0]
        )[:2]
        return ib_silhouette, ob_silhouette, up_zone

    def _make_bb_cut_geometry(self, vector):
        """
        Make the void geometry for the BB cut in the poloidal plane.
        """
        ri, ro, ci, gamma = vector
        c_rm = self.params.c_rm.value
        p0 = self.get_inner_cut_point(self.bb_xz, ci)
        p1 = [p0[0], 0, p0[2] + 2]
        p2 = [p0[0] - c_rm, 0, p1[2]]
        p3 = [p2[0], 0, p0[2] - np.sqrt(2) * c_rm]
        cut_zone = BluemiraFace(make_polygon([p0, p1, p2, p3], closed=True))
        if gamma != 0.0:
            cut_zone.rotate(base=p0, direction=(0, 1, 0), degree=gamma)
        return cut_zone

    @staticmethod
    def _make_upper_port_zone(vector):
        """
        Make the void geometry for the upper port in the poloidal plane.
        """
        ri, ro, ci, gamma = vector
        return BluemiraFace(
            make_polygon({"x": [ri, ro, ro, ri], "z": [0, 0, 10, 10]}, closed=True)
        )


if __name__ == "__main__":

    from bluemira.base.config import Configuration
    from bluemira.display import show_cad

    params = Configuration()
    bb = make_polygon(
        {"x": [5, 6, 6, 11, 11, 12, 12, 5], "y": 0, "z": [-5, -5, 5, 5, -5, -5, 6, 6]},
        closed=True,
    )
    bb = BluemiraFace(bb)
    optimiser = Optimiser("SLSQP", opt_conditions={"max_eval": 1000, "ftol_rel": 1e-8})

    design_problem = UpperPortOP(params, optimiser, bb)

    ib, ob, up_port = design_problem.optimise([7, 10, 9, 0])
    show_cad([ib, ob, up_port])

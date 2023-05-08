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

"""
Some crude EU-DEMO remote maintenance considerations
"""

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np

from bluemira.base.constants import EPS
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import make_polygon, slice_shape
from bluemira.utilities.opt_problems import (
    OptimisationConstraint,
    OptimisationObjective,
    OptimisationProblem,
)
from bluemira.utilities.optimiser import Optimiser, approx_derivative
from eudemo.tools import get_inner_cut_point


@dataclass
class UpperPortOPParameters(ParameterFrame):
    """Parameters required to run :class:`UpperPortOP`."""

    c_rm: Parameter[float]
    """Remote maintenance clearance [m]."""
    R_0: Parameter[float]
    """Major radius [m]."""
    bb_min_angle: Parameter[float]
    """Minimum blanket module angle [degrees]."""
    tk_bb_ib: Parameter[float]
    """Blanket inboard thickness [m]."""
    tk_bb_ob: Parameter[float]
    """Blanket outboard thickness [m]."""


class UpperPortOP(OptimisationProblem):
    """
    Reduced model optimisation problem for the vertical upper port size optimisation.

    Parameters
    ----------
    params:
        Parameter frame for the problem. See
        :class:`UpperPortOPParameters` for parameter details.
    optimiser:
        Optimiser object to use when solving this problem
    breeding_blanket_xz:
        Unsegmented breeding blanket x-z geometry
    constraint_tol:
        Constraint tolerance
    """

    params: UpperPortOPParameters

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        optimiser: Optimiser,
        breeding_blanket_xz: BluemiraFace,
        constraint_tol: float = 1e-6,
    ):
        params = make_parameter_frame(params, UpperPortOPParameters)
        objective = OptimisationObjective(self.minimise_port_size, f_objective_args={})

        box = breeding_blanket_xz.bounding_box
        r_ib_min = box.x_min
        r_ob_max = box.x_max
        c_rm = params.c_rm.value
        R_0 = params.R_0.value
        bb_min_angle = 90 - params.bb_min_angle.value
        tk_bb_ib = params.tk_bb_ib.value
        tk_bb_ob = params.tk_bb_ob.value

        n_variables = 4
        n_constraints = 3

        constraints = [
            OptimisationConstraint(
                self.constrain_blanket_cut,
                f_constraint_args={
                    "bb": breeding_blanket_xz,
                    "c_rm": c_rm,
                    "r_ib_min": r_ib_min,
                    "r_ob_max": r_ob_max,
                },
                tolerance=constraint_tol * np.ones(n_constraints),
            )
        ]
        super().__init__(np.array([]), optimiser, objective, constraints)

        lower_bounds = [r_ib_min - c_rm, R_0, r_ib_min + tk_bb_ib, 0]
        upper_bounds = [R_0, r_ob_max + c_rm, r_ob_max - tk_bb_ob, bb_min_angle]
        self.set_up_optimiser(n_variables, bounds=[lower_bounds, upper_bounds])
        self.params = params

    @staticmethod
    def minimise_port_size(vector, grad):
        """
        Minimise the size of the port.
        """
        ri, ro, ci, gamma = vector
        # Dual objective: minimise port (ro - ri) and minimise cut angle
        value = ro - ri + gamma
        if grad.size > 0:
            grad[0] = -1
            grad[1] = 1
            grad[2] = 0
            grad[3] = 1
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
    def get_outer_cut_point(bb, ci, gamma):
        """
        Get the outer cut point radius of the cutting plane with the breeding blanket
        geometry.
        """
        intersection = get_inner_cut_point(bb, ci)
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

    def optimise(self, x0=None):
        """
        Solve the optimisation problem.
        """
        if x0 is None:
            R_0 = self.params.R_0.value
            x0 = np.array([R_0, R_0, R_0, 0])
        return self.opt.optimise(x0)


class UpperPortDesigner(Designer):
    """Upper Port Designer"""

    param_cls = UpperPortOPParameters

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        blanket_face: BluemiraFace,
        upper_port_extrema=10,
    ):
        super().__init__(params, build_config)
        self.blanket_face = blanket_face
        self.opt_conditions = {
            **{"max_eval": 1000, "ftol_rel": 1e-8},
            **self.build_config.get("opt_conditions", {}),
        }
        self.upper_port_extrema = upper_port_extrema

    def run(self):
        """Run Method of Designer"""
        design_problem = UpperPortOP(
            self.params,
            Optimiser("SLSQP", opt_conditions=self.opt_conditions),
            self.blanket_face,
        )

        r_up_inner, r_up_outer, r_cut, cut_angle = design_problem.optimise()

        return (
            build_upper_port_zone(r_up_inner, r_up_outer, z_max=self.upper_port_extrema),
            r_cut,
            cut_angle,
        )


def build_upper_port_zone(
    r_up_inner: float, r_up_outer: float, z_max: float = 10, z_min: float = 0
) -> BluemiraFace:
    """
    Make the void geometry for the upper port in the poloidal plane.

    Parameters
    ----------
    r_up_inner:
        Inner radius of the upper port void space
    r_up_outer:
        Outer radius of the upper port void space
    z_max:
        Maximum vertical height of the upper port void space
    z_min:
        Minimum vertical height of the upper port void space

    Returns
    -------
    Face representing the upper port void space in the x-z plane
    """
    x = [r_up_inner, r_up_outer, r_up_outer, r_up_inner]
    z = [z_min, z_min, z_max, z_max]
    return BluemiraFace(make_polygon({"x": x, "y": 0, "z": z}, closed=True))

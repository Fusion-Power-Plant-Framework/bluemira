#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
TF coil cage structural optimisation
"""

import numpy as np
from scipy.optimize import minimize
from bluemira.utilities.opt_tools import process_scipy_result
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.offset import varied_angular_offset
from BLUEPRINT.geometry.boolean import convex_hull
from BLUEPRINT.nova.structuralsolver import StructuralSolver


class StructuralOptimiser:
    """
    Structural solver for coil cage structure optimisations.

    Parameters
    ----------
    architect: Type[CoilArchitect]
        The CoilArchitect object used to build the solver model
    coilcage: Type[CoilCage]
        The CoilCage used to calculate the TF forces
    equilibria: List[Union[Type[Equilibrium], Type[Breakdown]]
        The list of equilibria objects used to calculate the PF forces
    """

    def __init__(self, architect, coilcage, equilibria):
        # Handle inputs
        self.architect = architect
        self.coilcage = coilcage
        self.equilibria = equilibria

        # Save initial geometries
        self.inner_ref = self.architect.case_loops["inner"].copy()
        self.outer_ref = self.architect.case_loops["outer"].copy()

        # Constructors
        self.opt = None
        self.result = None
        self.bounds = None
        self.args = ()

        # Initial set-up
        self.solver = StructuralSolver(architect, coilcage, equilibria)

        # Determine worst case scenario, and optimise over this
        worst_case = self._get_worst_case()
        self.solver.equilibria = [equilibria[worst_case]]

    def _get_worst_case(self):
        """
        Find the worst case equilibrium and optimise structure with this.
        """
        results = self.solver.solve()
        return np.argmax([np.max(r._max_deflections) for r in results])

    def solve_structure(self):
        """
        Solve the FE model of the structure for all equilibria.

        Returns
        -------
        deflections: np.array
            The vector of deflections in the model for all equilibria
        """
        results = self.solver.solve()
        deflections = np.array([r._max_deflections for r in results]).flatten()
        # We can ignore the Breakdown for y deflections, as they are generally small
        y_deflections = np.array([r.deflections_xyz.T[1] for r in results[1:]]).flatten()
        # Clear model loads
        self.solver.model.clear_loads()
        return deflections, y_deflections

    def f_objective(self, x_norm, *args):
        """
        The optimiser objective function: minimise the volume of the structure.
        This is done through an ersatz objective for the area of the structure.

        Returns
        -------
        value: float
            The value of the outer structure Loop - inner structure Loop
        """
        # Modify the shapes based on the variable vector x
        inner, outer = self.update_architect(x_norm)
        area = outer.area - inner.area
        return area / 50

    def f_struct_constraints(self, x_norm, *args):
        """
        The structural constraints function.

        Returns
        -------
        constraints: np.array
            The array of constraint equation values
        """
        # Modify the shapes based on the variable vector x
        # Update the Architect geometry
        self.update_architect(x_norm)

        # Update the StructuralSolver model
        self.solver.define_geometry()

        # Re-solve the FE problem
        deflections, y_deflections = self.solve_structure()

        # Re-calculate the structural constraint function
        # TODO: Stitch these into Parameters later
        max_deflection = 0.1  # [m]
        # TODO: Work this out once the gravity supports are better
        # max_y_deflection = 0.08  # [m]

        constraints = max_deflection - deflections
        # constraints = max_y_deflection - np.abs(y_deflections)
        return constraints

    def optimise(self):
        """
        Run the coil cage structural optimisation.
        """
        # Get the relevant bounds from the CoilArchitect
        tk_in_max, tk_out_max = self.architect.get_max_casing_tks()
        x_gs_min, x_gs_max = self.architect.get_GS_inclusion_zone()

        # Correct for the existing thicknesses
        tk_in_max -= self.architect.params.tk_tf_case_out_in
        tk_out_max -= self.architect.params.tk_tf_case_out_out

        constraints = [
            {"type": "ineq", "fun": self.f_struct_constraints, "args": self.args},
        ]
        bounds = [[0, tk_in_max], [0, tk_out_max], [x_gs_min, x_gs_max]]
        self.bounds = bounds
        x0_norm = np.zeros(3)

        self.result = minimize(
            self.f_objective,
            x0_norm,
            bounds=[[0, 1], [0, 1], [0, 1]],
            method="SLSQP",
            constraints=constraints,
            tol=1e-3,
            options={"eps": 1e-4, "ftol": 1e-3},
        )

        x_norm = process_scipy_result(self.result)
        self.update_architect(x_norm)
        # Update params
        tk_in, tk_out, x_gs = self._denormalise_vars(x_norm)
        self.architect.params.tk_tf_case_out_in += tk_in
        self.architect.params.tk_tf_case_out_out += tk_out
        self.architect._generate_xz_loops()

    def update_architect(self, x):
        """
        Update the Architect object with the optimisation variables.

        Parameters
        ----------
        x: np.array
            The normalised variable vector

        Returns
        -------
        inner: Loop
            The inner casing loop
        outer: Loop
            The outer casing loop
        """
        x = self._denormalise_vars(x)
        inner = self.build_inner_loop(x[0])
        outer = self.build_outer_loop(x[1])
        # Update 2-D geometry in the CoilArchitect and ToroidalFieldCoils
        self.architect.case_loops["inner"] = inner
        self.architect.tf.loops["in"] = inner.as_dict()
        self.architect.case_loops["outer"] = outer
        self.architect.tf.loops["out"] = outer.as_dict()

        # Update geometry for CAD in ToroidalFieldCoils
        self.architect.tf.geom["TF case in"].inner = inner
        self.architect.tf.geom["TF case out"].outer = outer
        # Update gravity support
        self.architect.params.x_g_support = x[2]
        self.architect._build_gravity_supports()
        return inner, outer

    def build_outer_loop(self, tk_outer):
        """
        Build the outer casing Loop.

        Parameters
        ----------
        tk_outer: float
            The outer thickness offset variable

        Returns
        -------
        outer_loop: Loop
            The outer casing Loop
        """
        outer = self.outer_ref
        offset = outer.offset(tk_outer)
        # Make TF case closing corners
        x_min = np.min(outer.x)
        z_min = np.min(offset.z)
        z_max = np.max(offset.z)
        arg_min = np.argmin(offset.z)
        arg_max = np.argmax(offset.z)
        x = np.concatenate([[x_min], offset.x[arg_min:arg_max], [x_min]])
        z = np.concatenate([[z_min], offset.z[arg_min:arg_max], [z_max]])
        outer_loop = Loop(x=x, z=z)
        outer_loop.close()
        return outer_loop

    def build_inner_loop(self, tk_inner):
        """
        Build the inner casing Loop.

        Parameters
        ----------
        tk_inner: float
            The outer thickness offset variable

        Returns
        -------
        inner_loop: Loop
            The inner casing Loop
        """
        inner = self.inner_ref
        offset = varied_angular_offset(
            inner, 0, -tk_inner, outer_angle=np.pi / 4, blend_angle=np.pi / 4
        )
        inner_loop = convex_hull([offset, self.architect.tf.inputs["koz_loop"]])
        return inner_loop

    def _normalise_vars(self, x_vars):
        """
        Normalise the optimisation variables.
        """
        x_norm = np.zeros(len(x_vars))
        for i, (x, bounds) in enumerate(zip(x_vars, self.bounds)):
            x_norm[i] = (x - bounds[0]) / (bounds[1] - bounds[0])

        return x_norm

    def _denormalise_vars(self, x_norm):
        """
        Denormalise the optimisation variables.
        """
        x_vars = np.zeros(len(x_norm))
        for i, (x, bounds) in enumerate(zip(x_norm, self.bounds)):
            x_vars[i] = bounds[0] + (bounds[1] - bounds[0]) * x

        return x_vars


if __name__ == "__main__":
    pass

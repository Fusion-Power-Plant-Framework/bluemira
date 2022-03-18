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
Built-in build steps for making parameterised TF coils.
"""

from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_debug_flush
from bluemira.display import plot_2d
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.optimisation import (
    GeometryOptimisationProblem,
    constrain_koz,
    minimise_length,
)
from bluemira.geometry.tools import offset_wire
from bluemira.geometry.wire import BluemiraWire
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.magnetostatics.circuits import HelmholtzCage
from bluemira.utilities.opt_problems import OptimisationConstraint, OptimisationObjective
from bluemira.utilities.optimiser import approx_derivative


class RippleConstrainedLengthOpt(GeometryOptimisationProblem):
    """
    Toroidal field coil winding pack shape optimisation problem.

    Parameters
    ----------
    parameterisation: GeometryParameterisation
        Geometry parameterisation for the winding pack current centreline
    optimiser: Optimiser
        Optimiser to use to solve the optimisation problem
    params: ParameterFrame
        Parameters required to solve the optimisation problem
    separatrix: BluemiraWire
        Separatrix shape at which the TF ripple is to be constrained
    keep_out_zone: Optional[BluemiraWire]
        Zone boundary which the WP may not enter
    rip_con_tol: float
        Tolerance with which to apply the ripple constraints
    kox_con_tol: float
        Tolerance with which to apply the keep-out-zone constraints
    nx: int
        Number of radial Biot-Savart filaments to use
    ny: int
        Number of toroidal Biot-Savart filaments to use
    n_koz_points: int
        Number of discretised points to use when enforcing the keep-out-zone constraint

    Notes
    -----
    x^* = minimise: winding_pack_length
          subject to:
              ripple|separatrix \\preceq TF_ripple_limit
              SDF(wp_shape, keep_out_zone) \\preceq 0

    The geometry parameterisation is updated in place
    """

    def __init__(
        self,
        parameterisation,
        optimiser,
        params,
        wp_cross_section,
        separatrix,
        keep_out_zone=None,
        rip_con_tol=1e-3,
        koz_con_tol=1e-3,
        nx=1,
        ny=1,
        n_rip_points=100,
        n_koz_points=100,
    ):
        self.params = params
        self.separatrix = separatrix
        self.wp_cross_section = wp_cross_section
        self.keep_out_zone = keep_out_zone

        self.n_rip_points = n_rip_points
        self.ripple_points = self._make_ripple_points(separatrix)
        self.ripple_values = None

        objective = OptimisationObjective(
            minimise_length, f_objective_args={"parameterisation": parameterisation}
        )

        ripple_points = self._make_ripple_points(separatrix)
        ripple_constraint = OptimisationConstraint(
            self.constrain_ripple,
            f_constraint_args={
                "parameterisation": parameterisation,
                "ripple_points": ripple_points,
                "wp_xs": wp_cross_section,
                "nx": nx,
                "ny": ny,
                "params": params,
            },
            tolerance=rip_con_tol * np.ones(n_rip_points),
        )
        constraints = [ripple_constraint]
        if keep_out_zone is not None:
            koz_points = self._make_koz_points(keep_out_zone)
            koz_constraint = OptimisationConstraint(
                constrain_koz,
                f_constraint_args={
                    "parameterisation": parameterisation,
                    "n_shape_discr": n_koz_points,
                    "koz_points": koz_points,
                },
                tolerance=koz_con_tol * np.ones(n_koz_points),
            )
            constraints.append(koz_constraint)

        super().__init__(parameterisation, optimiser, objective, constraints)

    def _make_koz_points(self, keep_out_zone):
        """
        Make a set of points at which to evaluate the KOZ constraint
        """
        return keep_out_zone.discretize(byedges=True, dl=keep_out_zone.length / 200).xz

    def _make_ripple_points(self, separatrix):
        """
        Make a set of points at which to check the ripple

        Parameters
        ----------
        separatrix: BluemiraWire
            The geometry on which to check the ripple
        """
        # TODO: Handle case where the face is made up of multiple wires
        if not isinstance(separatrix, BluemiraWire):
            raise BuilderError(
                "Ripple points on faces made from multiple wires not yet supported."
            )
        points = separatrix.discretize(ndiscr=self.n_rip_points)
        points.set_ccw((0, 1, 0))
        # Real argument to making the points the inputs... but then the plot would look
        # sad! :D
        # Can speed this up a lot if you know about your problem... I.e. with a princeton
        # D I could only check one point and get it right faster.

        # idx = np.where(points[0] > self.params.R_0.value)[0]
        # points = points[:, idx]
        return points

    @staticmethod
    def constrain_ripple(
        constraint,
        vector,
        grad,
        parameterisation,
        ripple_points,
        wp_xs,
        nx,
        ny,
        params,
        ad_args=None,
    ):
        func = RippleConstrainedLengthOpt.calculate_ripple
        constraint[:] = func(
            vector, parameterisation, ripple_points, wp_xs, nx, ny, params
        )
        if grad.size > 0:
            grad[:] = approx_derivative(
                func,
                vector,
                f0=constraint,
                args=(parameterisation, ripple_points, wp_xs, nx, ny, params),
                **ad_args,
            )

        bluemira_debug_flush(
            f"Max ripple: {max(constraint+params.TF_ripple_limit.value)}"
        )
        return constraint

    @staticmethod
    def calculate_ripple(vector, parameterisation, ripple_points, wp_xs, nx, ny, params):
        cage = RippleConstrainedLengthOpt.update_cage(
            vector, parameterisation, wp_xs, nx, ny, params
        )
        ripple = cage.ripple(*ripple_points)
        ripple_values = ripple
        return ripple - params.TF_ripple_limit.value

    @staticmethod
    def update_cage(vector, parameterisation, wp_xs, nx, ny, params):
        parameterisation.variables.set_values_from_norm(vector)
        wire = parameterisation.create_shape()
        circuit = RippleConstrainedLengthOpt._make_single_circuit(wire, wp_xs, nx, ny)

        cage = HelmholtzCage(circuit, params.n_TF.value)
        field = cage.field(params.R_0, 0, params.z_0)
        current = -params.B_0 / field[1]  # single coil amp-turns
        current /= nx * ny  # single filament amp-turns
        cage.set_current(current)
        return cage

    @staticmethod
    def _make_single_circuit(wire, wp_cross_section, nx, ny):
        """
        Make a single BioSavart Filament for a single TF coil
        """
        bb = wp_cross_section.bounding_box
        dx_xs = 0.5 * (bb.x_max - bb.x_min)
        dy_xs = 0.5 * (bb.y_max - bb.y_min)

        dx_wp, dy_wp = [0], [0]  # default to coil centreline
        if nx > 1:
            dx_wp = np.linspace(dx_xs * (1 / nx - 1), dx_xs * (1 - 1 / nx), nx)

        if ny > 1:
            dy_wp = np.linspace(dy_xs * (1 / ny - 1), dy_xs * (1 - 1 / ny), ny)

        current_wires = []
        for dx in dx_wp:
            c_wire = offset_wire(wire, dx)
            for dy in dy_wp:
                c_w = deepcopy(c_wire)
                c_w.translate((0, dy, 0))
                current_wires.append(c_w)

        current_arrays = [
            w.discretize(byedges=True, dl=wire.length / 200) for w in current_wires
        ]

        for c in current_arrays:
            c.set_ccw((0, 1, 0))

        radius = 0.5 * BluemiraFace(wp_cross_section).area / (nx * ny)
        filament = BiotSavartFilament(
            current_arrays, radius=radius, current=1 / (nx * ny)
        )
        return filament

    # def update_cage(self, x):
    #     """
    #     Update the magnetostatic solver
    #     """
    #     super().update_parameterisation(x)
    #     wire = self.parameterisation.create_shape()
    #     circuit = self._make_single_circuit(wire)

    #     self.cage = HelmholtzCage(circuit, self.params.n_TF.value)
    #     field = self.cage.field(self.params.R_0, 0, self.params.z_0)
    #     current = -self.params.B_0 / field[1]  # single coil amp-turns
    #     current /= self.nx * self.ny  # single filament amp-turns
    #     self.cage.set_current(current)

    # def calculate_ripple(self, x):
    #     """
    #     Calculate the ripple on the target points for a given variable vector
    #     """
    #     self.update_cage(x)
    #     ripple = self.cage.ripple(*self.ripple_points)
    #     self.ripple_values = ripple
    #     return ripple - self.params.TF_ripple_limit

    # def f_constrain_ripple(self, constraint, x, grad):
    #     """
    #     Toroidal field ripple constraint function
    #     """
    #     constraint[:] = self.calculate_ripple(x)

    #     if grad.size > 0:
    #         # Only called if a gradient-based optimiser is used
    #         grad[:] = self.optimiser.approx_derivative(
    #             self.calculate_ripple, x, constraint
    #         )
    #     bluemira_debug_flush(
    #         f"Max ripple: {max(constraint+self.params.TF_ripple_limit)}"
    #     )
    #     return constraint

    def plot(self, ax=None):
        """
        Plot the optimisation problem.

        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
        if ax is None:
            ax = plt.gca()

        plot_2d(
            self.separatrix,
            ax=ax,
            show=False,
            wire_options={"color": "red", "linewidth": "0.5"},
        )
        plot_2d(
            self.parameterisation.create_shape(),
            ax=ax,
            show=False,
            wire_options={"color": "blue", "linewidth": 1.0},
        )

        if self.keep_out_zone:
            plot_2d(
                self.keep_out_zone,
                ax=ax,
                show=False,
                wire_options={"color": "k", "linewidth": 0.5},
            )

        rv = self.ripple_values
        norm = matplotlib.colors.Normalize()
        norm.autoscale(rv)
        cm = matplotlib.cm.viridis
        sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        ax.scatter(
            self.ripple_points.x,
            self.ripple_points.z,
            color=cm(norm(rv)),
        )
        color_bar = plt.gcf().colorbar(sm)
        color_bar.ax.set_ylabel("Toroidal field ripple [%]")

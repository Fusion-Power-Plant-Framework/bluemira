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
Built-in build steps for making parameterised TF coils.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_debug_flush
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.display import plot_2d
from bluemira.geometry.coordinates import Coordinates
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


class ParameterisedRippleSolver:
    """
    A parameterised Biot-Savart HelmholtzCage solver.

    Parameters
    ----------
    wp_xs: BluemiraWire
        Geometry of the TF coil winding pack cross-section
    nx: int
        Number of radial Biot-Savart filaments to use
    ny: int
        Number of toroidal Biot-Savart filaments to use
    n_TF: int
        Number of TF coils
    R_0: float
        Major radius at which to calculate B_0
    z_0: float
        Vertical coordinate at which to calculate B_0
    B_0: float
        Toroidal field at (R_0, z_0)
    ripple_points: np.ndarray
        Points at which to calculate TF ripple
    """

    def __init__(self, wp_xs, nx, ny, n_TF, R_0, z_0, B_0):
        self.wp_xs = wp_xs
        self.nx = nx
        self.ny = ny
        self.n_TF = n_TF
        self.R_0 = R_0
        self.z_0 = z_0
        self.B_0 = B_0
        self.cage = None

    def update_cage(self, wire):
        """
        Update the HelmHoltzCage, setting the current to produce a field of B_0 at
        (R_0, z_0).

        Parameters
        ----------
        wire: BluemiraWire
            TF coil winding pack current centreline
        """
        circuit = self._make_single_circuit(wire)
        self.cage = HelmholtzCage(circuit, self.n_TF)
        field = self.cage.field(self.R_0, 0, self.z_0)
        current = -self.B_0 / field[1]  # single coil amp-turns
        current /= self.nx * self.ny  # single filament amp-turns
        self.cage.set_current(current)

    def _make_single_circuit(self, wire):
        """
        Make a single BioSavart Filament for a single TF coil
        """
        bb = self.wp_xs.bounding_box
        dx_xs = 0.5 * (bb.x_max - bb.x_min)
        dy_xs = 0.5 * (bb.y_max - bb.y_min)

        dx_wp, dy_wp = [0], [0]  # default to coil centreline
        if self.nx > 1:
            dx_wp = np.linspace(
                dx_xs * (1 / self.nx - 1), dx_xs * (1 - 1 / self.nx), self.nx
            )

        if self.ny > 1:
            dy_wp = np.linspace(
                dy_xs * (1 / self.ny - 1), dy_xs * (1 - 1 / self.ny), self.ny
            )

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

        radius = 0.5 * BluemiraFace(self.wp_xs).area / (self.nx * self.ny)
        filament = BiotSavartFilament(
            current_arrays, radius=radius, current=1 / (self.nx * self.ny)
        )
        return filament

    def ripple(self, x, y, z):
        """
        Get the toroidal field ripple at points.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the ripple
        y:
            The y coordinate(s) of the points at which to calculate the ripple
        z:
            The z coordinate(s) of the points at which to calculate the ripple

        Returns
        -------
        The value of the TF ripple at the point(s) [%]
        """
        return self.cage.ripple(x, y, z)


class RipplePointConstraintStrategy(ABC):
    @abstractmethod
    def make_ripple_points(self) -> Coordinates:
        pass

    @abstractmethod
    def make_calc_ripple_func(self) -> callable:
        pass


class EquispacedRipplePointConstraintStrategy(RipplePointConstraintStrategy):
    def __init__(self, wire: BluemiraWire, n_rip_points: int):
        points = wire.discretize(byedges=True, ndiscr=n_rip_points)
        self.points = points

    def make_ripple_points(self) -> Coordinates:
        return self.points

    def make_calc_ripple_func(self) -> callable:
        return ParameterisedRippleSolver.ripple


class RippleConstrainedLengthGOP(GeometryOptimisationProblem):
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
    wp_cross_section: BluemiraWire
        Geometry of the TF coil winding pack cross-section
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

    @dataclass
    class _Params(ParameterFrame):
        n_TF: Parameter[int]
        R_0: Parameter[float]
        z_0: Parameter[float]
        B_0: Parameter[float]
        TF_ripple_limit: Parameter[float]

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
        self.params = make_parameter_frame(params, self._Params)
        self.separatrix = separatrix
        self.wp_cross_section = wp_cross_section
        self.keep_out_zone = keep_out_zone

        objective = OptimisationObjective(
            minimise_length, f_objective_args={"parameterisation": parameterisation}
        )

        self.ripple_points = self._make_ripple_points(separatrix, n_rip_points)
        self.ripple_values = None
        self.solver = ParameterisedRippleSolver(
            wp_cross_section,
            nx,
            ny,
            params.n_TF.value,
            params.R_0.value,
            params.z_0.value,
            params.B_0.value,
        )
        ripple_constraint = OptimisationConstraint(
            self.constrain_ripple,
            f_constraint_args={
                "parameterisation": parameterisation,
                "solver": self.solver,
                "points": self.ripple_points,
                "TF_ripple_limit": params.TF_ripple_limit.value,
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

    def _make_ripple_points(self, separatrix, n_rip_points):
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
        points = separatrix.discretize(ndiscr=n_rip_points)
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
        solver,
        points,
        TF_ripple_limit,
        ad_args=None,
    ):
        """
        Ripple constraint function

        Parameters
        ----------
        constraint: np.ndarray
            Constraint vector (updated in place)
        vector: np.ndarray
            Variable vector
        grad: np.ndarray
            Jacobian matrix of the constraint (updated in place)
        parameterisation: GeometryParameterisation
            Geometry parameterisation
        solver: ParameterisedHelmholtzSolver
            TF ripple solver
        points: Coordinates
            Coordinates at which to calculate the ripple
        TF_ripple_limit: float
            Maximum allowable TF ripple
        """
        func = RippleConstrainedLengthGOP.calculate_ripple
        constraint[:] = func(vector, parameterisation, solver, points, TF_ripple_limit)
        if grad.size > 0:
            grad[:] = approx_derivative(
                func,
                vector,
                f0=constraint,
                args=(parameterisation, solver, TF_ripple_limit),
                **ad_args,
            )

        bluemira_debug_flush(f"Max ripple: {max(constraint+TF_ripple_limit)}")
        return constraint

    @staticmethod
    def calculate_ripple(vector, parameterisation, solver, points, TF_ripple_limit):
        """
        Calculate ripple constraint

        Parameters
        ----------
        vector: np.ndarray
            Variable vector
        parameterisation: GeometryParameterisation
            Geometry parameterisation
        solver: ParameterisedHelmholtzSolver
            TF ripple solver
        points: Coordinates
            Coordinates at which to calculate the ripple
        TF_ripple_limit: float
            Maximum allowable TF ripple

        Returns
        -------
        c_values: np.ndarray
            Ripple constraint values
        """
        parameterisation.variables.set_values_from_norm(vector)
        wire = parameterisation.create_shape()
        solver.update_cage(wire)
        ripple = solver.ripple(*points)
        return ripple - TF_ripple_limit

    def optimise(self, x0=None):
        """
        Solve the GeometryOptimisationProblem.
        """
        parameterisation = super().optimise(x0=x0)
        self.solver.update_cage(parameterisation.create_shape())
        self.ripple_values = self.solver.ripple(*self.ripple_points)
        return parameterisation

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
            self._parameterisation.create_shape(),
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

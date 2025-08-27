# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Built-in build steps for making parameterised TF coils.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.look_and_feel import bluemira_debug_flush
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.display import plot_2d
from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.optimisation import GeomOptimisationProblem, KeepOutZone
from bluemira.geometry.tools import boolean_cut, make_polygon, offset_wire
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.magnetostatics.circuits import HelmholtzCage
from bluemira.optimisation import optimise

if TYPE_CHECKING:
    from bluemira.base.parameter_frame.typed import ParameterFrameLike
    from bluemira.geometry.optimisation import GeomConstraintT
    from bluemira.geometry.parameterisations import GeometryParameterisation
    from bluemira.geometry.wire import BluemiraWire


class ParameterisedRippleSolver:
    """
    A parameterised Biot-Savart HelmholtzCage solver.

    Parameters
    ----------
    wp_xs:
        Geometry of the TF coil winding pack cross-section
    nx:
        Number of radial Biot-Savart filaments to use
    ny:
        Number of toroidal Biot-Savart filaments to use
    n_TF:
        Number of TF coils
    R_0:
        Major radius at which to calculate B_0
    z_0:
        Vertical coordinate at which to calculate B_0
    B_0:
        Toroidal field at (R_0, z_0)
    """

    def __init__(
        self,
        wp_xs: BluemiraWire,
        nx: int,
        ny: int,
        n_TF: int,
        R_0: float,
        z_0: float,
        B_0: float,
    ):
        self.wp_xs = wp_xs
        self.nx = nx
        self.ny = ny
        self.n_TF = n_TF
        self.R_0 = R_0
        self.z_0 = z_0
        self.B_0 = B_0
        self.cage = None

    def update_cage(self, wire: BluemiraWire):
        """
        Update the HelmHoltzCage, setting the current to produce a field of B_0 at
        (R_0, z_0).

        Parameters
        ----------
        wire:
            TF coil winding pack current centreline
        """
        circuit = self._make_single_circuit(wire)
        self.cage = HelmholtzCage(circuit, self.n_TF)
        field = self.cage.field(self.R_0, 0, self.z_0)
        current = -self.B_0 / field[1]  # single coil amp-turns
        current /= self.nx * self.ny  # single filament amp-turns
        self.cage.set_current(current)

    def _make_single_circuit(self, wire: BluemiraWire) -> BiotSavartFilament:
        """
        Returns
        -------
        :
            A single BioSavart Filament for a single TF coil
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
            w.discretise(byedges=True, dl=wire.length / 200) for w in current_wires
        ]

        for c in current_arrays:
            c.set_ccw((0, 1, 0))

        radius = 0.5 * BluemiraFace(self.wp_xs).area / (self.nx * self.ny)
        return BiotSavartFilament(
            current_arrays, radius=radius, current=1 / (self.nx * self.ny)
        )

    def ripple(
        self, x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
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
        :
            The value of the TF ripple at the point(s) [%]
        """
        return self.cage.ripple(x, y, z)


class RipplePointSelector:
    """
    Baseclass for ripple point selection strategies.
    """

    def __init__(self):
        self._wire: BluemiraWire = None
        self.points: Coordinates = None

    def set_wire(self, wire: BluemiraWire):
        """
        Set the wire along which the points will be selected

        Parameters
        ----------
        wire:
            Wire along which the points will be selected
        """
        self._wire = wire

    def make_ripple_constraint(
        self, parameterisation, solver, TF_ripple_limit, rip_con_tol
    ) -> GeomConstraintT:
        """
        Make the ripple OptimisationConstraint
        """  # noqa: DOC201
        self.parameterisation = parameterisation
        self.solver = solver
        self.TF_ripple_limit = TF_ripple_limit
        return {
            "name": type(self).__name__,
            "f_constraint": self._constrain_ripple,
            "tolerance": np.full(len(self.points), rip_con_tol),
        }

    def _constrain_ripple(
        self, parameterisation: GeometryParameterisation
    ) -> np.ndarray:
        """
        Ripple constraint function

        Parameters
        ----------
        parameterisation:
            Geometry parameterisation
        """  # noqa: DOC201
        wire = parameterisation.create_shape()
        self.solver.update_cage(wire)
        ripple = self.solver.ripple(*self.points)
        # TODO @hsaunders1904: This print will call every time now,
        # Might be a case of explicitly
        # defining a df_constraint on this class, would be good for me to play with.
        # 3526
        bluemira_debug_flush(f"Max ripple: {max(ripple)}")
        return ripple - self.TF_ripple_limit


class EquispacedSelector(RipplePointSelector):
    """
    Equispaced ripple points along a wire for a given number of points.

    Parameters
    ----------
    n_rip_points:
        Number of points along the wire constrain the ripple
    x_frac:
        If specified, the fraction of radius above which the points will
        be selected.
        If unspecified, the points will be selected on the full wire
    """

    def __init__(self, n_rip_points: int, x_frac: float | None = None):
        self.n_rip_points = n_rip_points
        self.x_frac = x_frac

    def set_wire(self, wire: BluemiraWire):
        """
        Set the wire along which the points will be selected

        Parameters
        ----------
        wire:
            Wire along which the points will be selected
        """
        super().set_wire(wire)
        if self.x_frac is not None and not np.isclose(self.x_frac, 0.0):
            self.x_frac = np.clip(self.x_frac, 0.005, 0.995)
            bb = wire.bounding_box

            x_min = bb.x_min + self.x_frac * (bb.x_max - bb.x_min)

            z_min, z_max = bb.z_min - 10, bb.z_max + 10
            cut_face = BluemiraFace(
                make_polygon(
                    {
                        "x": [0, x_min, x_min, 0],
                        "y": 0,
                        "z": [z_min, z_min, z_max, z_max],
                    },
                    closed=True,
                )
            )
            wire = boolean_cut(wire, cut_face)[0]
        self.points = wire.discretise(byedges=True, ndiscr=self.n_rip_points)


class ExtremaSelector(RipplePointSelector):
    """
    Select the extrema of the wire and constrain ripple there.
    """

    def set_wire(self, wire: BluemiraWire):
        """
        Set the wire along which the points will be selected

        Parameters
        ----------
        wire:
            Wire along which the points will be selected
        """
        super().set_wire(wire)
        coords = wire.discretise(byedges=True, ndiscr=2000)
        self.points = Coordinates([
            coords.points[np.argmin(coords.x)],
            coords.points[np.argmax(coords.x)],
            coords.points[np.argmin(coords.z)],
            coords.points[np.argmax(coords.z)],
        ])


class FixedSelector(RipplePointSelector):
    """
    Specified points at which to constrain the ripple, overrides any information
    relating directly to the separatrix.

    Parameters
    ----------
    points:
        Points at which the ripple should be constrained.
    """

    def __init__(self, points: Coordinates):
        self.points = points


class MaximiseSelector(RipplePointSelector):
    """
    Finds and constrains the maximum ripple along the specified wire during
    each minimisation function call.
    """

    def __init__(self):
        self.points = None

    def set_wire(self, wire: BluemiraWire):
        """
        Set the wire along which the points will be selected

        Parameters
        ----------
        wire:
            Wire along which the points will be selected
        """
        super().set_wire(wire)
        points = wire.discretise(byedges=True, ndiscr=200)
        arg_x_max = np.argmax(points.x)
        x_max_point = points[:, arg_x_max]
        self._alpha_0 = wire.parameter_at(x_max_point, tolerance=EPS_FREECAD)

    def make_ripple_constraint(
        self, parameterisation, solver, TF_ripple_limit, rip_con_tol
    ) -> GeomConstraintT:
        """
        Make the ripple OptimisationConstraint
        """  # noqa: DOC201
        self.parameterisation = parameterisation
        self.solver = solver
        self.TF_ripple_limit = TF_ripple_limit
        return {
            "name": type(self).__name__,
            "f_constraint": self._constrain_max_ripple,
            "tolerance": np.full(2, rip_con_tol),
        }

    def _constrain_max_ripple(self, parameterisation: GeometryParameterisation) -> float:
        """
        Ripple constraint function

        Parameters
        ----------
        parameterisation:
            Geometry parameterisation
        """  # noqa: DOC201
        tf_wire = parameterisation.create_shape()
        self.solver.update_cage(tf_wire)

        def f_max_ripple(alpha):
            point = self._wire.value_at(alpha)
            return -self.solver.ripple(*point)

        result = optimise(
            f_max_ripple,
            x0=np.array([self._alpha_0]),
            dimensions=1,
            bounds=[(0), (1)],
            algorithm="SLSQP",
            opt_conditions={"ftol_rel": 1e-6, "max_eval": 2000},
        )

        max_ripple_point = self._wire.value_at(result.x)

        self.points = Coordinates(max_ripple_point.reshape(3, -1))
        ripple = self.solver.ripple(*self.points)
        # TODO @hsaunders1904: This print will call every time now,
        # Might be a case of explicitly
        # defining a df_constraint on this class, would be good for me to play with.
        # 3526
        bluemira_debug_flush(f"Max ripple: {ripple}")
        return ripple - self.TF_ripple_limit


@dataclass
class RippleConstrainedLengthGOPParams(ParameterFrame):
    """
    Parameters for the RippleConstrainedLengthGOP
    """

    n_TF: Parameter[int]
    R_0: Parameter[float]
    z_0: Parameter[float]
    B_0: Parameter[float]
    TF_ripple_limit: Parameter[float]


class RippleConstrainedLengthGOP(GeomOptimisationProblem):
    """
    Toroidal field coil winding pack shape optimisation problem.

    Parameters
    ----------
    parameterisation:
        Geometry parameterisation for the winding pack current centreline
    algorithm:
        Optimisation algorithm to use
    opt_conditions:
        Optimisation termination conditions dictionary
    opt_parameters:
        Optimisation parameters dictionary
    params:
        Parameters required to solve the optimisation problem
    wp_cross_section:
        Geometry of the TF coil winding pack cross-section
    separatrix:
        Separatrix shape at which the TF ripple is to be constrained
    ripple_selector: RipplePointSelector
        Selection strategy for the poitns at which to calculate ripple.
    keep_out_zone:
        Zone boundary which the WP may not enter
    rip_con_tol:
        Tolerance with which to apply the ripple constraints
    kox_con_tol:
        Tolerance with which to apply the keep-out-zone constraints
    nx:
        Number of radial Biot-Savart filaments to use
    ny:
        Number of toroidal Biot-Savart filaments to use
    n_koz_points:
        Number of discretised points to use when enforcing the keep-out-zone constraint

    Notes
    -----
    x^* = minimise: winding_pack_length
          subject to:

          .. math::
              \\text{ripple}|\\text{separatrix} \\preceq
              \\text{TF_ripple_limit SDF(wp_shape, keep_out_zone)} \\preceq 0

    The geometry parameterisation is updated in place
    """

    def __init__(
        self,
        parameterisation: GeometryParameterisation,
        algorithm: str,
        opt_conditions: dict[str, float],
        opt_parameters: dict[str, float],
        params: ParameterFrameLike,
        wp_cross_section: BluemiraWire,
        ripple_wire: BluemiraWire,
        ripple_selector: RipplePointSelector | None = None,
        keep_out_zone: BluemiraWire | None = None,
        rip_con_tol: float = 1e-3,
        koz_con_tol: float = 1e-3,
        nx: int = 1,
        ny: int = 1,
        n_koz_points: int = 100,
    ):
        self.parameterisation = parameterisation
        self.params = make_parameter_frame(params, RippleConstrainedLengthGOPParams)
        self.wp_cross_section = wp_cross_section
        self.algorithm = algorithm
        self.opt_parameters = opt_parameters
        self.opt_conditions = opt_conditions
        self.ripple_wire = ripple_wire
        if keep_out_zone:
            self._keep_out_zone = [
                KeepOutZone(
                    keep_out_zone,
                    byedges=True,
                    dl=keep_out_zone.length / 200,
                    tol=koz_con_tol,
                    shape_n_discr=n_koz_points,
                )
            ]
        else:
            self._keep_out_zone = []

        ripple_selector.set_wire(self.ripple_wire)
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
        self._ripple_constraint = ripple_selector.make_ripple_constraint(
            parameterisation, self.solver, params.TF_ripple_limit.value, rip_con_tol
        )
        self.ripple_selector = ripple_selector

    @staticmethod
    def objective(parameterisation: GeometryParameterisation) -> float:
        """
        Objective function (minimise length)
        """  # noqa: DOC201
        return parameterisation.create_shape().length

    def keep_out_zones(self) -> list[KeepOutZone]:
        """
        Keep out zone
        """  # noqa: DOC201
        return self._keep_out_zone

    def ineq_constraints(self) -> GeomConstraintT:
        """
        Inequality constraints
        """  # noqa: DOC201
        return [self._ripple_constraint]

    def optimise(self) -> GeometryParameterisation:
        """
        Solve the GeometryOptimisationProblem.
        """  # noqa: DOC201
        self.parameterisation = (
            super()
            .optimise(
                self.parameterisation,
                algorithm=self.algorithm,
                opt_conditions=self.opt_conditions,
                opt_parameters=self.opt_parameters,
            )
            .geom
        )

        self.solver.update_cage(self.parameterisation.create_shape())
        self.ripple_values = self.solver.ripple(*self.ripple_selector.points)
        if isinstance(self.ripple_values, float):
            self.ripple_values = np.array([self.ripple_values])
        return self.parameterisation

    def plot(self, ax: plt.Axes | None = None):
        """
        Plot the optimisation problem.

        Parameters
        ----------
        ax:
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
        if ax is None:
            _f, ax = plt.subplots()

        plot_2d(
            self.ripple_wire,
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

        for koz in self._keep_out_zone:
            plot_2d(
                koz.wire,
                ax=ax,
                show=False,
                wire_options={"color": "k", "linewidth": 0.5},
            )

        rv = self.ripple_values
        norm = mpl.colors.Normalize()
        norm.autoscale(rv)
        cm = mpl.cm.viridis
        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        vmin, vmax = np.min(rv) - 1e-6, np.max(rv) + 1e-6
        sm.set_clim(vmin, vmax)
        ax.scatter(
            self.ripple_selector.points.x,
            self.ripple_selector.points.z,
            color=cm(norm(rv)),
        )
        color_bar = plt.colorbar(sm, ax=ax)
        color_bar.ax.set_ylabel("Toroidal field ripple [%]")

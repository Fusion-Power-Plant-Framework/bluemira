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
Built-in build steps for making a parameterised plasma
"""

import numpy as np
from typing import Dict, List, Tuple, Type, Union

from bluemira.base.components import Component, PhysicalComponent
import bluemira.geometry as geo
from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.magnetostatics.circuits import HelmholtzCage
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.utilities.tools import get_module
from bluemira.utilities.optimiser import Optimiser

from bluemira.builders.shapes import ParameterisedShapeBuilder


class RippleConstrainedLengthOpt(GeometryOptimisationProblem):
    """
    Toroidal field coil winding pack shape optimisation problem

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

    Notes
    -----
    x^* = minimise: winding_pack_length
          subject to:
              ripple|separatrix < TF_ripple_limit
              SDF(wp_shape, keep_out_zone) \\prereq 0

    The geometry parameterisation is updated in place
    """

    def __init__(
        self,
        parameterisation,
        optimiser,
        params,
        separatrix,
        keep_out_zone=None,
        n_koz_points=100,
    ):
        super().__init__(parameterisation, optimiser)
        self.params = params
        self.separatrix = separatrix
        self.keep_out_zone = keep_out_zone

        self.ripple_points = self._make_ripple_points(separatrix)
        self.ripple_values = None

        self.optimiser.add_ineq_constraints(
            self.f_constrain_ripple, 1e-3 * np.ones(len(self.ripple_points[0]))
        )

        # self.optimiser.add_ineq_constraints(
        #     parameterisation.shape_constraints, np.zeros(1)
        # )

        if self.keep_out_zone:
            self.n_koz_points = n_koz_points
            self.koz_points = self._make_koz_points(keep_out_zone)

            self.optimiser.add_ineq_constraints(
                self.f_constrain_koz, 1e-3 * np.ones(n_koz_points)
            )

    def _make_koz_points(self, keep_out_zone):
        return keep_out_zone.discretize(byedges=True, dl=keep_out_zone.length / 200)[
            :, [0, 2]
        ]

    def _make_ripple_points(self, separatrix):
        points = separatrix.discretize(ndiscr=100).T
        # idx = np.where(points[0] > self.params.R_0.value)[0]
        return points  # [:, idx]

    def update_cage(self, x):
        """
        Update the magnetostatic solver
        """
        super().update_parameterisation(x)
        wire = self.parameterisation.create_shape()
        points = wire.discretize(byedges=True, dl=wire.length / 200).T

        self.cage = HelmholtzCage(
            BiotSavartFilament(points.T, radius=1, current=1), self.params.n_TF.value
        )
        field = self.cage.field(self.params.R_0, 0, self.params.z_0)
        current = -self.params.B_0 / field[1]  # single coil amp-turns
        self.cage.set_current(current)

    def calculate_ripple(self, x):
        """
        Calculate the ripple on the target points for a given variable vector
        """
        self.update_cage(x)
        ripple = self.cage.ripple(*self.ripple_points)
        print(max(ripple))
        self.ripple_values = ripple
        return ripple - self.params.TF_ripple_limit

    def f_constrain_ripple(self, constraint, x, grad):
        """
        Toroidal field ripple constraint function
        """
        constraint[:] = self.calculate_ripple(x)

        if grad.size > 0:
            # Only called if a gradient-based optimiser is used
            grad[:] = self.optimiser.approx_derivative(
                self.calculate_ripple, x, constraint
            )

        return constraint

    def calculate_signed_distance(self, x):
        self.update_cage(x)
        shape = self.parameterisation.create_shape()
        s = shape.discretize(ndiscr=self.n_koz_points)[:, [0, 2]]
        return geo.tools.signed_distance_2D_polygon(s, self.koz_points)

    def f_constrain_koz(self, constraint, x, grad):
        """
        Geometry constraint function
        """
        constraint[:] = self.calculate_signed_distance(x)

        if grad.size > 0:
            grad[:] = self.optimiser.approx_derivative(
                self.calculate_signed_distance, x, constraint
            )
        return constraint

    def calculate_length(self, x):
        """
        Calculate the length of the GeometryParameterisation
        """
        self.update_parameterisation(x)
        return self.parameterisation.create_shape().length

    def f_objective(self, x, grad):
        """
        Length minimisation objective
        """
        length = self.calculate_length(x)

        if grad.size > 0:
            # Only called if a gradient-based optimiser is used
            grad[:] = self.optimiser.approx_derivative(
                self.calculate_length, x, f0=length
            )

        return length

    # I just need to see... I worry about the proliferation of Plotters
    def plot(self, ax=None, **kwargs):
        """
        Plot the optimisation problem.

        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
        import matplotlib.pyplot as plt
        import matplotlib
        from bluemira.display import plot_2d

        if ax is None:
            ax = kwargs.get("ax", plt.gca())

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

        # Yet again... CCW default one of the main motivations of Loop
        xpl, zpl = self.ripple_points[0, :][::-1], self.ripple_points[2, :][::-1]
        rv = self.ripple_values[::-1]
        dx, dz = rv * np.gradient(xpl), rv * np.gradient(zpl)
        norm = matplotlib.colors.Normalize()
        norm.autoscale(rv)
        cm = matplotlib.cm.viridis
        sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        ax.quiver(
            xpl,
            zpl,
            dz,
            -dx,
            color=cm(norm(rv)),
            headaxislength=0,
            headlength=0,
            width=0.02,
        )
        color_bar = plt.gcf().colorbar(sm)
        color_bar.ax.set_ylabel("Toroidal field ripple [%]")


class MakeOptimisedTFWindingPack(ParameterisedShapeBuilder):
    """
    A class that optimises a TF winding pack based on a parameterised shape
    """

    _required_config = ParameterisedShapeBuilder._required_config + [
        "targets",
        "segment_angle",
        "problem_class",
    ]

    _param_class: Type[GeometryParameterisation]
    _variables_map: Dict[str, str]
    _targets: Dict[str, str]
    _problem_class: Type[GeometryOptimisationProblem]

    def _extract_config(self, build_config: Dict[str, Union[float, int, str]]):
        def get_problem_class(class_path: str) -> Type[GeometryOptimisationProblem]:
            if "::" in class_path:
                module, class_name = class_path.split("::")
            else:
                class_path_split = class_path.split(".")
                module, class_name = (
                    ".".join(class_path_split[:-1]),
                    class_path_split[-1],
                )
            return getattr(get_module(module), class_name)

        super()._extract_config(build_config)

        self._targets = build_config["targets"]
        self._segment_angle: float = build_config["segment_angle"]
        self._problem_class = get_problem_class(build_config["problem_class"])
        self._algorithm_name = build_config.get("algorithm_name", "SLSQP")
        self._opt_conditions = build_config.get("opt_conditions", {"max_eval": 100})
        self._opt_parameters = build_config.get("opt_parameters", {})

    def build(self, params, **kwargs) -> List[Tuple[str, Component]]:
        """
        Build a TF using the requested targets and methods.
        """
        super().build(params, **kwargs)

        boundary = self.optimise()

        result_components = []
        for target, func in self._targets.items():
            result_components.append(getattr(self, func)(boundary, target))

        return result_components

    def optimise(self):
        """
        Optimise the shape using the provided parameterisation and optimiser.
        """
        shape = self.create_parameterisation()
        optimiser = Optimiser(
            self._algorithm_name,
            shape.variables.n_free_variables,
            self._opt_conditions,
            self._opt_parameters,
        )
        problem = self._problem_class(shape, optimiser)
        problem.solve()
        return shape.create_shape()

    def build_xz(self, boundary: geo.wire.BluemiraWire, target: str):
        """
        Build the boundary as a wire at the requested target.
        """
        label = target.split("/")[-1]
        return (
            target,
            PhysicalComponent(label, geo.wire.BluemiraWire(boundary, label)),
        )

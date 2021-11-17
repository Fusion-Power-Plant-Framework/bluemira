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

from typing import Any, Dict, List, Tuple, Type, Union
import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
import bluemira.geometry as geo
from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.tools import get_module

from bluemira.builders.shapes import ParameterisedShapeBuilder
from bluemira.magnetostatics.circuits import HelmholtzCage
from bluemira.magnetostatics.biot_savart import BiotSavartFilament


class TFWPOptimisationProblem(GeometryOptimisationProblem):
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
    x_* = minimise: winding_pack_length
          subject to:
              ripple|separatrix <= TF_ripple_limit

    The geometry parameterisation is updated in place
    """

    def __init__(self, parameterisation, optimiser, params, separatrix):
        super().__init__(parameterisation, optimiser)
        self.params = params
        self.separatrix = separatrix
        self.ripple_points = self._make_ripple_points(separatrix)
        self.ripple_values = None
        self.optimiser.add_ineq_constraints(
            self.f_constrain_ripple, 1e-3 * np.ones(len(self.ripple_points[0]))
        )

    def _make_ripple_points(self, separatrix):
        points = separatrix.discretize(ndiscr=100).T
        idx = np.where(points[0] > self.params.R_0.value)[0]
        return points[:, idx]

    def update_cage(self, x):
        """
        Update the magnetostatic solver
        """
        super().update_parameterisation(x)
        points = self.parameterisation.create_array(200)
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
    def plot_ripple(self, ax=None, **kwargs):
        """
        Plot the ripple along the separatrix loop.

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
        plot_2d(self.parameterisation.create_shape(), ax=ax, show=False)

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
        return sm


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


class BuildTFCoils(Builder):
    """
    A class to build TF coils in the same way as BLUEPRINT.
    """

    _required_config = [...]
    _required_params = [...]

    def __init__(self, params, build_config: Dict[str, Any], **kwargs):
        super().__init__(params, build_config, **kwargs)


if __name__ == "__main__":

    # Sorry for the script... I needed to check if this was working
    from bluemira.geometry.parameterisations import PrincetonD
    from bluemira.equilibria.shapes import JohnerLCFS
    from bluemira.base.parameter import ParameterFrame

    parameterisation = PrincetonD(
        {
            "x1": {"lower_bound": 2, "value": 4, "upper_bound": 6},
            "x2": {"lower_bound": 10, "value": 14, "upper_bound": 18},
            "dz": {"lower_bound": -0.5, "value": 0, "upper_bound": 0.5},
        }
    )
    parameterisation.fix_variable("x1", 4)
    parameterisation.fix_variable("dz", 0)
    optimiser = Optimiser(
        "SLSQP",
        opt_conditions={
            "ftol_rel": 1e-3,
            "xtol_rel": 1e-12,
            "xtol_abs": 1e-12,
            "max_eval": 1000,
        },
    )

    # I just don't know where to get these any more
    params = ParameterFrame(
        [
            ["R_0", "Major radius", 9, "m", None, "Input", None],
            ["z_0", "Vertical height at major radius", 0, "m", None, "Input", None],
            ["B_0", "Toroidal field at R_0", 6, "T", None, "Input", None],
            ["n_TF", "Number of TF coils", 16, "N/A", None, "Input", None],
            ["TF_ripple_limit", "TF coil ripple limit", 0.6, "%", None, "Input", None],
        ]
    )

    separatrix = JohnerLCFS(
        {
            "r_0": {"value": 9},
            "z_0": {"value": 0},
            "a": {"value": 9 / 3.1},
            "kappa_u": {"value": 1.65},
            "kappa_l": {"value": 1.8},
        }
    ).create_shape()

    # Need to pass around lots of information between different parts of the build
    # procedure.
    # This is just the bare minimum TF optimisation, we don't have much in the way of
    # configuration yet, and we're missing geometry constraints from some arbitrary keep
    # out zone. Also the KOZ constraint should be enforced on the plasma-facing casing
    # geometry, which needs to be built off the winding pack. Gonna get messy again :D

    # Starting to worry we're making things too configurable:
    #   - what about different magnetostatics solvers
    #   - different discretisations if we use BiotSavart
    #   - different separatrix shapes need to be checked at different areas for peak
    #     ripple..

    # Keeping ultra-configurable classes is going to slow us down.
    # Might be simpler just to have a SystemBuilder that people subclass or write
    # replacements for, I don't know.

    # I fear the full build config just for the TF coil WP design optimisation will be
    # absolutely massive.
    problem = TFWPOptimisationProblem(parameterisation, optimiser, params, separatrix)
    problem.solve()

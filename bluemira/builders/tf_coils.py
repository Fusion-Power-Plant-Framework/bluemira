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
from bluemira.display.displayer import show_cad
import bluemira.geometry as geo
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.tools import get_module

from bluemira.builders.shapes import ParameterisedShapeBuilder
from bluemira.magnetostatics.circuits import HelmholtzCage
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.geometry.tools import signed_distance_2D_polygon


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

        self.optimiser.add_ineq_constraints(
            parameterisation.shape_constraints, np.zeros(1)
        )

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
        idx = np.where(points[0] > self.params.R_0.value)[0]
        return points[:, idx]

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
        return signed_distance_2D_polygon(s, self.koz_points)

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


from bluemira.geometry.tools import (
    sweep_shape,
    make_polygon,
    offset_wire,
    circular_pattern,
    boolean_cut,
)
from bluemira.base.parameter import ParameterFrame
from bluemira.base.constants import MU_0


class BuildTFWindingPack:  # (ActualFuckingComponent)
    """
    A class to build TF coil winding pack geometry
    """

    name = "TFWindingPack"

    def __init__(self, wp_centreline, wp_cross_section):
        self.wp_centreline = wp_centreline
        self.wp_cross_section = wp_cross_section

    def build_xy(self):
        pass

    def build_xz(self):
        # Christ... this is only robust because we know the shape isn't nuts
        x = self.wp_cross_section.discretize(100).T[0]
        x_in = min(x)
        x = self.wp_centreline.discretize(100).T[0]
        x_centreline_in = min(x)
        dx = x_centreline_in - x_in
        outer = offset_wire(self.wp_centreline, dx, join="arc")
        inner = offset_wire(self.wp_centreline, -dx, join="arc")
        # Why do we have two labels, and why do we return target if it is an input?
        return PhysicalComponent(self.name, BluemiraFace([outer, inner], self.name))

    def build_xyz(self):
        solid = sweep_shape(self.wp_cross_section, self.wp_centreline, label=self.name)
        return PhysicalComponent(self.name, solid)


class BuildTFInsulation:
    name = "TFWPInsulation"

    def __init__(self, wp_solid, wp_centreline, wp_cross_section, insulation_thickness):
        self.wp_solid = wp_solid
        self.wp_centreline = wp_centreline
        self.wp_cross_section = wp_cross_section
        self.tk_insulation = insulation_thickness

    def build_xz(self):
        x = self.wp_centreline.discretize(100).T[0]
        x_centreline_in = min(x)

        x_wp = self.wp_cross_section.discretize(100).T[0]
        x_in_wp = min(x_wp)
        x_out_wp = max(x_wp)
        dx_wp = x_centreline_in - x_in_wp

        ins_xs = offset_wire(self.wp_cross_section, self.tk_insulation)
        x = ins_xs.discretize(100).T[0]
        x_in_ins = min(x)
        x_out_ins = max(x)

        dx_ins = x_centreline_in - x_in_ins
        outer = offset_wire(self.wp_centreline, dx_ins, join="arc")
        inner = offset_wire(self.wp_centreline, dx_wp, join="arc")

        outer_face = BluemiraFace([outer, inner])

        outer = offset_wire(self.wp_centreline, -dx_wp, join="arc")
        inner = offset_wire(self.wp_centreline, -dx_ins, join="arc")
        inner_face = BluemiraFace([outer, inner])
        # Why do we have two labels, and why do we return target if it is an input?
        return [
            PhysicalComponent(self.name, outer_face, self.name),
            PhysicalComponent(self.name, inner_face, self.name),
        ]

    def build_xyz(self):
        ins_xs = offset_wire(self.wp_cross_section, self.tk_insulation)

        solid = sweep_shape(ins_xs, self.wp_centreline)
        # This doesnt frigging work
        ins_solid = boolean_cut(solid, self.wp_solid)
        return PhysicalComponent(self.name, ins_solid)


class BuildTFCasing:
    pass


class BuildTFCoils(Builder):
    """
    A class to build TF coils in the same way as BLUEPRINT.
    """

    _required_config = Builder._required_config + ["targets"]
    _required_params = [
        "R_0",
        "B_0",
        "n_TF",
        "r_tf_in",
        "tk_tf_wp",
        "tk_tf_nose",
        "tf_wp_depth",
    ]

    def __init__(self, params, build_config: Dict[str, Any], **kwargs):
        super().__init__(params, build_config, **kwargs)


class ToroidalFieldSystem:
    def __init__(self, params, wp_parameterisation):
        self.params = params
        self.wp_parameterisation = wp_parameterisation

    def build(self):
        # TODO: I see that nobody ever got to the bottom of the PROCESS insulation story
        r_wp_centroid = self.params.r_tf_in + self.params.tk_tf_wp
        self.wp_parameterisation.adjust_variable("x1", r_wp_centroid)
        wp_xs = self.make_wp_cross_section(r_wp_centroid)
        builder = BuildTFWindingPack(self.wp_parameterisation.create_shape(), wp_xs)
        builder.build()

        builder = BuildTFInsulation(
            self.wp_parameterisation.create_shape(), wp_xs, self.params.tk_tf_ins
        )

        builder = BuildTFCasing()
        builder.build()

    def optimise(self):
        pass

    def calculate_wp_current(self):
        # Back of the envelope
        bm = -self.params.B_0 * self.params.R_0
        current = abs(2 * np.pi * bm / (self.params.n_TF * MU_0))
        self.params.add_parameter(
            "I_tf", "TF coil current", current, "A", None, "bluemira"
        )

    def make_wp_cross_section(self, r_wp_centroid):
        r_wp_in = r_wp_centroid - 0.5 * self.params.tk_tf_wp
        r_wp_out = r_wp_centroid + 0.5 * self.params.tk_tf_wp
        y_down = -0.5 * self.params.tf_wp_depth
        y_up = -y_down
        return BluemiraFace(
            make_polygon(
                [
                    [r_wp_in, y_down, 0],
                    [r_wp_out, y_down, 0],
                    [r_wp_out, y_up, 0],
                    [r_wp_in, y_up, 0],
                ],
                closed=True,
            )
        )


if __name__ == "__main__":

    from bluemira.geometry.parameterisations import PrincetonD, TripleArc, PolySpline
    from bluemira.equilibria.shapes import JohnerLCFS
    from bluemira.base.parameter import ParameterFrame

    x_tf_wp_center = 3.2
    parameterisation = PrincetonD(
        {
            "x1": {"value": x_tf_wp_center, "fixed": True},
            "x2": {"lower_bound": 10, "value": 14, "upper_bound": 18},
            "dz": {"lower_bound": -0.5, "value": 0, "upper_bound": 0.5, "fixed": True},
        }
    )

    parameterisation = TripleArc(
        {
            "x1": {"value": x_tf_wp_center, "fixed": True},
            "z1": {"value": -2, "lower_bound": -2, "fixed": True},
        }
    )

    # parameterisation = PolySpline(
    #     {
    #         "x1": {
    #             "value": x_tf_wp_center,
    #             "lower_bound": 3.999,
    #             "upper_bound": 4.001,
    #             "fixed": True,
    #         },
    #         "x2": {"value": 16, "lower_bound": 13, "upper_bound": 20},
    #         "z2": {"value": 0, "lower_bound": -0.9, "upper_bound": 0.9},
    #         "height": {"value": 18, "lower_bound": 10, "upper_bound": 20},
    #         "top": {"value": 0.5, "lower_bound": 0.05, "upper_bound": 1},
    #         "upper": {"value": 0.7, "lower_bound": 0.2, "upper_bound": 1},
    #         "tilt": {"value": 0, "fixed": True},
    #         "flat": {"value": 0, "fixed": True},
    #         # "f1": {"value": 4, "lower_bound": 4},
    #         # "f2": {"value": 4, "lower_bound": 4},
    #     }
    # )

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

    from bluemira.geometry.tools import offset_wire

    koz = offset_wire(separatrix, 2.0, join="arc")

    problem = TFWPOptimisationProblem(
        parameterisation, optimiser, params, separatrix, koz
    )
    problem.solve()

    centreline = parameterisation.create_shape()
    x_c = 4
    d_xc = 0.25
    d_yc = 0.6
    wp_xs = make_polygon(
        [
            [x_c - d_xc, -d_yc, 0],
            [x_c + d_xc, -d_yc, 0],
            [x_c + d_xc, d_yc, 0],
            [x_c - d_xc, d_yc, 0],
        ],
        closed=True,
    )

    from bluemira.geometry.tools import sweep_shape, circular_pattern, revolve_shape
    from bluemira.display import show_cad
    from bluemira.geometry.parameterisations import PictureFrame
    from bluemira.geometry.face import BluemiraFace
    from bluemira.display.displayer import DisplayCADOptions

    tf_wp = sweep_shape(wp_xs, centreline)
    shapes = circular_pattern(tf_wp, n_shapes=16)
    options = 16 * [DisplayCADOptions(color=(0.2, 0.3, 0.4))]
    plasma = revolve_shape(BluemiraFace(separatrix), degree=360)
    shapes.append(plasma)
    options.append(DisplayCADOptions(color=(1.0, 0.2, 0.5), transparency=0.5))

    pf1 = PictureFrame(
        {
            "x1": {"value": 6, "lower_bound": 6, "upper_bound": 6},
            "x2": {"value": 7, "lower_bound": 7, "upper_bound": 7},
            "z1": {"value": -6, "lower_bound": -6, "upper_bound": -6},
            "z2": {"value": -7, "lower_bound": -7, "upper_bound": -7},
            "ri": {"value": 0.0, "lower_bound": 0.0},
            "ro": {"value": 0.0, "lower_bound": 0.0},
        }
    ).create_shape()
    pf1 = BluemiraFace(pf1)
    pf1 = revolve_shape(pf1, degree=359)
    shapes.append(pf1)
    options.append(DisplayCADOptions(color=(0.2, 0.2, 0.6)))

    show_cad(shapes, options)
    # from bluemira.geometry.parameterisations import PrincetonD, TripleArc
    # from bluemira.geometry.tools import make_polygon

    # x_wp_centroid = 4.0
    # dx_wp = 0.5
    # dy_wp = 0.6
    # tk_ins = 0.3
    # # Offset wire is sadly very unstable...
    # wp_centreline = TripleArc({"x1": {"value": x_wp_centroid}}).create_shape()
    # wp_xs = make_polygon(
    #     [
    #         [x_wp_centroid - dx_wp, -dy_wp, 0],
    #         [x_wp_centroid + dx_wp, -dy_wp, 0],
    #         [x_wp_centroid + dx_wp, dy_wp, 0],
    #         [x_wp_centroid - dx_wp, dy_wp, 0],
    #     ],
    #     closed=True,
    # )

    # builder = BuildTFWindingPack(wp_centreline, wp_xs)

    # outer = offset_wire(wp_centreline, -dx_wp)
    # inner = offset_wire(wp_centreline, dx_wp)
    # xz_shape = builder.build_xz()
    # xyz_shape = builder.build_xyz()
    # xz_shape.plot_2d()
    # xyz_shape.show_cad()

    # builder = BuildTFInsulation(xyz_shape, wp_centreline, wp_xs, tk_ins)
    # xz_ins_shape = builder.build_xz()

    # xz_shapes = [xz_shape]

    # xz_shapes.extend(xz_ins_shape)

    # import matplotlib.pyplot as plt

    # f, ax = plt.subplots()
    # for shape in xz_shapes:
    #     shape.plot_2d(ax=ax, show=False)

    # shapes = circular_pattern(xyz_shape.shape, n_shapes=16)
    # show_cad(shapes)

    # # # Sorry for the script... I needed to check if this was working
    # # from bluemira.geometry.parameterisations import PrincetonD
    # # from bluemira.equilibria.shapes import JohnerLCFS
    # # from bluemira.base.parameter import ParameterFrame

    # # parameterisation = PrincetonD(
    # #     {
    # #         "x1": {"lower_bound": 2, "value": 4, "upper_bound": 6},
    # #         "x2": {"lower_bound": 10, "value": 14, "upper_bound": 18},
    # #         "dz": {"lower_bound": -0.5, "value": 0, "upper_bound": 0.5},
    # #     }
    # # )
    # # parameterisation.fix_variable("x1", 4)
    # # parameterisation.fix_variable("dz", 0)
    # # optimiser = Optimiser(
    # #     "SLSQP",
    # #     opt_conditions={
    # #         "ftol_rel": 1e-3,
    # #         "xtol_rel": 1e-12,
    # #         "xtol_abs": 1e-12,
    # #         "max_eval": 1000,
    # #     },
    # # )

    # # # I just don't know where to get these any more
    # # params = ParameterFrame(
    # #     [
    # #         ["R_0", "Major radius", 9, "m", None, "Input", None],
    # #         ["z_0", "Vertical height at major radius", 0, "m", None, "Input", None],
    # #         ["B_0", "Toroidal field at R_0", 6, "T", None, "Input", None],
    # #         ["n_TF", "Number of TF coils", 16, "N/A", None, "Input", None],
    # #         ["TF_ripple_limit", "TF coil ripple limit", 0.6, "%", None, "Input", None],
    # #     ]
    # # )

    # # separatrix = JohnerLCFS(
    # #     {
    # #         "r_0": {"value": 9},
    # #         "z_0": {"value": 0},
    # #         "a": {"value": 9 / 3.1},
    # #         "kappa_u": {"value": 1.65},
    # #         "kappa_l": {"value": 1.8},
    # #     }
    # # ).create_shape()

    # # # Need to pass around lots of information between different parts of the build
    # # # procedure.
    # # # This is just the bare minimum TF optimisation, we don't have much in the way of
    # # # configuration yet, and we're missing geometry constraints from some arbitrary keep
    # # # out zone. Also the KOZ constraint should be enforced on the plasma-facing casing
    # # # geometry, which needs to be built off the winding pack. Gonna get messy again :D

    # # # Starting to worry we're making things too configurable:
    # # #   - what about different magnetostatics solvers
    # # #   - different discretisations if we use BiotSavart
    # # #   - different separatrix shapes need to be checked at different areas for peak
    # # #     ripple..

    # # # Keeping ultra-configurable classes is going to slow us down.
    # # # Might be simpler just to have a SystemBuilder that people subclass or write
    # # # replacements for, I don't know.

    # # # I fear the full build config just for the TF coil WP design optimisation will be
    # # # absolutely massive.
    # # problem = TFWPOptimisationProblem(parameterisation, optimiser, params, separatrix)
    # # problem.solve()

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
    keep_out_zone: Optional[BluemiraWire]
        Zone boundary which the WP may not enter

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
        wp_cross_section,
        separatrix,
        keep_out_zone=None,
        nx=2,
        ny=2,
        n_koz_points=100,
    ):
        super().__init__(parameterisation, optimiser)
        self.params = params
        self.separatrix = separatrix
        self.wp_cross_section = wp_cross_section
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

        self.nx = nx
        self.ny = ny

    def _make_koz_points(self, keep_out_zone):
        return keep_out_zone.discretize(byedges=True, dl=keep_out_zone.length / 200)[
            :, [0, 2]
        ]

    def _make_ripple_points(self, separatrix):
        points = separatrix.discretize(ndiscr=100).T
        # idx = np.where(points[0] > self.params.R_0.value)[0]
        return points  # [:, idx]

    def _update_single_circuit(self, wire):
        bb = self.wp_cross_section.bounding_box
        dx_xs = 0.5 * (bb.x_max - bb.x_min)
        dy_xs = 0.5 * (bb.y_max - bb.y_min)

        if self.nx > 1:
            dx_wp = np.linspace(
                dx_xs * (1 / self.nx - 1), dx_xs * (1 - 1 / self.nx), self.nx
            )
        else:
            dx_wp = [0]  # coil centreline

        if self.ny > 1:
            dy_wp = np.linspace(
                dy_xs * (1 / self.ny - 1), dy_xs * (1 - 1 / self.ny), self.ny
            )
        else:
            dy_wp = [0]  # coil centreline

        current_wires = []
        for dx in dx_wp:
            c_wire = offset_wire(wire, dx)
            for dy in dy_wp:
                from copy import deepcopy

                c_w = deepcopy(c_wire)
                c_w.translate((0, dy, 0))
                current_wires.append(c_w)

        current_arrays = [
            w.discretize(byedges=True, dl=wire.length / 200) for w in current_wires
        ]

        # We need all arrays to be CCW
        from bluemira.geometry._deprecated_tools import check_ccw

        for c in current_arrays:
            if not check_ccw(c[:, 0], c[:, 2]):
                c[:, 0] = c[:, 0][::-1]
                c[:, 2] = c[:, 2][::-1]

        radius = 0.5 * self.wp_cross_section.area / (self.nx * self.ny)
        filament = BiotSavartFilament(
            current_arrays, radius=radius, current=1 / (self.nx * self.ny)
        )
        return filament

    def update_cage(self, x):
        """
        Update the magnetostatic solver
        """
        super().update_parameterisation(x)
        wire = self.parameterisation.create_shape()
        circuit = self._update_single_circuit(wire)

        self.cage = HelmholtzCage(circuit, self.params.n_TF.value)
        field = self.cage.field(self.params.R_0, 0, self.params.z_0)
        current = -self.params.B_0 / field[1]  # single coil amp-turns
        current /= self.nx * self.ny  # single filament amp-turns
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


from bluemira.geometry.tools import (
    sweep_shape,
    make_polygon,
    offset_wire,
    circular_pattern,
    boolean_cut,
)
from bluemira.base.parameter import ParameterFrame
from bluemira.base.constants import MU_0


class BuildTFWindingPack:
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
        x_min = self.wp_cross_section.bounding_box.x_min
        x_centreline_in = self.wp_centreline.bounding_box.x_min
        dx = abs(x_min - x_centreline_in)
        outer = offset_wire(self.wp_centreline, dx)
        inner = offset_wire(self.wp_centreline, -dx)
        return PhysicalComponent(self.name, BluemiraFace([outer, inner], self.name))

    def build_xyz(self):
        solid = sweep_shape(
            self.wp_cross_section.boundary[0], self.wp_centreline, label=self.name
        )
        return PhysicalComponent(self.name, solid)


class BuildTFInsulation:
    name = "TFWPInsulation"

    def __init__(self, wp_solid, wp_centreline, wp_cross_section, insulation_thickness):
        self.wp_solid = wp_solid
        self.wp_centreline = wp_centreline
        self.wp_cross_section = wp_cross_section
        self.tk_insulation = insulation_thickness

    def build_xy(self):
        outer_wire = offset_wire(self.wp_cross_section, self.tk_insulation)
        face = BluemiraFace([outer_wire, self.wp_cross_section])
        return PhysicalComponent(self.name, face)

    def build_xz(self):
        x_centreline_in = self.wp_centreline.bounding_box.x_min

        x_in_wp = self.wp_cross_section.bounding_box.x_min

        dx_wp = x_centreline_in - x_in_wp

        ins_xs = offset_wire(self.wp_cross_section, self.tk_insulation)
        x_in_ins = ins_xs.bounding_box.x_min

        dx_ins = x_centreline_in - x_in_ins
        outer = offset_wire(self.wp_centreline, dx_ins)
        inner = offset_wire(self.wp_centreline, dx_wp)

        outer_face = BluemiraFace([outer, inner])

        outer = offset_wire(self.wp_centreline, -dx_wp)
        inner = offset_wire(self.wp_centreline, -dx_ins)
        inner_face = BluemiraFace([outer, inner])
        return [
            PhysicalComponent(self.name, outer_face),
            PhysicalComponent(self.name, inner_face),
        ]

    def build_xyz(self):
        ins_xs = offset_wire(self.wp_cross_section, self.tk_insulation)

        solid = sweep_shape(ins_xs, self.wp_centreline)
        # This doesnt frigging work
        ins_solid = boolean_cut(solid, self.wp_solid)
        return PhysicalComponent(self.name, ins_solid)


class BuildTFCasing:
    name = "TFCasing"

    def __init__(
        self,
        ins_solid,
        wp_centreline,
        ins_cross_section,
        n_TF,
        tk_tf_nose,
        tk_tf_front_ib,
        tk_tf_side,
    ):
        self.ins_solid = ins_solid
        self.ins_cross_section = ins_cross_section
        self.wp_centreline = wp_centreline
        self.n_TF = n_TF
        self.tk_tf_nose = tk_tf_nose
        self.tk_tf_front_ib = tk_tf_front_ib
        self.tk_tf_side = tk_tf_side

    def build_xy(self):
        x_ins_in = self.ins_cross_section.bounding_box.x_min
        x_ins_out = self.ins_cross_section.bounding_box.x_max

        x_in = x_ins_in - self.tk_tf_nose
        x_out = x_ins_out + self.tk_tf_front_ib
        half_angle = np.pi / self.n_TF
        y_in = x_in * np.sin(half_angle)
        y_out = x_out * np.sin(half_angle)
        outer_wire = make_polygon(
            [[x_in, -y_in, 0], [x_out, -y_out, 0], [x_out, y_out, 0], [x_in, y_in, 0]],
            closed=True,
        )
        inner_face = BluemiraFace([outer_wire, self.ins_cross_section.boundary[0]])

        # Should normally be gotten with wire_plane_intersect
        x_out = self.wp_centreline.bounding_box.x_max
        # Split the total radial thickness equally on the outboard
        tk_total = self.tk_tf_front_ib + self.tk_tf_nose
        tk = 0.5 * tk_total
        outer_wire = make_polygon(
            [
                [x_out - tk, -self.tk_tf_side, 0],
                [x_out + tk, -self.tk_tf_side, 0],
                [x_out + tk, self.tk_tf_side, 0],
                [x_out - tk, self.tk_tf_side, 0],
            ],
            closed=True,
        )
        outer_ins = self.ins_cross_section.deepcopy()
        outer_ins.translate((x_out - outer_ins.center_of_mass[0], 0, 0))

        outer_face = BluemiraFace([outer_wire, outer_ins])
        return [
            PhysicalComponent(self.name, inner_face),
            PhysicalComponent(self.name, outer_face),
        ]

    def build_xz(self):
        pass

    def build_xyz(self):
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

    from bluemira.geometry.parameterisations import PrincetonD, TripleArc, PictureFrame
    from bluemira.geometry.face import BluemiraFace
    from bluemira.equilibria.shapes import JohnerLCFS
    from bluemira.base.parameter import ParameterFrame
    from bluemira.geometry.tools import sweep_shape, circular_pattern, revolve_shape
    from bluemira.display import show_cad
    from bluemira.display.displayer import DisplayCADOptions

    x_tf_wp_center = 3.2
    parameterisation = PrincetonD(
        {
            "x1": {"value": x_tf_wp_center, "fixed": True},
            "x2": {"lower_bound": 10, "value": 14, "upper_bound": 18},
            "dz": {"lower_bound": -0.5, "value": 0, "upper_bound": 0.5, "fixed": True},
        }
    )

    # parameterisation = TripleArc(
    #     {
    #         "x1": {"value": x_tf_wp_center, "fixed": True},
    #         "z1": {"value": 0, "lower_bound": -2, "fixed": True},
    #     }
    # )

    x_c = x_tf_wp_center
    d_xc = 0.25
    d_yc = 0.4
    tk_ins = 0.05
    wp_xs = make_polygon(
        [
            [x_c - d_xc, -d_yc, 0],
            [x_c + d_xc, -d_yc, 0],
            [x_c + d_xc, d_yc, 0],
            [x_c - d_xc, d_yc, 0],
        ],
        closed=True,
    )
    wp_xs = BluemiraFace(wp_xs, "TF WP x-y cross-section")

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
        parameterisation, optimiser, params, wp_xs, separatrix, koz
    )
    problem.solve()

    centreline = parameterisation.create_shape()

    tf_wp = sweep_shape(wp_xs, centreline)
    shapes = circular_pattern(tf_wp, n_shapes=16)
    options = 16 * [DisplayCADOptions(color=(0.2, 0.3, 0.4))]
    plasma = revolve_shape(BluemiraFace(separatrix), degree=360)
    shapes.append(plasma)
    options.append(DisplayCADOptions(color=(1.0, 0.2, 0.5), transparency=0.5))

    show_cad(shapes, options)

    wp_centreline = parameterisation.create_shape()

    builder = BuildTFWindingPack(wp_centreline, wp_xs)
    xz_comp = builder.build_xz()
    xyz_shape = builder.build_xyz().shape
    # xz.plot_2d()

    builder = BuildTFInsulation(xyz_shape, wp_centreline, wp_xs, tk_ins)
    xz_ins_comp = builder.build_xz()
    xy_ins_shape = builder.build_xy().shape
    # xyz_ins_shape = builder.build_xyz().shape
    builder = BuildTFCasing(None, wp_centreline, xy_ins_shape, 16, 0.4, 0.1, 0.1)
    xy_casing = builder.build_xy()

    xz_comps = [xz_comp]

    xz_comps.extend(xz_ins_comp)

    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    for shape in xz_comps:
        shape.plot_2d(ax=ax, show=False)

    # shapes = circular_pattern(xyz_shape.shape, n_shapes=16)
    # show_cad(shapes)

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

from typing import Tuple, Type, Union, Dict, List, Any
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib

from bluemira.base.parameter import ParameterFrame
from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.display import plot_2d
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.tools import get_module

from bluemira.builders.shapes import ParameterisedShapeBuilder
from bluemira.magnetostatics.circuits import HelmholtzCage
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.geometry.tools import (
    extrude_shape,
    revolve_shape,
    sweep_shape,
    make_polygon,
    offset_wire,
    circular_pattern,
    boolean_cut,
    signed_distance_2D_polygon,
)


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
        rip_con_tol=1e-3,
        koz_con_tol=1e-3,
        nx=1,
        ny=1,
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
            self.f_constrain_ripple, rip_con_tol * np.ones(len(self.ripple_points[0]))
        )

        # self.optimiser.add_ineq_constraints(
        #     parameterisation.shape_constraints, np.zeros(1)
        # )

        if self.keep_out_zone:
            self.n_koz_points = n_koz_points
            self.koz_points = self._make_koz_points(keep_out_zone)

            self.optimiser.add_ineq_constraints(
                self.f_constrain_koz, koz_con_tol * np.ones(n_koz_points)
            )

        self.nx = nx
        self.ny = ny

    def _make_koz_points(self, keep_out_zone):
        """
        Make a set of points at which to evaluate the KOZ constraint
        """
        return keep_out_zone.discretize(byedges=True, dl=keep_out_zone.length / 200)[
            :, [0, 2]
        ]

    def _make_ripple_points(self, separatrix):
        """
        Make a set of points at which to check the ripple
        """
        points = separatrix.discretize(ndiscr=100).T
        # idx = np.where(points[0] > self.params.R_0.value)[0]
        return points  # [:, idx]

    def _make_single_circuit(self, wire):
        """
        Make a single BioSavart Filament for a single TF coil
        """
        bb = self.wp_cross_section.bounding_box
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

        # We need all arrays to be CCW, this will hopefully go away with a fix for #482
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
        circuit = self._make_single_circuit(wire)

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
        """
        Calculate the signed distances from the parameterised shape to the keep-out zone.
        """
        self.update_cage(x)
        shape = self.parameterisation.create_shape()
        s = shape.discretize(ndiscr=self.n_koz_points)[:, [0, 2]]
        return signed_distance_2D_polygon(s, self.koz_points)

    def f_constrain_koz(self, constraint, x, grad):
        """
        Geometry constraint function to the keep-out-zone
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

    def plot(self, ax=None, **kwargs):
        """
        Plot the optimisation problem.

        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
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

    def build_xz(self, boundary: BluemiraWire, target: str):
        """
        Build the boundary as a wire at the requested target.
        """
        label = target.split("/")[-1]
        return PhysicalComponent(label, BluemiraWire(boundary, label))


class BuildTFWindingPack:
    """
    A class to build TF coil winding pack geometry
    """

    name = "TFWindingPack"

    def __init__(self, wp_centreline, wp_cross_section):
        self.wp_centreline = wp_centreline
        self.wp_cross_section = wp_cross_section

    def build_xy(self):
        # Should normally be gotten with wire_plane_intersect
        x_out = self.wp_centreline.bounding_box.x_max

        return [
            PhysicalComponent(),
            PhysicalComponent(),
        ]

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
        outer_wire = offset_wire(self.wp_cross_section.boundary[0], self.tk_insulation)
        face = BluemiraFace([outer_wire, self.wp_cross_section.boundary[0]])

        # Should normally be gotten with wire_plane_intersect
        x_out = self.wp_centreline.bounding_box.x_max
        outer_face = deepcopy(face)
        outer_face.translate((x_out - outer_face.center_of_mass[0], 0, 0))
        return [
            PhysicalComponent(self.name, face),
            PhysicalComponent(self.name, outer_face),
        ]

    def build_xz(self):
        x_centreline_in = self.wp_centreline.bounding_box.x_min

        x_in_wp = self.wp_cross_section.bounding_box.x_min

        dx_wp = x_centreline_in - x_in_wp

        ins_xs = offset_wire(self.wp_cross_section.boundary[0], self.tk_insulation)
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
        ins_xs = offset_wire(self.wp_cross_section.boundary[0], self.tk_insulation)

        solid = sweep_shape(ins_xs, self.wp_centreline)
        ins_solid = boolean_cut(solid, self.wp_solid)[0]
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
        self.ins_solid = deepcopy(ins_solid)
        self.ins_cross_section = deepcopy(ins_cross_section)
        self.wp_centreline = deepcopy(wp_centreline)
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
        inner_face = BluemiraFace(
            [outer_wire, deepcopy(self.ins_cross_section.boundary[0])]
        )

        bb = self.ins_cross_section.bounding_box
        dx_ins = 0.5 * (bb.x_max - bb.x_min)
        dy_ins = 0.5 * (bb.y_max - bb.y_min)

        # Split the total radial thickness equally on the outboard
        tk_total = self.tk_tf_front_ib + self.tk_tf_nose
        tk = 0.5 * tk_total

        dx_out = dx_ins + tk
        dy_out = dy_ins + self.tk_tf_side
        outer_wire = make_polygon(
            [
                [-dx_out, -dy_out, 0],
                [dx_out, -dy_out, 0],
                [dx_out, dy_out, 0],
                [-dx_out, dy_out, 0],
            ],
            closed=True,
        )

        outer_ins = deepcopy(self.ins_cross_section.boundary[0])

        # Should normally be gotten with wire_plane_intersect
        x_out = self.wp_centreline.bounding_box.x_max
        outer_wire.translate((x_out - outer_wire.center_of_mass[0], 0, 0))
        outer_ins.translate((x_out - outer_ins.center_of_mass[0], 0, 0))
        outer_face = BluemiraFace([outer_wire, outer_ins])
        return [
            PhysicalComponent(self.name, inner_face),
            PhysicalComponent(self.name, outer_face),
        ]

    def build_xz(self):
        pass

    def build_xyz(self):
        inner_xs, outer_xs = self.build_xy()
        inner_xs = inner_xs.shape.boundary[0]
        outer_xs = outer_xs.shape.boundary[0]

        solid = sweep_shape([inner_xs, outer_xs], self.wp_centreline)
        outer_ins_solid = BluemiraSolid(self.ins_solid.boundary[0])
        solid = boolean_cut(solid, outer_ins_solid)[0]

        return PhysicalComponent(self.name, solid)


class BuildTFCoils(Builder):
    """
    A class to build TF coils in the same way as BLUEPRINT.
    """

    _required_config = Builder._required_config
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

    def reinitialise(self):
        pass

    def optimise(self):
        pass

    def build(self):
        pass


if __name__ == "__main__":

    from bluemira.geometry.parameterisations import PrincetonD
    from bluemira.geometry.face import BluemiraFace
    from bluemira.equilibria.shapes import JohnerLCFS
    from bluemira.geometry.tools import circular_pattern
    from bluemira.display import show_cad
    from bluemira.display.displayer import DisplayCADOptions
    from bluemira.base.constants import BLUEMIRA_PALETTE

    params = ParameterFrame(
        [
            ["R_0", "Major radius", 9, "m", None, None, "Input"],
            ["z_0", "Vertical height at major radius", 0, "m", None, "Input", None],
            ["B_0", "Toroidal field at R_0", 6, "T", None, "Input", None],
            ["n_TF", "Number of TF coils", 16, "N/A", None, "Input", None],
            ["TF_ripple_limit", "TF coil ripple limit", 0.6, "%", None, "Input", None],
            [
                "r_tf_in",
                "Inboard radius of the TF coil inboard leg",
                3.2,
                "m",
                None,
                "PROCESS",
            ],
            ["tk_tf_nose", "TF coil inboard nose thickness", 0.6, "m", None, "Input"],
            [
                "tk_tf_front_ib",
                "TF coil inboard steel front plasma-facing",
                0.04,
                "m",
                None,
                "Input",
            ],
            [
                "tk_tf_side",
                "TF coil inboard case minimum side wall thickness",
                0.1,
                "m",
                None,
                "Input",
            ],
            [
                "tk_tf_ins",
                "TF coil ground insulation thickness",
                0.08,
                "m",
                None,
                "Input",
            ],
            # This isn't treated at the moment...
            [
                "tk_tf_insgap",
                "TF coil WP insertion gap",
                0.1,
                "m",
                "Backfilled with epoxy resin (impregnation)",
                "Input",
            ],
            # Dubious WP depth from PROCESS (I used to tweak this when building the TF coils)
            [
                "tf_wp_width",
                "TF coil winding pack radial width",
                0.76,
                "m",
                "Including insulation",
                "PROCESS",
            ],
            [
                "tf_wp_depth",
                "TF coil winding pack depth (in y)",
                1.05,
                "m",
                "Including insulation",
                "PROCESS",
            ],
        ]
    )
    parameterisation = PrincetonD(
        {
            "x1": {"value": params.r_tf_in.value, "fixed": True},
            # We should really be improving the bounds here; lots of ersatz techniques
            # for good estimates come to mind here
            "x2": {"lower_bound": 10, "value": 14, "upper_bound": 18},
            "dz": {"lower_bound": -0.5, "value": 0, "upper_bound": 0.5, "fixed": True},
        }
    )

    # This can be used but you probably need to switch on the parameterisation constraints
    # see #466

    # from bluemira.geometry.parameterisations import TripleArc
    # parameterisation = TripleArc(
    #     {
    #         "x1": {"value": x_tf_wp_center, "fixed": True},
    #         "z1": {"value": 0, "lower_bound": -2, "fixed": True},
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

    # Here we just make a face for the WP cross-section, I used to do this within the
    # TFSystem, because the PROCESS values didn't always match up with what EU-DEMO
    # wanted. Long story. May be fixed now.
    x_c = params.r_tf_in.value
    d_xc = 0.5 * params.tf_wp_width.value
    d_yc = 0.5 * params.tf_wp_depth.value
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

    # Arbitrary: Normally this would come from plasma
    separatrix = JohnerLCFS(
        {
            "r_0": {"value": 9},
            "z_0": {"value": 0},
            "a": {"value": 9 / 3.1},
            "kappa_u": {"value": 1.65},
            "kappa_l": {"value": 1.8},
        }
    ).create_shape()

    # Arbitrary: Normally this would come from somewhere in the Reactor
    koz = offset_wire(separatrix, 2.0, join="arc")

    # Design
    problem = TFWPOptimisationProblem(
        parameterisation, optimiser, params, wp_xs, separatrix, koz
    )
    problem.solve()

    wp_centreline = parameterisation.create_shape()

    # Build
    builder = BuildTFWindingPack(wp_centreline, wp_xs)
    xz_wp_comp = builder.build_xz()
    xyz_wp_shape = builder.build_xyz().shape

    builder = BuildTFInsulation(
        xyz_wp_shape, wp_centreline, wp_xs, params.tk_tf_ins.value
    )
    xz_ins_comp = builder.build_xz()
    xy_ins_shape = builder.build_xy()[0].shape
    xyz_ins_shape = builder.build_xyz().shape

    builder = BuildTFCasing(
        xyz_ins_shape,
        wp_centreline,
        xy_ins_shape,
        params.n_TF.value,
        params.tk_tf_nose.value,
        params.tk_tf_front_ib.value,
        params.tk_tf_side.value,
    )
    xy_casing = builder.build_xy()
    xyz_casing_shape = builder.build_xyz().shape

    xz_comps = [xz_wp_comp]

    xz_comps.extend(xz_ins_comp)

    # Visualise

    f, ax = plt.subplots()
    for shape in xz_comps:
        shape.plot_2d(ax=ax, show=False)

    shapes = [xyz_wp_shape, xyz_ins_shape, xyz_casing_shape]

    # Can't make a plane, can't section, this ensues
    plane = BluemiraFace(
        make_polygon(
            [[-100, 0, -100], [100, 0, -100], [100, 0, 100], [-100, 0, 100]], closed=True
        )
    )
    cut_box = extrude_shape(plane, (0, -10, 0))

    half_shapes = [boolean_cut(shape, cut_box)[0] for shape in shapes]

    shapes = circular_pattern(xyz_casing_shape, n_shapes=params.n_TF.value)[1:9]

    all_shapes = half_shapes + shapes
    options = [
        DisplayCADOptions(color=BLUEMIRA_PALETTE[6]),
        DisplayCADOptions(color=BLUEMIRA_PALETTE[1]),
        DisplayCADOptions(color=BLUEMIRA_PALETTE[0]),
    ]
    options.extend([DisplayCADOptions(color=BLUEMIRA_PALETTE[0])] * len(shapes))

    plasma = revolve_shape(BluemiraFace(separatrix))
    all_shapes.append(plasma)
    options.append(DisplayCADOptions(color=BLUEMIRA_PALETTE[7], transparency=0.5))
    show_cad(all_shapes, options=options)

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
EU-DEMO build classes for TF Coils.
"""
from typing import Type, Optional, List
from copy import deepcopy
from matplotlib.pyplot import plot
import numpy as np

from bluemira.base.builder import Builder, BuildConfig
from bluemira.base.look_and_feel import bluemira_warn, bluemira_debug, bluemira_print
from bluemira.base.parameter import ParameterFrame
from bluemira.base.components import Component, PhysicalComponent
from bluemira.builders.shapes import OptimisedShapeBuilder
from bluemira.display.plotter import PlotOptions, plot_2d
from bluemira.geometry.parameterisations import GeometryParameterisation, PrincetonD
from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.tools import offset_wire, sweep_shape, make_polygon, boolean_cut
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.solid import BluemiraSolid
from bluemira.utilities.optimiser import Optimiser
from bluemira.display.palettes import BLUE_PALETTE


class TFCoilsComponent(Component):
    def __init__(
        self,
        name: str,
        parent: Optional[Component] = None,
        children: Optional[List[Component]] = None,
        magnetics=None,
    ):
        super().__init__(name, parent=parent, children=children)
        self._magnetics = magnetics

    @property
    def magnetics(self):
        return self._magnetics

    @magnetics.setter
    def magnetics(self, magnetics):
        self._magnetics = magnetics


class TFCoilsBuilder(OptimisedShapeBuilder):
    _required_params: List[str] = [
        "R_0",
        "z_0",
        "B_0",
        "n_TF",
        "TF_ripple_limit",
        "r_tf_in",
        "r_tf_in_centre",
        "tk_tf_nose",
        "tk_tf_front_ib",
        "tk_tf_side",
        "tk_tf_ins",
        # This isn't treated at the moment...
        "tk_tf_insgap",
        # Dubious WP depth from PROCESS (I used to tweak this when building the TF coils)
        "tf_wp_width",
        "tf_wp_depth",
    ]
    _required_config = OptimisedShapeBuilder._required_config + []
    _params: ParameterFrame
    _param_class: Type[GeometryParameterisation]
    _default_run_mode: str = "run"
    _design_problem: Optional[GeometryOptimisationProblem] = None
    _centreline: BluemiraWire

    def __init__(self, params, build_config: BuildConfig, **kwargs):
        super().__init__(params, build_config, **kwargs)

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        """
        super().reinitialise(params, **kwargs)

        self._reset_params(params)
        self._wp_cross_section = self._make_wp_xs()

    def _make_wp_xs(self):
        x_c = self.params.r_tf_in_centre.value
        # PROCESS WP thickness includes insulation and insertion gap
        d_xc = 0.5 * (self.params.tf_wp_width - self.params.tk_tf_ins)
        d_yc = 0.5 * (self.params.tf_wp_depth - self.params.tk_tf_ins)
        wp_xs = make_polygon(
            [
                [x_c - d_xc, -d_yc, 0],
                [x_c + d_xc, -d_yc, 0],
                [x_c + d_xc, d_yc, 0],
                [x_c - d_xc, d_yc, 0],
            ],
            closed=True,
        )
        return wp_xs

    def _make_ins_xs(self):
        x_out = self._centreline.bounding_box.x_max
        ins_outer = offset_wire(self._wp_cross_section, self._params.tk_tf_ins.value)
        face = BluemiraFace([ins_outer, self._wp_cross_section])

        outer_face = deepcopy(face)
        outer_face.translate((x_out - outer_face.center_of_mass[0], 0, 0))
        return face, outer_face

    def _make_cas_xs(self):
        x_in = self.params.r_tf_in
        # Insulation included in WP dith
        x_out = (
            x_in
            + self.params.tk_tf_nose
            + self.params.tf_wp_width
            + self.params.tk_tf_front_ib
        )
        half_angle = np.pi / self.params.n_TF.value
        y_in = x_in * np.tan(half_angle)
        y_out = x_out * np.tan(half_angle)
        inboard_wire = make_polygon(
            [[x_in, -y_in, 0], [x_out, -y_out, 0], [x_out, y_out, 0], [x_in, y_in, 0]],
            closed=True,
        )

        dx_ins = 0.5 * self.params.tf_wp_width.value
        dy_ins = 0.5 * self.params.tf_wp_depth.value

        # Split the total radial thickness equally on the outboard
        # This could be done with input params too..
        tk_total = self.params.tk_tf_front_ib.value + self.params.tk_tf_nose.value
        tk = 0.5 * tk_total

        dx_out = dx_ins + tk
        dy_out = dy_ins + self.params.tk_tf_side.value
        outboard_wire = make_polygon(
            [
                [-dx_out, -dy_out, 0],
                [dx_out, -dy_out, 0],
                [dx_out, dy_out, 0],
                [-dx_out, dy_out, 0],
            ],
            closed=True,
        )
        x_out = self._centreline.bounding_box.x_max
        outboard_wire.translate((x_out, 0, 0))
        return inboard_wire, outboard_wire

    def run(self, separatrix, keep_out_zone=None, nx=1, ny=1):
        super().run(
            params=self._params,
            wp_cross_section=self._wp_cross_section,
            separatrix=separatrix,
            keep_out_zone=keep_out_zone,
            nx=nx,
            ny=ny,
        )
        self._centreline = self._design_problem.parameterisation.create_shape()

    def read(self, variables):
        parameterisation = self._param_class(variables)
        self._centreline = parameterisation.create_shape()

    def mock(self, variables):
        parameterisation = self._param_class(variables)
        self._centreline = parameterisation.create_shape()

    def build(self, label: str = "TF Coils", **kwargs) -> Component:
        """
        Build the TF Coils component.

        Returns
        -------
        component: TFCoilsComponent
            The Component built by this builder.
        """
        super().build(**kwargs)

        component = TFCoilsComponent(self.name)

        component.add_child(self.build_xz(component, label=label))
        component.add_child(self.build_xy(component, label=label))
        component.add_child(self.build_xyz(component, label=label))
        return component

    def build_xz(self, component_tree: Component, **kwargs):
        component = Component("xz")

        # Winding pack
        x_min = self._wp_cross_section.bounding_box.x_min
        x_centreline_in = self._centreline.bounding_box.x_min
        dx = abs(x_min - x_centreline_in)
        wp_outer = offset_wire(self._centreline, dx)
        wp_inner = offset_wire(self._centreline, -dx)

        winding_pack = PhysicalComponent(
            "Winding pack", BluemiraFace([wp_outer, wp_inner])
        )
        winding_pack.plot_options.face_options["color"] = BLUE_PALETTE["TF"][1]
        component.add_child(winding_pack)

        # Insulation
        ins_o_outer = offset_wire(wp_outer, self.params.tk_tf_ins.value)
        # plot_2d([wp_outer, ins_o_outer], options=[PlotOptions(wire_options={"color": "r", "linewidth":0.1}), PlotOptions(wire_options={"color": "b", "linewidth":0.1})])
        ins_outer = PhysicalComponent("inner", BluemiraFace([ins_o_outer, wp_outer]))
        ins_outer.plot_options.face_options["color"] = BLUE_PALETTE["TF"][2]
        ins_i_inner = offset_wire(wp_inner, -self.params.tk_tf_ins)
        ins_inner = PhysicalComponent(
            "Insulation", BluemiraFace([wp_inner, ins_i_inner])
        )
        insulation = Component("Insulation", children=[ins_outer, ins_inner])
        component.add_child(insulation)

        # Casing
        # TODO: Either via section of 3-D or some varied thickness offset that we can't
        # really do with primitives

        for child in component.children:  # :'(
            child.plot_options.plane = "xz"
            for sub_child in child.children:
                sub_child.plot_options.plane = "xz"
        component.plot_options.plane = "xz"
        return component

    def build_xy(self, component_tree: Component, **kwargs):
        component = Component("xy")

        # Winding pack
        # Should normally be gotten with wire_plane_intersect
        # (it's not OK to assume that the maximum x value occurs on the midplane)
        x_out = self._centreline.bounding_box.x_max
        xs = BluemiraFace(deepcopy(self._wp_cross_section))
        xs2 = deepcopy(xs)
        xs2.translate((x_out - xs2.center_of_mass[0], 0, 0))

        ib_wp_comp = PhysicalComponent("inboard", xs)
        ib_wp_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][1]
        ob_wp_comp = PhysicalComponent("outboard", xs2)
        ob_wp_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][1]
        winding_pack = Component(
            "Winding pack",
            children=[ib_wp_comp, ob_wp_comp],
        )
        component.add_child(winding_pack)

        # Insulation
        ins_inner_face, ins_outer_face = self._make_ins_xs()

        ib_ins_comp = PhysicalComponent("inboard", ins_inner_face)
        ib_ins_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][2]
        ob_ins_comp = PhysicalComponent("outboard", ins_outer_face)
        ob_ins_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][2]
        insulation = Component(
            "Insulation",
            children=[
                ib_ins_comp,
                ob_ins_comp,
            ],
        )
        component.add_child(insulation)

        # Casing
        ib_cas_wire, ob_cas_wire = self._make_cas_xs()
        cas_inner_face = BluemiraFace(
            [ib_cas_wire, deepcopy(ins_inner_face.boundary[0])]
        )
        cas_outer_face = BluemiraFace(
            [ob_cas_wire, deepcopy(ins_outer_face.boundary[0])]
        )

        ib_cas_comp = PhysicalComponent("inboard", cas_inner_face)
        ib_cas_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][0]
        ob_cas_comp = PhysicalComponent("outboard", cas_outer_face)
        ob_cas_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][0]
        casing = Component(
            "Casing",
            children=[ib_cas_comp, ob_cas_comp],
        )
        component.add_child(casing)

        for child in component.children:  # :'(
            child.plot_options.plane = "xy"
            for sub_child in child.children:
                sub_child.plot_options.plane = "xy"
        component.plot_options.plane = "xy"
        return component

    def build_xyz(self, component_tree: Component, **kwargs):
        component = Component("xyz")

        # Winding pack
        wp_solid = sweep_shape(self._wp_cross_section, self._centreline)
        winding_pack = PhysicalComponent("Winding pack", wp_solid)
        winding_pack.display_cad_options.color = BLUE_PALETTE["TF"][1]
        component.add_child(winding_pack)

        # Insulation
        inner_xs, _ = self._make_ins_xs()
        inner_xs = inner_xs.boundary[0]
        solid = sweep_shape(inner_xs, deepcopy(self._centreline))
        ins_solid = boolean_cut(solid, wp_solid)[0]
        insulation = PhysicalComponent("Insulation", ins_solid)
        insulation.display_cad_options.color = BLUE_PALETTE["TF"][2]
        component.add_child(insulation)

        # Casing
        # Normally I'd do lots more here to get to a proper casing
        # This is just a proof-of-principle
        inner_xs, outer_xs = self._make_cas_xs()
        solid = sweep_shape([inner_xs, outer_xs], self._centreline)
        outer_ins_solid = BluemiraSolid(ins_solid.boundary[0])
        solid = boolean_cut(solid, outer_ins_solid)[0]

        casing = PhysicalComponent("Casing", solid)
        casing.display_cad_options.color = BLUE_PALETTE["TF"][0]
        component.add_child(casing)
        return component


def break_test():
    p = PrincetonD(
        {
            "x1": {"value": 3.649},
            "x2": {"value": 15.933007419714876},
            "dz": {"value": 0.0},
        }
    )
    wire = p.create_shape()
    o1 = offset_wire(wire, 0.3399999999620089)
    o2 = offset_wire(o1, 0.08)

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
import numpy as np
from numpy.core.fromnumeric import shape

from bluemira.base.look_and_feel import bluemira_warn, bluemira_debug, bluemira_print
from bluemira.base.parameter import ParameterFrame
from bluemira.base.components import Component, PhysicalComponent
from bluemira.builders.shapes import OptimisedShapeBuilder
from bluemira.display.plotter import PlotOptions, plot_2d
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.tools import (
    boolean_fuse,
    extrude_shape,
    offset_wire,
    sweep_shape,
    make_polygon,
    boolean_cut,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.solid import BluemiraSolid
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.magnetostatics.circuits import (
    ArbitraryPlanarRectangularXSCircuit,
    HelmholtzCage,
)


class TFCoilsComponent(Component):
    def __init__(
        self,
        name: str,
        parent: Optional[Component] = None,
        children: Optional[List[Component]] = None,
        field_solver=None,
    ):
        super().__init__(name, parent=parent, children=children)
        self._field_solver = field_solver

    def field(self, x, y, z):
        """
        Calculate the magnetic field due to the TF coils at a set of points.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinate(s) of the points at which to calculate the field
        y: Union[float, np.array]
            The y coordinate(s) of the points at which to calculate the field
        z: Union[float, np.array]
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        field: np.array
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        return self._field_solver.field(x, y, z)


class TFCoilsBuilder(OptimisedShapeBuilder):
    """
    Builder for the TF Coils.
    """

    _required_params: List[str] = [
        "R_0",
        "z_0",
        "B_0",
        "n_TF",
        "TF_ripple_limit",
        "r_tf_in",
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

    def _derive_shape_params(self):
        shape_params = super()._derive_shape_params()
        # PROCESS doesn't output the radius of the current centroid on the inboard
        r_current_in_board = (
            self.params.r_tf_in
            + self.params.tk_tf_nose
            + self.params.tk_tf_ins
            + 0.5 * (self.params.tf_wp_width - 2 * self.params.tk_tf_ins)
        )
        self._params.add_parameter(
            "r_tf_current_ib",
            "Radius of the TF coil current centroid on the inboard",
            r_current_in_board,
            "m",
            source="bluemira",
        )
        shape_params["x1"] = {"value": r_current_in_board, "fixed": True}
        return shape_params

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
        self._centreline = None
        self._wp_cross_section = self._make_wp_xs()

    def run(self, separatrix, keep_out_zone=None, nx=1, ny=1):
        """
        Run the specified design optimisation problem to generate the TF coil winding
        pack current centreline.
        """
        super().run(
            params=self._params,
            wp_cross_section=self._wp_cross_section,
            separatrix=separatrix,
            keep_out_zone=keep_out_zone,
            nx=nx,
            ny=ny,
        )
        self._centreline = self._design_problem.parameterisation.create_shape()

    def read(self, variables: dict):
        """
        Read in a variable dictionary to set up a specified GeometryParameterisation.
        """
        parameterisation = self._param_class(variables)
        self._centreline = parameterisation.create_shape()

    def mock(self, centreline):
        """
        Mock a design of TF coils using a specified current centreline.
        """
        self._centreline = centreline

    def build(self, label: str = "TF Coils", **kwargs) -> Component:
        """
        Build the TF Coils component.

        Returns
        -------
        component: TFCoilsComponent
            The Component built by this builder.
        """
        super().build(**kwargs)

        field_solver = self._make_field_solver()
        component = TFCoilsComponent(self.name, field_solver=field_solver)

        # component.add_child(self.build_xz())
        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())
        return component

    def build_xz(self):
        """
        Build the x-z components of the TF coils.
        """
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

    def build_xy(self):
        """
        Build the x-y components of the TF coils.
        """
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

    def build_xyz(self):
        """
        Build the x-y-z components of the TF coils.
        """
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
        # Make inner xs into a rectangle
        bb = inner_xs.bounding_box
        x_min = bb.x_min
        x_max = bb.x_max

        half_angle = np.pi / self.params.n_TF.value
        y_in = self.params.r_tf_in * np.sin(half_angle)
        inner_xs_rect = make_polygon(
            [[x_min, -y_in, 0], [x_max, -y_in, 0], [x_max, y_in, 0], [x_min, y_in, 0]],
            closed=True,
        )

        # Sweep with a varying rectangular cross-section
        centreline_points = self._centreline.discretize(byedges=True, ndiscr=2000).T
        idx = np.where(np.isclose(centreline_points[0], np.min(centreline_points[0])))[0]
        z_turn_top = np.max(centreline_points[2][idx])
        z_turn_bot = np.min(centreline_points[2][idx])

        inner_xs_rect_top = deepcopy(inner_xs_rect)
        inner_xs_rect_top.translate((0, 0, z_turn_top))
        inner_xs_rect_bot = deepcopy(inner_xs_rect)
        inner_xs_rect_bot.translate((0, 0, z_turn_bot))
        solid = sweep_shape(
            [inner_xs_rect_top, outer_xs, inner_xs_rect_bot], self._centreline
        )

        # Christ, need offset or bounding_box or section_shape, can't trust any atm.
        # The bounding box of the solid is much bigger than I'd expect
        bb = solid.bounding_box
        z_min = bb.z_min
        z_max = bb.z_max

        inner_xs.translate((0, 0, z_min - inner_xs.center_of_mass[2]))
        inboard_casing = extrude_shape(BluemiraFace(inner_xs), (0, 0, z_max - z_min))

        z_min_cl = np.min(centreline_points[2])
        z_max_cl = np.max(centreline_points[2])
        # Join the straight leg to the curvy bits
        bb = inboard_casing.bounding_box
        x_min = bb.x_min
        idx = np.where(np.isclose(centreline_points[2], z_max_cl))[0]
        x_turn_top = np.min(centreline_points[0][idx])
        idx = np.where(np.isclose(centreline_points[2], z_min_cl))[0]
        x_turn_bot = np.min(centreline_points[0][idx])
        joiner_top = make_polygon(
            [
                [x_min, -y_in, z_max],
                [x_turn_top, -y_in, z_max],
                [x_turn_top, y_in, z_max],
                [x_min, y_in, z_max],
            ],
            closed=True,
        )
        joiner_top = extrude_shape(BluemiraFace(joiner_top), (0, 0, -z_max))
        joiner_bot = make_polygon(
            [
                [x_min, -y_in, z_min],
                [x_turn_bot, -y_in, z_min],
                [x_turn_bot, y_in, z_min],
                [x_min, y_in, z_min],
            ],
            closed=True,
        )
        joiner_bot = extrude_shape(BluemiraFace(joiner_bot), (0, 0, -z_min))

        solid = boolean_fuse([solid, inboard_casing, joiner_top, joiner_bot])

        # Need to cut away the excess, but I need section_shape or something
        # solid = boolean_cut(solid, cutter)[0]

        outer_ins_solid = BluemiraSolid(ins_solid.boundary[0])
        solid = boolean_cut(solid, outer_ins_solid)[0]

        casing = PhysicalComponent("Casing", solid)
        casing.display_cad_options.color = BLUE_PALETTE["TF"][0]
        component.add_child(casing)
        return component

    def _make_field_solver(self):
        circuit = ArbitraryPlanarRectangularXSCircuit(
            self._centreline.discretize(byedges=True, ndiscr=100),
            breadth=0.5 * self.params.tf_wp_width - self.params.tk_tf_ins,
            depth=0.5 * self.params.tf_wp_depth - self.params.tk_tf_ins,
            current=1,
        )
        cage = HelmholtzCage(circuit, self.params.n_TF.value)
        field = cage.field(self.params.R_0, 0, self.params.z_0)
        current = -self.params.B_0 / field[1]  # single coil amp-turns
        # Ooops..
        circuit = ArbitraryPlanarRectangularXSCircuit(
            self._centreline.discretize(byedges=True, ndiscr=100),
            breadth=0.5 * self.params.tf_wp_width - self.params.tk_tf_ins,
            depth=0.5 * self.params.tf_wp_depth - self.params.tk_tf_ins,
            current=current,
        )
        cage = HelmholtzCage(circuit, self.params.n_TF.value)
        return cage

    def _make_wp_xs(self):
        """
        Make the winding pack x-y cross-section wire
        """
        x_c = self.params.r_tf_current_ib
        # PROCESS WP thickness includes insulation and insertion gap
        d_xc = 0.5 * (self.params.tf_wp_width - 2 * self.params.tk_tf_ins)
        d_yc = 0.5 * (self.params.tf_wp_depth - 2 * self.params.tk_tf_ins)
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
        """
        Make the insulation x-y cross-section faces
        """
        x_out = self._centreline.bounding_box.x_max
        ins_outer = offset_wire(self._wp_cross_section, self._params.tk_tf_ins.value)
        face = BluemiraFace([ins_outer, self._wp_cross_section])

        outer_face = deepcopy(face)
        outer_face.translate((x_out - outer_face.center_of_mass[0], 0, 0))
        return face, outer_face

    def _make_cas_xs(self):
        """
        Make the casing x-y cross-section wires
        """
        x_in = self.params.r_tf_in
        # Insulation included in WP width
        x_out = (
            x_in
            + self.params.tk_tf_nose
            + self.params.tf_wp_width
            + self.params.tk_tf_front_ib
        )
        half_angle = np.pi / self.params.n_TF.value
        y_in = x_in * np.sin(half_angle)
        y_out = x_out * np.sin(half_angle)
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

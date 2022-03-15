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
import os
from copy import deepcopy
from typing import List, Optional, Type

import numpy as np

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_print
from bluemira.builders.EUDEMO.tools import circular_pattern_component
from bluemira.builders.shapes import OptimisedShapeBuilder
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    extrude_shape,
    make_polygon,
    offset_wire,
    slice_shape,
    sweep_shape,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.magnetostatics.circuits import (
    ArbitraryPlanarRectangularXSCircuit,
    HelmholtzCage,
)


class TFCoilsComponent(Component):
    """
    Toroidal field coils component, with a solver for the magnetic field from all of the
    coils.

    Parameters
    ----------
    name: str
        Name of the component
    parent: Optional[Component] = None
        Parent component
    children: Optional[List[Component]] = None
        List of child components
    field_solver: Optional[CurrentSource]
        Magnetic field solver
    """

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
    _params: Configuration
    _param_class: Type[GeometryParameterisation]
    _default_runmode: str = "run"
    _design_problem: Optional[GeometryOptimisationProblem] = None
    _centreline: BluemiraWire
    _geom_path: Optional[str] = None
    _keep_out_zone: Optional[BluemiraWire] = None
    _separatrix: Optional[BluemiraWire] = None

    def __init__(
        self,
        params,
        build_config: BuildConfig,
        separatrix: Optional[BluemiraWire] = None,
        keep_out_zone: Optional[BluemiraWire] = None,
    ):
        super().__init__(
            params, build_config, separatrix=separatrix, keep_out_zone=keep_out_zone
        )

    @property
    def geom_path(self) -> str:
        """
        The path at which the geometry parameterisation can be written to or read from.
        """
        return self._geom_path

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        self._geom_path = build_config.get("geom_path", None)
        has_geom_path = self._geom_path is not None
        valid_geom_path = has_geom_path and os.path.exists(self._geom_path)
        if self._runmode.name.lower() == "read" and not valid_geom_path:
            raise BuilderError(
                "Must supply a geom_path that at either points to the directory "
                "containing the geometry parameterisation, or points to the geometry "
                "parameterisation file, in build_config when using 'read' mode."
            )

    def _derive_shape_params(self):
        shape_params = super()._derive_shape_params()
        # PROCESS doesn't output the radius of the current centroid on the inboard
        r_current_in_board = (
            self._params.r_tf_in
            + self._params.tk_tf_nose
            + self._params.tk_tf_ins
            + 0.5 * (self._params.tf_wp_width - 2 * self._params.tk_tf_ins)
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

    def reinitialise(
        self,
        params,
        separatrix: Optional[BluemiraWire] = None,
        keep_out_zone: Optional[BluemiraWire] = None,
    ) -> None:
        """
        Initialise the state of this builder ready for a new run.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        separatrix: Optional[BluemiraWire]
            The separatrix to pass into constrained optimisation routines. Must be
            provided if this Builder's runmode is set to run. By default, None.
        keep_out_zone: Optional[BluemiraWire]
            Exclusion zone, if any to apply to the build. By default None.

        Raises
        ------
        BuilderError
            If the runmode is set to run but a separatrix is not provided.
        """
        super().reinitialise(params)

        if self.runmode == "run" and separatrix is None:
            raise BuilderError(
                "A separatrix must be provided as the runmode for this builder is set "
                "to run"
            )

        self._centreline = None
        self._wp_cross_section = self._make_wp_xs()
        self._separatrix = separatrix
        self._keep_out_zone = keep_out_zone

        if self._geom_path is not None and os.path.isdir(self._geom_path):
            default_file_name = (
                f"tf_coils_{self._param_class.__name__}_{self._params.n_TF.value}.json"
            )
            self._geom_path = os.sep.join([self._geom_path, default_file_name])

    def run(self):
        """
        Run the specified design optimisation problem to generate the TF coil winding
        pack current centreline.
        """
        super().run(
            params=self._params,
            wp_cross_section=self._wp_cross_section,
            separatrix=self._separatrix,
            keep_out_zone=self._keep_out_zone,
        )
        self._centreline = self._design_problem.parameterisation.create_shape()

    def read(self):
        """
        Read in a file to set up a specified GeometryParameterisation and extract the
        current centreline.
        """
        bluemira_print(f"Reading TF Coil centreline shape from file {self._geom_path}")

        with open(self._geom_path, "r") as fh:
            self._shape = self._param_class.from_json(fh)
        self._centreline = self._shape.create_shape()

    def mock(self):
        """
        Mock a design of TF coils using the original parameterisation of the current
        centreline.
        """
        bluemira_print(
            "Mocking TF Coil centreline shape from parameterisation "
            f"{self._shape.variables}"
        )

        self._centreline = self._shape.create_shape()

    def build(self) -> TFCoilsComponent:
        """
        Build the TF Coils component.

        Returns
        -------
        component: TFCoilsComponent
            The Component built by this builder.
        """
        super().build()

        field_solver = self._make_field_solver()
        component = TFCoilsComponent(self.name, field_solver=field_solver)

        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())
        component.add_child(self.build_xz())
        return component

    def build_xz(self) -> Component:
        """
        Build the x-z components of the TF coils.
        """
        component = Component("xz")

        # Winding pack
        x_min = self._wp_cross_section.bounding_box.x_min
        x_centreline_in = self._centreline.bounding_box.x_min
        dx = abs(x_min - x_centreline_in)
        wp_outer = offset_wire(self._centreline, dx, join="arc")
        wp_inner = offset_wire(self._centreline, -dx, join="arc")

        winding_pack = PhysicalComponent(
            "Winding pack", BluemiraFace([wp_outer, wp_inner])
        )
        winding_pack.plot_options.face_options["color"] = BLUE_PALETTE["TF"][1]
        component.add_child(winding_pack)

        # Insulation
        ins_o_outer = offset_wire(wp_outer, self.params.tk_tf_ins.value, join="arc")
        ins_outer = PhysicalComponent("inner", BluemiraFace([ins_o_outer, wp_outer]))
        ins_outer.plot_options.face_options["color"] = BLUE_PALETTE["TF"][2]
        ins_i_inner = offset_wire(wp_inner, -self.params.tk_tf_ins, join="arc")
        ins_inner = PhysicalComponent(
            "Insulation", BluemiraFace([wp_inner, ins_i_inner])
        )
        ins_inner.plot_options.face_options["color"] = BLUE_PALETTE["TF"][2]
        insulation = Component("Insulation", children=[ins_outer, ins_inner])
        component.add_child(insulation)

        # Casing
        cas_inner, cas_outer = self._temp_casing
        cas_inner = PhysicalComponent("inner", cas_inner)
        cas_inner.plot_options.face_options["color"] = BLUE_PALETTE["TF"][0]
        cas_outer = PhysicalComponent("outer", cas_outer)
        cas_outer.plot_options.face_options["color"] = BLUE_PALETTE["TF"][0]
        casing = Component("Casing", children=[cas_inner, cas_outer])
        component.add_child(casing)

        bm_plot_tools.set_component_view(component, "xz")

        return component

    def build_xy(self) -> Component:
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
        sectors = circular_pattern_component(winding_pack, self._params.n_TF.value)
        component.add_children(sectors, merge_trees=True)

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
        sectors = circular_pattern_component(insulation, self._params.n_TF.value)
        component.add_children(sectors, merge_trees=True)

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
        sectors = circular_pattern_component(casing, self._params.n_TF.value)
        component.add_children(sectors, merge_trees=True)

        bm_plot_tools.set_component_view(component, "xy")

        return component

    def build_xyz(self, degree: float = 360.0) -> Component:
        """
        Build the x-y-z components of the TF coils.

        Parameters
        ----------
        degree: float
            The angle [°] around which to build the components, by default 360.0.

        Returns
        -------
        component: Component
            The component grouping the results in 3D (xyz).
        """
        component = Component("xyz")

        # Minimum angle per TF coil (nudged by a tiny length since we start counting a
        # sector at theta=0). This means we can draw a sector as 360 / n_TF and get one
        # TF coil per sector. Python represents floats with 16 significant figures before
        # getting round off, so adding on 1e-13 works here, in case someone sets n_TF
        # to be 2.
        min_tf_deg = (360.0 / self._params.n_TF.value) + 1e-13
        n_tf_draw = min(int(degree // min_tf_deg) + 1, self._params.n_TF.value)
        degree = (360.0 / self._params.n_TF.value) * n_tf_draw

        # Winding pack
        wp_solid = sweep_shape(self._wp_cross_section, self._centreline)
        winding_pack = PhysicalComponent("Winding pack", wp_solid)
        winding_pack.display_cad_options.color = BLUE_PALETTE["TF"][1]
        sectors = circular_pattern_component(winding_pack, n_tf_draw, degree=degree)
        component.add_children(sectors, merge_trees=True)

        # Insulation
        inner_xs, _ = self._make_ins_xs()
        inner_xs = inner_xs.boundary[0]
        solid = sweep_shape(inner_xs, deepcopy(self._centreline))
        ins_solid = boolean_cut(solid, wp_solid)[0]
        insulation = PhysicalComponent("Insulation", ins_solid)
        insulation.display_cad_options.color = BLUE_PALETTE["TF"][2]
        sectors = circular_pattern_component(insulation, n_tf_draw, degree=degree)
        component.add_children(sectors, merge_trees=True)

        # Casing
        # Normally I'd do lots more here to get to a proper casing
        # This is just a proof-of-principle
        inner_xs, outer_xs = self._make_cas_xs()
        # Make inner xs into a rectangle
        bb = inner_xs.bounding_box
        x_min = bb.x_min
        x_max = bb.x_max

        half_angle = np.pi / self.params.n_TF.value
        y_in = self.params.r_tf_in * np.tan(half_angle)
        inner_xs_rect = make_polygon(
            [[x_min, x_max, x_max, x_min], [-y_in, -y_in, y_in, y_in], [0, 0, 0, 0]],
            closed=True,
        )

        # Sweep with a varying rectangular cross-section
        centreline_points = self._centreline.discretize(byedges=True, ndiscr=2000)
        idx = np.where(np.isclose(centreline_points.x, np.min(centreline_points.x)))[0]
        z_turn_top = np.max(centreline_points.z[idx])
        z_turn_bot = np.min(centreline_points.z[idx])
        z_min_cl = np.min(centreline_points.z)
        z_max_cl = np.max(centreline_points.z)

        inner_xs_rect_top = deepcopy(inner_xs_rect)
        inner_xs_rect_top.translate((0, 0, z_turn_top))
        inner_xs_rect_bot = deepcopy(inner_xs_rect)
        inner_xs_rect_bot.translate((0, 0, z_turn_bot))
        solid = sweep_shape(
            [inner_xs_rect_top, outer_xs, inner_xs_rect_bot], self._centreline
        )

        # This is because the bounding box of a solid is not to be trusted
        cut_wires = slice_shape(
            solid, BluemiraPlacement.from_3_points([0, 0, 0], [1, 0, 0], [1, 0, 1])
        )
        cut_wires.sort(key=lambda wire: wire.length)
        boundary = cut_wires[-1]
        bb = boundary.bounding_box
        z_min = bb.z_min
        z_max = bb.z_max
        y_in = 0.5 * (
            self.params.tf_wp_depth + self.params.tk_tf_ins + self.params.tk_tf_side
        )

        inner_xs.translate((0, 0, z_min - inner_xs.center_of_mass[2]))
        inboard_casing = extrude_shape(BluemiraFace(inner_xs), (0, 0, z_max - z_min))

        # Join the straight leg to the curvy bits
        x_min = np.min(centreline_points.x)
        idx = np.where(np.isclose(centreline_points.z, z_max_cl))[0]
        x_turn_top = np.min(centreline_points.x[idx])
        idx = np.where(np.isclose(centreline_points.z, z_min_cl))[0]
        x_turn_bot = np.min(centreline_points.x[idx])
        joiner_top = make_polygon(
            [
                [x_min, x_turn_top, x_turn_top, x_min],
                [-y_in, -y_in, y_in, y_in],
                [z_max, z_max, z_max, z_max],
            ],
            closed=True,
        )
        joiner_top = extrude_shape(BluemiraFace(joiner_top), (0, 0, -z_max))
        joiner_bot = make_polygon(
            [
                [x_min, x_turn_bot, x_turn_bot, x_min],
                [-y_in, -y_in, y_in, y_in],
                [z_min, z_min, z_min, z_min],
            ],
            closed=True,
        )
        joiner_bot = extrude_shape(BluemiraFace(joiner_bot), (0, 0, -z_min))

        # Need to cut away the excess joiner extrusions
        cl = deepcopy(self._centreline)
        cl.translate((0, -2 * self.params.tf_wp_depth, 0))
        cl_face = BluemiraFace(cl)
        cutter = extrude_shape(cl_face, (0, 4 * self.params.tf_wp_depth, 0))
        joiner_top = boolean_cut(joiner_top, cutter)[0]
        joiner_bot = boolean_cut(joiner_bot, cutter)[0]

        # Cut away straight sweep before fusing to protect against degenerate faces
        # Keep the largest piece
        pieces = boolean_cut(solid, inboard_casing)
        pieces.sort(key=lambda x: x.volume)
        solid = pieces[-1]

        case_solid = boolean_fuse([solid, inboard_casing, joiner_top, joiner_bot])
        outer_ins_solid = BluemiraSolid(ins_solid.boundary[0])
        case_solid_hollow = boolean_cut(case_solid, outer_ins_solid)[0]
        self._make_cas_xz(case_solid_hollow)

        casing = PhysicalComponent("Casing", case_solid_hollow)
        casing.display_cad_options.color = BLUE_PALETTE["TF"][0]
        sectors = circular_pattern_component(casing, n_tf_draw, degree=degree)
        component.add_children(sectors, merge_trees=True)

        return component

    def _make_field_solver(self):
        """
        Make a magnetostatics solver for the field from the TF coils.
        """
        circuit = ArbitraryPlanarRectangularXSCircuit(
            self._centreline.discretize(byedges=True, ndiscr=100),
            breadth=0.5 * self.params.tf_wp_width - self.params.tk_tf_ins,
            depth=0.5 * self.params.tf_wp_depth - self.params.tk_tf_ins,
            current=1,
        )
        solver = HelmholtzCage(circuit, self.params.n_TF.value)
        field = solver.field(self.params.R_0, 0, self.params.z_0)
        current = -self.params.B_0 / field[1]  # single coil amp-turns
        solver.set_current(current)
        return solver

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
                [x_c - d_xc, x_c + d_xc, x_c + d_xc, x_c - d_xc],
                [-d_yc, -d_yc, d_yc, d_yc],
                [0, 0, 0, 0],
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
        tan_half_angle = np.tan(np.pi / self.params.n_TF.value)
        y_in = x_in * tan_half_angle
        y_out = x_out * tan_half_angle
        inboard_wire = make_polygon(
            [
                [x_in, x_out, x_out, x_in],
                [-y_in, -y_out, y_out, y_in],
                [0, 0, 0, 0],
            ],
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
                [-dx_out, dx_out, dx_out, -dx_out],
                [-dy_out, -dy_out, dy_out, dy_out],
                [0, 0, 0, 0],
            ],
            closed=True,
        )
        x_out = self._centreline.bounding_box.x_max
        outboard_wire.translate((x_out, 0, 0))
        return inboard_wire, outboard_wire

    def _make_cas_xz(self, solid):
        """
        Make the casing x-z cross-section from a 3-D volume.
        """
        wires = slice_shape(
            solid, BluemiraPlacement.from_3_points([0, 0, 0], [1, 0, 0], [1, 0, 1])
        )
        wires.sort(key=lambda wire: wire.length)
        if len(wires) != 4:
            raise BuilderError(
                "Unexpected TF coil x-z cross-section. It is likely that a previous "
                "boolean cutting operation failed to create a hollow solid."
            )

        inner = BluemiraFace([wires[1], wires[0]])
        outer = BluemiraFace([wires[3], wires[2]])
        self._temp_casing = [inner, outer]

    def save_shape(self, filename: str = None, **kwargs):
        """
        Save the shape to a json file.

        Parameters
        ----------
        filename: str
            The path to the file that the shape should be written to. By default this
            will be the geom_path.
        """
        if filename is None:
            filename = self._geom_path
        super().save_shape(filename, **kwargs)

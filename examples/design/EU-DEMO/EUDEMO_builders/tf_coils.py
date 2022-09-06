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
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Union

import numpy as np

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.constants import EPS
from bluemira.base.designer import Designer
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.builders.tools import circular_pattern_component, get_n_sectors
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.plane import BluemiraPlane
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
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.tools import get_class_from_module


class TFCoil(ComponentManager):
    """
    Wrapper around the TF Coil component tree.
    """

    def __init__(self, component, field_solver):
        super().__init__(component)
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


@dataclass
class TFCoilDesignerParams(ParameterFrame):
    """
    TF Coil builder parameters
    """

    r_tf_current_ib: Parameter[float]
    tk_tf_wp: Parameter[float]
    tk_tf_wp_y: Parameter[float]
    tf_wp_width: Parameter[float]
    tk_tf_front_ib: Parameter[float]
    g_ts_tf: Parameter[float]


@dataclass
class TFCoilBuilderParams(ParameterFrame):
    """
    TF Coil builder parameters
    """

    R_O: Parameter[float]
    z_0: Parameter[float]
    B_0: Parameter[float]
    n_TF: Parameter[int]
    r_tf_in: Parameter[float]
    tf_wp_depth: Parameter[float]
    tf_wp_width: Parameter[float]
    tk_tf_front_ib: Parameter[float]
    tk_tf_ins: Parameter[float]
    tk_tf_ins_gap: Parameter[float]
    tk_tf_nose: Parameter[float]
    tk_tf_side: Parameter[float]
    tk_tf_wp: Parameter[float]
    tk_tf_wp_y: Parameter[float]


class TFCoilDesigner(Designer[GeometryParameterisation]):

    param_cls = TFCoilDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        separatrix: BluemiraWire,
        keep_out_zone: BluemiraWire,
    ):
        super().__init__(params, build_config)

        self.parameterisation_cls: Type[
            GeometryParameterisation
        ] = get_class_from_module(
            self.build_config["param_class"],
            default_module="bluemira.geometry.parameterisations",
        )

        self.variables_map = self.build_config.get("variables_map", {})

        self.file_path = self.build_config.get("file_path", None)

        problem_class = self.build_config.get("problem_class", None)

        if problem_class is not None:
            self.problem_class = get_class_from_module(problem_class)
            self.problem_settings = self.build_config.get("problem_settings", {})

            self.opt_config = self.build_config.get("optimisation_settings", {})

            self.algorithm_name = self.opt_config.get("algorithm_name", "SLSQP")
            self.opt_conditions = self.opt_config.get("conditions", {"max_eval": 100})
            self.opt_parameters = self.opt_config.get("parameters", {})

    def _make_wp_xs(self):
        """
        Make the winding pack x-y cross-section wire (excluding insulation and
        insertion gap)
        """
        d_xc = 0.5 * self.params.tk_tf_wp.value
        d_yc = np.full(4, 0.5 * self.params.tk_tf_wp_y.value)
        d_yc[:2] = -d_yc[:2]

        x_c = np.full(4, self.params.r_tf_current_ib)
        x_c[[0, -1]] -= d_xc
        x_c[[1, 2]] += d_xc

        wp_xs = make_polygon([x_c, d_yc, np.zeros(4)], closed=True)
        return wp_xs

    def _make_centreline_koz(self, keep_out_zone):
        """
        Make a keep-out-zone for the TF coil centreline optimisation problem.
        """
        # The keep-out zone is for the TF WP centreline, so we need to add to it to
        # prevent clashes when the winding pack thickness and casing are added.
        tk_offset = 0.5 * self._params.tf_wp_width.value
        # Variable thickness of the casing is problematic...
        # TODO: Improve this estimate (or use variable offset here too..)
        tk_offset += 1.5 * self._params.tk_tf_front_ib
        tk_offset += self._params.g_ts_tf.value
        return offset_wire(keep_out_zone, tk_offset, open_wire=False, join="arc")

    def _derive_shape_params(self, variables_map: Dict[str, str]) -> Dict:
        shape_params = {}
        for key, val in variables_map.items():
            if isinstance(val, str):
                val = getattr(self.params, val).value

            if isinstance(val, dict):
                if isinstance(val["value"], str):
                    val["value"] = getattr(self.params, val["value"]).value
            else:
                val = {"value": val}

            shape_params[key] = val
        # Radial width of the winding pack with no insulation or insertion gap
        dr_wp = (
            self._params.tf_wp_width.value
            - 2 * self._params.tk_tf_ins.value
            - 2 * self._params.tk_tf_insgap.value
        )
        self.params.add_parameter(
            "tk_tf_wp", value=dr_wp, unit="m", source=type(self).__name__
        )

        # Toroidal width of the winding pack no insulation
        dy_wp = (
            self._params.tf_wp_depth.value
            - 2 * self._params.tk_tf_ins
            - 2 * self._params.tk_tf_insgap
        )
        self.params.add_parameter(
            "tk_tf_wp_y", value=dy_wp, unit="m", source=type(self).__name__
        )

        # PROCESS doesn't output the radius of the current centroid on the inboard
        r_current_in_board = (
            self._params.r_tf_in.value
            + self._params.tk_tf_nose.value
            + self._params.tk_tf_ins.value
            + self._params.tk_tf_insgap.value
            + 0.5 * dr_wp
        )
        self._params.add_parameter(
            "r_tf_current_ib",
            "Radius of the TF coil current centroid on the inboard",
            r_current_in_board,
            "m",
            source=type(self).__name__,
        )
        shape_params["x1"] = {"value": r_current_in_board, "fixed": True}
        return shape_params

    def _get_parameterisation(self):
        return self.parameterisation_cls(self._derive_shape_params(self.variables_map))

    def run(self) -> GeometryParameterisation:
        """
        Run the specified design optimisation problem to generate the TF coil winding
        pack current centreline.
        """
        parameterisation = self._get_parameterisation()

        if not hasattr(self, "problem_class"):
            raise ValueError(
                f"Cannot execute {type(self).__name__} in RUN mode: no problem_class specified."
            )

        bluemira_debug(
            f"Setting up design problem with:\n"
            f"algorithm_name: {self.algorithm_name}\n"
            f"n_variables: {parameterisation.variables.n_free_variables}\n"
            f"opt_conditions: {self.opt_conditions}\n"
            f"opt_parameters: {self.opt_parameters}"
        )
        optimiser = Optimiser(
            self.algorithm_name,
            parameterisation.variables.n_free_variables,
            self.opt_conditions,
            self.opt_parameters,
        )

        if self._problem_settings != {}:
            bluemira_debug(
                f"Applying non-default settings to problem: {self.problem_settings}"
            )
        design_problem = self._problem_class(
            parameterisation,
            optimiser,
            wp_cross_section=self._make_wp_xs(),
            separatrix=self.separatrix,
            keep_out_zone=self._make_centreline_koz(self.keep_out_zone),
            **self.problem_settings,
        )

        bluemira_print(f"Solving design problem: {type(design_problem).__name__}")
        if parameterisation.n_ineq_constraints > 0:
            bluemira_debug("Applying shape constraints")
            design_problem.apply_shape_constraints()

        bluemira_debug("Solving...")
        return design_problem.optimise()

    def read(self) -> GeometryParameterisation:
        """
        Read in a file to set up a specified GeometryParameterisation and extract the
        current centreline.
        """
        if not self.file_path:
            raise ValueError(
                f"Cannot execute {type(self).__name__} in 'read' mode: no file path specified."
            )
        return self.parameterisation_cls.from_json(file=self.file_path)

    def mock(self) -> GeometryParameterisation:
        """
        Mock a design of TF coils using the original parameterisation of the current
        centreline.
        """
        return self._get_parameterisation()


class TFCoilBuilder(Builder):

    WP = "Winding Pack"
    OUT = "outer"
    IN = "inner"
    CASING = "Casing"
    INS = "Insulation"
    INB = "inboard"
    OUTB = "outboard"
    param_cls: Type[TFCoilBuilderParams] = TFCoilBuilderParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
        centreline: BluemiraWire,
        wp_cross_section: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.centreline = centreline

        self.wp_cross_section = wp_cross_section

    def build(self) -> TFCoil:
        """
        Build the vacuum vessel component.
        """
        ins_inner_face, ins_outer_face = self._make_ins_xsec()
        xyz = self.build_xyz(ins_inner_face)
        return TFCoil(
            self.component_tree(
                xz=self.build_xz(xyz[0].get_component_properties("shape")),
                xy=self.build_xy(ins_inner_face, ins_outer_face),
                xyz=xyz,
            ),
            self._make_field_solver(),
        )

    def build_xz(self) -> List[PhysicalComponent, Component, Component]:
        """
        Build the x-z components of the TF coils.
        """
        wp_inner, wp_outer, winding_pack = self._build_xz_wp()

        return [
            winding_pack,
            self._build_xz_ins(wp_inner, wp_outer),
            self._build_xz_case(),
        ]

    def build_xy(
        self, ins_inner_face: BluemiraFace, ins_outer_face: BluemiraFace
    ) -> List[Component, ...]:
        """
        Build the x-y components of the TF coils.
        """
        return [
            *self._build_xy_wp(),
            *self._build_xy_ins(ins_inner_face, ins_outer_face),
            *self._build_xy_case(ins_inner_face, ins_outer_face),
        ]

    def build_xyz(
        self, ins_inner_face: BluemiraFace, degree: float = 360.0
    ) -> List[Component]:
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
        # Minimum angle per TF coil (nudged by a tiny length since we start counting a
        # sector at theta=0). This means we can draw a sector as 360 / n_TF and get one
        # TF coil per sector. Python represents floats with 16 significant figures before
        # getting round off, so adding on 1e3*EPS works here, in case someone sets n_TF
        # to be 2.
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)
        n_sectors = min(
            n_sectors, get_n_sectors(self.params.n_TF.value + 1e3 * EPS, degree)[1] + 1
        )

        wp_solid, wp_sectors = self._build_xyz_wp(n_sectors, sector_degree)

        ins_solid, ins_sectors = self._build_xyz_ins(
            wp_solid, ins_inner_face, n_sectors, sector_degree
        )

        case_sectors = self._build_xyz_case(ins_solid, n_sectors, sector_degree)

        return [*wp_sectors, *ins_sectors, *case_sectors]

    def _build_xz_wp(self) -> Tuple[BluemiraWire, BluemiraWire, PhysicalComponent]:
        # Winding pack
        # x_min = self.wp_cross_section.bounding_box.x_min
        # x_centreline_in = self._centreline.bounding_box.x_min
        # dx = abs(x_min - x_centreline_in)
        wp_outer = offset_wire(
            self.centreline, 0.5 * self.params.tk_tf_wp.value, join="arc"
        )
        wp_inner = offset_wire(
            self.centreline, -0.5 * self.params.tk_tf_wp.value, join="arc"
        )

        winding_pack = PhysicalComponent(self.WP, BluemiraFace([wp_outer, wp_inner]))
        winding_pack.plot_options.face_options["color"] = BLUE_PALETTE["TF"][1]

        return wp_inner, wp_outer, winding_pack

    def _build_xz_ins(self, wp_inner: BluemiraWire, wp_outer: BluemiraWire) -> Component:
        # Insulation + Insertion gap
        offset_tk = self.params.tk_tf_ins.value + self.params.tk_tf_insgap.value

        ins_o_outer = offset_wire(wp_outer, offset_tk, join="arc")
        ins_outer = PhysicalComponent(self.OUT, BluemiraFace([ins_o_outer, wp_outer]))
        ins_outer.plot_options.face_options["color"] = BLUE_PALETTE["TF"][2]

        ins_i_inner = offset_wire(wp_inner, -offset_tk, join="arc")
        ins_inner = PhysicalComponent(self.IN, BluemiraFace([wp_inner, ins_i_inner]))
        ins_inner.plot_options.face_options["color"] = BLUE_PALETTE["TF"][2]

        return Component(self.INS, children=[ins_outer, ins_inner])

    def _build_xz_case(self, xyz_shape) -> Component:
        # Casing
        cas_inner, cas_outer = self._make_cas_xz(xyz_shape)

        cas_inner = PhysicalComponent(self.IN, cas_inner)
        cas_inner.plot_options.face_options["color"] = BLUE_PALETTE["TF"][0]

        cas_outer = PhysicalComponent(self.OUT, cas_outer)
        cas_outer.plot_options.face_options["color"] = BLUE_PALETTE["TF"][0]

        return Component(self.CASING, children=[cas_inner, cas_outer])

    def _build_xy_wp(self) -> List[Component]:

        # Winding pack
        # Should normally be gotten with wire_plane_intersect
        # (it's not OK to assume that the maximum x value occurs on the midplane)
        x_out = self._centreline.bounding_box.x_max
        xs = BluemiraFace(deepcopy(self.wp_cross_section))
        xs2 = deepcopy(xs)
        xs2.translate((x_out - xs2.center_of_mass[0], 0, 0))

        ib_wp_comp = PhysicalComponent(self.INB, xs)
        ib_wp_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][1]
        ob_wp_comp = PhysicalComponent(self.OUTB, xs2)
        ob_wp_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][1]
        return circular_pattern_component(
            Component(self.WP, children=[ib_wp_comp, ob_wp_comp]), self.params.n_TF.value
        )

    def _build_xy_ins(
        self, ins_inner_face: BluemiraFace, ins_outer_face: BluemiraFace
    ) -> List[Component]:
        # Insulation
        ib_ins_comp = PhysicalComponent(self.INB, ins_inner_face)
        ib_ins_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][2]
        ob_ins_comp = PhysicalComponent(self.OUTB, ins_outer_face)
        ob_ins_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][2]
        return circular_pattern_component(
            Component(self.INS, children=[ib_ins_comp, ob_ins_comp]),
            self.params.n_TF.value,
        )

    def _build_xy_case(
        self, ins_inner_face: BluemiraFace, ins_outer_face: BluemiraFace
    ) -> List[Component]:
        # Casing
        ib_cas_wire, ob_cas_wire = self._make_cas_xsec()
        cas_inner_face = BluemiraFace(
            [ib_cas_wire, deepcopy(ins_inner_face.boundary[0])]
        )
        cas_outer_face = BluemiraFace(
            [ob_cas_wire, deepcopy(ins_outer_face.boundary[0])]
        )

        ib_cas_comp = PhysicalComponent(self.INB, cas_inner_face)
        ib_cas_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][0]
        ob_cas_comp = PhysicalComponent(self.OUTB, cas_outer_face)
        ob_cas_comp.plot_options.face_options["color"] = BLUE_PALETTE["TF"][0]
        casing = Component(
            self.CASING,
            children=[ib_cas_comp, ob_cas_comp],
        )
        return circular_pattern_component(casing, self.params.n_TF.value)

    def _build_xyz_wp(
        self, n_sectors: int, sector_degree: float
    ) -> Tuple[BluemiraSolid, List[PhysicalComponent]]:
        # Winding pack
        wp_solid = sweep_shape(self.wp_cross_section, self._centreline)
        winding_pack = PhysicalComponent(self.WP, wp_solid)
        winding_pack.display_cad_options.color = BLUE_PALETTE["TF"][1]
        return wp_solid, circular_pattern_component(
            winding_pack, n_sectors, degree=sector_degree * n_sectors
        )

    def _build_xyz_ins(
        self,
        wp_solid: BluemiraSolid,
        ins_inner_face: BluemiraFace,
        n_sectors: int,
        sector_degree: float,
    ) -> Tuple[BluemiraSolid, List[PhysicalComponent]]:
        # Insulation
        # Get the insulation internal boundary sweep it over the centreline
        # and cut out the windingpack
        ins_solid = boolean_cut(
            sweep_shape(ins_inner_face.boundary[0], self._centreline), wp_solid
        )[0]
        insulation = PhysicalComponent(
            self.INS,
            ins_solid,
        )
        insulation.display_cad_options.color = BLUE_PALETTE["TF"][2]
        return ins_solid, circular_pattern_component(
            insulation, n_sectors, degree=sector_degree * n_sectors
        )

    def _build_xyz_case(
        self, ins_solid: BluemiraSolid, n_sectors: int, sector_degree: float
    ) -> List[PhysicalComponent]:
        # Casing
        # Normally I'd do lots more here to get to a proper casing
        # This is just a proof-of-principle
        inner_xs, outer_xs = self._make_cas_xsec()

        # Make inner xs into a rectangle
        bb = inner_xs.bounding_box
        x_in = np.zeros(4)
        x_in[[0, -1]] = bb.x_min
        x_in[[1, 2]] = bb.x_max

        y_in = np.full(4, self.params.r_tf_in * np.tan(np.pi / self.params.n_TF.value))
        y_in[:2] = -y_in[:2]
        inner_xs_rect = make_polygon([x_in, y_in, np.zeros(4)], closed=True)

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
            solid, BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 0, 1])
        )
        cut_wires.sort(key=lambda wire: wire.length)
        bb = cut_wires[-1].bounding_box
        z_min = bb.z_min
        z_max = bb.z_max

        z_min_arr = np.full(4, z_min)
        z_max_arr = np.full(4, z_max)
        y_in = np.full(4, 0.5 * (self.params.tf_wp_depth + self.params.tk_tf_side))
        y_in[:2] = -y_in[:2]

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
                y_in,
                z_max_arr,
            ],
            closed=True,
        )
        joiner_top = extrude_shape(BluemiraFace(joiner_top), (0, 0, -z_max))
        joiner_bot = make_polygon(
            [
                [x_min, x_turn_bot, x_turn_bot, x_min],
                y_in,
                z_min_arr,
            ],
            closed=True,
        )
        joiner_bot = extrude_shape(BluemiraFace(joiner_bot), (0, 0, -z_min))

        # Need to cut away the excess joiner extrusions
        cl = deepcopy(self._centreline)
        cl.translate((0, -2 * self.params.tf_wp_depth, 0))
        cutter = extrude_shape(BluemiraFace(cl), (0, 4 * self.params.tf_wp_depth, 0))
        joiner_top = boolean_cut(joiner_top, cutter)[0]
        joiner_bot = boolean_cut(joiner_bot, cutter)[0]

        # Cut away straight sweep before fusing to protect against degenerate faces
        # Keep the largest piece
        pieces = boolean_cut(solid, inboard_casing)
        pieces.sort(key=lambda x: x.volume)
        solid = pieces[-1]

        case_solid = boolean_fuse([solid, inboard_casing, joiner_top, joiner_bot])
        case_solid_hollow = boolean_cut(
            case_solid, BluemiraSolid(ins_solid.boundary[0])
        )[0]

        casing = PhysicalComponent(self.CASING, case_solid_hollow)
        casing.display_cad_options.color = BLUE_PALETTE["TF"][0]
        return circular_pattern_component(
            casing, n_sectors, degree=n_sectors * sector_degree
        )

    def _make_ins_xsec(self):
        """
        Make the insulation + insertion gap x-y cross-section faces
        """
        ins_outer = offset_wire(
            self.wp_cross_section,
            self.params.tk_tf_ins.value + self.params.tk_tf_insgap.value,
        )
        face = BluemiraFace([ins_outer, self.wp_cross_section])

        outer_face = deepcopy(face)
        outer_face.translate(
            (self._centreline.bounding_box.x_max - outer_face.center_of_mass[0], 0, 0)
        )
        return face, outer_face

    def _make_cas_xsec(self):
        """
        Make the casing x-y cross-section wires
        """
        x_in = self.params.r_tf_in
        # Insulation and insertion gap included in WP width
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
        dx_out = np.full(
            4,
            dx_ins
            + 0.5 * (self.params.tk_tf_front_ib.value + self.params.tk_tf_nose.value),
        )
        dx_out[[0, -1]] = -dx_out[[0, -1]]

        dy_out = np.full(4, dy_ins + self.params.tk_tf_side.value)
        dy_out[[0, 1]] = -dy_out[[0, 1]]

        outboard_wire = make_polygon([dx_out, dy_out, np.zeros(4)], closed=True)
        outboard_wire.translate((self._centreline.bounding_box.x_max, 0, 0))

        return inboard_wire, outboard_wire

    def _make_cas_xz(self, solid: BluemiraSolid) -> Tuple[BluemiraFace, BluemiraFace]:
        """
        Make the casing x-z cross-section from a 3-D volume.
        """
        wires = slice_shape(
            solid, BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 0, 1])
        )
        wires.sort(key=lambda wire: wire.length)
        if len(wires) != 4:
            raise BuilderError(
                "Unexpected TF coil x-z cross-section. It is likely that a previous "
                "boolean cutting operation failed to create a hollow solid."
            )

        return BluemiraFace([wires[1], wires[0]]), BluemiraFace([wires[3], wires[2]])

    def _make_field_solver(self):
        """
        Make a magnetostatics solver for the field from the TF coils.
        """
        circuit = ArbitraryPlanarRectangularXSCircuit(
            self._centreline.discretize(byedges=True, ndiscr=100),
            breadth=0.5 * self.params.tk_tf_wp.value,
            depth=0.5 * self.params.tk_tf_wp_y.value,
            current=1,
        )
        solver = HelmholtzCage(circuit, self.params.n_TF.value)
        field = solver.field(self.params.R_0, 0, self.params.z_0)
        current = -self.params.B_0 / field[1]  # single coil amp-turns
        solver.set_current(current)
        return solver

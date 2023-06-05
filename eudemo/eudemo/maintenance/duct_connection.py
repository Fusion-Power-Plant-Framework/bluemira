# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Creating ducts for the port
"""
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.error import BuilderError
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.reactor_config import ConfigParams
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import extrude_shape, make_polygon
from bluemira.materials import Void


@dataclass
class TSUpperPortDuctBuilderParams(ParameterFrame):
    """Thermal shield upper port duct builder Parameter Frame"""

    n_TF: Parameter[int]
    tf_wp_depth: Parameter[float]
    g_ts_tf: Parameter[float]
    tk_ts: Parameter[float]


class TSUpperPortDuctBuilder(Builder):
    """Thermal shield upper port duct builder"""

    params: TSUpperPortDuctBuilderParams
    param_cls = TSUpperPortDuctBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        port_koz: BluemiraFace,
    ):
        super().__init__(params, None)
        self.x_min = port_koz.bounding_box.x_min
        self.x_max = port_koz.bounding_box.x_max
        self.z_max = port_koz.bounding_box.z_max

        if self.params.tk_ts.value <= 0:
            raise ValueError("Port wall thickness must be > 0")

        self.y_offset = self.params.tf_wp_depth.value + self.params.g_ts_tf.value

    def build(self) -> Component:
        """Build upper port"""
        xy_face = make_upper_port_xy_face(
            self.params.n_TF.value,
            self.x_min,
            self.x_max,
            self.params.tk_ts.value,
            self.params.tk_ts.value,
            self.y_offset,
        )

        return self.component_tree(
            None, [self.build_xy(xy_face)], [self.build_xyz(xy_face)]
        )

    def build_xyz(self, xy_face: BluemiraFace) -> PhysicalComponent:
        """Build upper port xyz"""
        port = extrude_shape(xy_face, (0, 0, self.z_max))
        comp = PhysicalComponent(self.name, port)
        apply_component_display_options(comp, BLUE_PALETTE["TS"][0])
        return comp

    def build_xy(self, face: BluemiraFace) -> PhysicalComponent:
        """Build upport port xy face"""
        comp = PhysicalComponent(self.name, face)
        apply_component_display_options(comp, BLUE_PALETTE["TS"][0])
        return comp


@dataclass
class VVUpperPortDuctBuilderParams(ParameterFrame):
    """Vacuum vessel upper port duct builder Parameter Frame"""

    n_TF: Parameter[int]
    tf_wp_depth: Parameter[float]
    g_ts_tf: Parameter[float]
    tk_ts: Parameter[float]
    g_vv_ts: Parameter[float]
    tk_upper_port_wall_end: Parameter[float]
    tk_upper_port_wall_side: Parameter[float]


class VVUpperPortDuctBuilder(Builder):
    """Vacuum vessel upper port duct builder"""

    params: VVUpperPortDuctBuilderParams
    param_cls = VVUpperPortDuctBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        port_koz: BluemiraFace,
    ):
        super().__init__(params, None)
        koz_offset = self.params.tk_ts.value + self.params.g_vv_ts.value
        self.x_min = port_koz.bounding_box.x_min + koz_offset
        self.x_max = port_koz.bounding_box.x_max - koz_offset
        self.z_max = port_koz.bounding_box.z_max

        if (
            self.params.tk_upper_port_wall_end.value <= 0
            or self.params.tk_upper_port_wall_side.value <= 0
        ):
            raise ValueError("Port wall thickness must be > 0")

        self.y_offset = (
            self.params.tf_wp_depth.value
            + self.params.g_ts_tf.value
            + self.params.tk_ts.value
            + self.params.g_vv_ts.value
        )

    def build(self) -> Component:
        """Build upper port"""
        xy_face = make_upper_port_xy_face(
            self.params.n_TF.value,
            self.x_min,
            self.x_max,
            self.params.tk_upper_port_wall_end.value,
            self.params.tk_upper_port_wall_side.value,
            self.y_offset,
        )
        xy_voidface = BluemiraFace(xy_face.boundary[1])

        return self.component_tree(
            None,
            self.build_xy(xy_face, xy_voidface),
            self.build_xyz(xy_face, xy_voidface),
        )

    def build_xyz(
        self, xy_face: BluemiraFace, xy_voidface: BluemiraFace
    ) -> PhysicalComponent:
        """Build upper port xyz"""
        port = extrude_shape(xy_face, (0, 0, self.z_max))
        comp = PhysicalComponent(self.name, port)
        apply_component_display_options(comp, BLUE_PALETTE["VV"][0])
        void = PhysicalComponent(
            self.name + " voidspace",
            extrude_shape(xy_voidface, (0, 0, self.z_max)),
            material=Void("vacuum"),
        )
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]

    def build_xy(
        self, face: BluemiraFace, xy_voidface: BluemiraFace
    ) -> PhysicalComponent:
        """Build upper port xy face"""
        comp = PhysicalComponent(self.name, face)
        apply_component_display_options(comp, BLUE_PALETTE["VV"][0])
        void = PhysicalComponent(
            self.name + " voidspace", xy_voidface, material=Void("vacuum")
        )
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]


def make_upper_port_xy_face(
    n_TF: int,
    x_min: float,
    x_max: float,
    wall_end_tk: float,
    wall_side_tk: float,
    y_offset: float,
) -> BluemiraFace:
    """
    Creates a xy cross section of the port

    translates the port koz to the origin,
    builds the port at the origin and moves it back

    Parameters
    ----------
    n_TF:
        Number of TF coils
    x_min:
        Inner radius of the port keep-out zone
    x_max:
        Outer radius of the port keep-out zone
    wall_end_tk:
        Port wall end thickness
    wall_size_tk:
        Port wall side thickness
    y_offset:
        Offset value from the x-z plane at which to start building the port
        (excluding port side wall thickness)

    Returns
    -------
    xy_face:
        x-y face of the upper port

    Notes
    -----
    the port koz is slightly trimmed to allow for square ends to the port

    """
    half_beta = np.pi / n_TF
    cos_hb = np.cos(half_beta)
    tan_hb = np.tan(half_beta)

    y_tf_out = y_offset / cos_hb
    y_tf_in = y_tf_out + wall_side_tk / cos_hb

    x1 = x_min

    # This is the correct way to retrieve the outer radius of the port, accounting
    # for the TF thickness
    # It's just the intersection of a line and a circle in the positive quadrant.
    a1 = 1 + tan_hb**2
    b1 = -2 * y_tf_out * tan_hb
    c1 = y_tf_out**2 - x_max**2
    discriminant = b1**2 - 4 * a1 * c1
    x4 = 0.5 * (-b1 + np.sqrt(discriminant)) / a1

    x2, x3 = x1 + wall_end_tk, x4 - wall_end_tk

    if x2 >= x3:
        raise BuilderError("Port dimensions too small")

    y1 = x1 * tan_hb - y_tf_out

    if y1 < 0:
        # Triangular outer port wall
        y1 = 0
        x1 = y_tf_out / tan_hb
        x2 = x1 + wall_end_tk

    y2, y3 = x2 * tan_hb - y_tf_in, x3 * tan_hb - y_tf_in

    if y3 <= 0:
        raise BuilderError("Port dimensions too small")

    if y2 < 0:
        # Triangular inner port wall
        y2 = 0
        c = y3 - tan_hb * x3
        x2 = -c / tan_hb
        x1 = x2 - wall_end_tk

    y1, y4 = x1 * tan_hb - y_tf_out, x4 * tan_hb - y_tf_out

    inner_wire = make_polygon(
        {"x": [x2, x3, x3, x2], "y": [-y2, -y3, y3, y2]}, closed=True
    )
    outer_wire = make_polygon(
        {"x": [x1, x4, x4, x1], "y": [-y1, -y4, y4, y1]}, closed=True
    )

    xy_face = BluemiraFace((outer_wire, inner_wire))
    xy_face.rotate(degree=np.rad2deg(half_beta))

    return xy_face


if __name__ == "__main__":
    from bluemira.base.parameter_frame import ParameterFrame
    from bluemira.display import show_cad
    from bluemira.geometry.parameterisations import PrincetonD
    from bluemira.geometry.tools import (
        boolean_cut,
        boolean_fragments,
        boolean_fuse,
        extrude_shape,
        make_polygon,
        point_inside_shape,
    )
    from eudemo.vacuum_vessel import VacuumVesselBuilder, VacuumVesselBuilderParams

    p = PrincetonD().create_shape()

    params = VacuumVesselBuilderParams.from_dict(
        {
            "n_TF": {"value": 16, "unit": ""},
            "r_vv_ib_in": {"value": 3.0, "unit": "m"},
            "r_vv_ob_in": {"value": 9.0, "unit": "m"},
            "tk_vv_in": {"value": 0.6, "unit": "m"},
            "tk_vv_out": {"value": 1.1, "unit": "m"},
            "g_vv_bb": {"value": 0.05, "unit": "m"},
            "vv_in_off_deg": {"value": 20, "unit": "deg"},
            "vv_out_off_deg": {"value": 160, "unit": "deg"},
        }
    )
    builder = VacuumVesselBuilder(params, {}, ivc_koz=p)
    VV = builder.build()

    port_koz = make_polygon({"x": [7, 12, 12, 7], "z": [20, 20, 0, 0]}, closed=True)
    params = VVUpperPortDuctBuilderParams.from_dict(
        {
            "n_TF": {"value": 16, "unit": ""},
            "tf_wp_depth": {"value": 1.0, "unit": "m"},
            "g_ts_tf": {"value": 0.05, "unit": "m"},
            "tk_ts": {"value": 0.05, "unit": "m"},
            "g_vv_ts": {"value": 0.05, "unit": "m"},
            "tk_upper_port_wall_end": {"value": 0.2, "unit": "m"},
            "tk_upper_port_wall_side": {"value": 0.1, "unit": "m"},
        }
    )
    builder = VVUpperPortDuctBuilder(params, port_koz=port_koz)
    UP = builder.build()
    c = Component("test", children=[VV, UP])
    c.show_cad()

    vv_xyz = VV.get_component("xyz").get_component("Sector 1")
    target_void = vv_xyz.get_component("Vessel voidspace 1").shape
    target_shape = vv_xyz.get_component("Body 1").shape
    up_xyz = UP.get_component("xyz")
    tool_void = up_xyz.get_component("VVUpperPortDuct voidspace").shape
    tool_shape = up_xyz.get_component("VVUpperPortDuct").shape

    import time

    def pipe_pipe_join(target_shape, target_void, tool_shape, tool_void):
        """Naive"""
        t1 = time.time()
        void = boolean_fuse([target_void, tool_void])
        shape = boolean_fuse([target_shape, tool_shape])
        shape = boolean_cut(shape, void)
        print(f"{time.time()-t1} seconds")
        return shape, void

    def pipe_pipe_join2(target_shape, target_void, tool_shape, tool_void):
        """Smart but dumb"""
        t1 = time.time()
        void = boolean_fuse([target_void, tool_void])
        _, void_fragments = boolean_fragments([target_void, tool_void])
        target_void_fragments, tool_void_fragments = void_fragments
        new_void_pieces = []
        for void_frag in tool_void_fragments:
            if not point_inside_shape(void_frag.center_of_mass, target_void):
                new_void_pieces.append(void_frag)

        _, fragments = boolean_fragments([target_shape, tool_shape])
        target_fragments, tool_fragments = fragments

        new_shape_pieces = []
        for targ_frag in target_fragments:
            # Find and remove the target piece(s) that are inside the tool
            com = targ_frag.center_of_mass
            for tool_void_frag in new_void_pieces:
                if point_inside_shape(com, tool_void_frag):
                    continue
                else:
                    new_shape_pieces.append(targ_frag)

        for tool_frag in tool_fragments:
            # Find and remove the tool piece(s) that are inside the target
            com = tool_frag.center_of_mass
            if not point_inside_shape(com, target_void):
                new_shape_pieces.append(tool_frag)
            else:
                for targ_frag in target_fragments:
                    # Find the union piece
                    if tool_frag.is_same(targ_frag):
                        new_shape_pieces.append(tool_frag)

        shape = boolean_fuse(new_shape_pieces)
        print(f"{time.time()-t1} seconds")
        return shape, void
        # compound, fragments = boolean_fragments([target_shape, tool_shape])

        # target_fragments, tool_fragments = fragments
        # # Find the piece to remove from the target
        # new_shape_pieces = []
        # for solid in target_fragments:
        #     if point_inside
        #     for other in tool_fragments:
        #         if solid.is_same(other):
        #             new_target.append(solid)
        #             # target_fragments.remove(other)
        #         else:
        #             if point_inside_shape(solid.center_of_mass, target_shape):
        #                 new_target.append(solid)

        # for solid in tool_fragments:
        #     pass

        # new_target = boolean_fuse(new_target)
        # new_tool = boolean_fuse(new_tool)
        # return new_target, new_tool

    # shape, void = pipe_pipe_join(target_shape, target_void, tool_shape, tool_void)
    shape2, void2 = pipe_pipe_join2(target_shape, target_void, tool_shape, tool_void)

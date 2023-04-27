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
        self.port_koz = port_koz.deepcopy()

        if (
            self.params.tk_upper_port_wall_end.value <= 0
            or self.params.tk_upper_port_wall_side.value <= 0
        ):
            raise ValueError("Port wall thickness must be > 0")

        self.y_offset = self.params.tf_wp_depth.value + self.params.g_ts_tf

    def build(self) -> Component:
        """Build upper port"""
        xy_face = make_upper_port_xy_face(
            self.params.n_TF.value,
            self.port_koz.bounding_box.x_min,
            self.port_koz.bounding_box.x_max,
            self.params.tk_ts.value,
            self.params.tk_ts.value,
            self.y_offset,
        )

        return self.component_tree(
            None, [self.build_xy(xy_face)], [self.build_xyz(xy_face)]
        )

    def build_xyz(self, xy_face: BluemiraFace) -> PhysicalComponent:
        """Build upper port xyz"""
        port = extrude_shape(xy_face, (0, 0, self.port_koz.bounding_box.z_max))
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
        self.port_koz = port_koz.deepcopy()

        if (
            self.params.tk_upper_port_wall_end.value <= 0
            or self.params.tk_upper_port_wall_side.value <= 0
        ):
            raise ValueError("Port wall thickness must be > 0")

        self.y_offset = (
            self.params.tf_wp_depth.value
            + self.params.g_ts_tf
            + self.params.tk_ts.value
            + self.params.g_vv_ts.value
        )

    def build(self) -> Component:
        """Build upper port"""
        xy_face = make_upper_port_xy_face(
            self.params.n_TF.value,
            self.port_koz.bounding_box.x_min,
            self.port_koz.bounding_box.x_max,
            self.params.tk_ts.value,
            self.params.tk_ts.value,
            self.y_offset,
        )

        return self.component_tree(
            None, [self.build_xy(xy_face)], [self.build_xyz(xy_face)]
        )

    def build_xyz(self, xy_face: BluemiraFace) -> PhysicalComponent:
        """Build upper port xyz"""
        port = extrude_shape(xy_face, (0, 0, self.port_koz.bounding_box.z_max))
        comp = PhysicalComponent(self.name, port)
        apply_component_display_options(comp, BLUE_PALETTE["VV"][0])
        return comp

    def build_xy(self, face: BluemiraFace) -> PhysicalComponent:
        """Build upper port xy face"""
        comp = PhysicalComponent(self.name, face)
        apply_component_display_options(comp, BLUE_PALETTE["VV"][0])
        return comp


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

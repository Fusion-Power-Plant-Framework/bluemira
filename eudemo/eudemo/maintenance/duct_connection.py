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
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.reactor_config import ConfigParams
from bluemira.builders.tools import apply_component_display_options, get_n_sectors
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import extrude_shape, make_polygon, slice_shape, split_wire
from bluemira.geometry.wire import BluemiraWire


@dataclass
class UpperPortDuctBuilderParams(ParameterFrame):
    """Duct Builder Parameter Frame"""

    n_TF: Parameter[int]
    tk_upper_port_wall: Parameter[float]


class UpperPortDuctBuilder(Builder):
    """Duct Builder"""

    params: UpperPortDuctBuilderParams
    param_cls = UpperPortDuctBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        port_koz: BluemiraFace,
        tf_coil_thickness: float,
    ):
        super().__init__(params, None)
        self.port_koz = port_koz.deepcopy()

        if self.params.tk_upper_port_wall.value <= 0:
            raise ValueError("Port wall thickness must be > 0")

        self.tf_coil_thickness = tf_coil_thickness

    def build(self) -> Component:
        """Build upper port"""
        xy_face = self._single_xy_face()

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
        """Build upport port xy face"""
        comp = PhysicalComponent(self.name, face)
        apply_component_display_options(comp, BLUE_PALETTE["VV"][0])
        return comp

    def _single_xy_face(self) -> BluemiraFace:
        """
        Creates a xy cross section of the port

        translates the port koz to the origin,
        builds the port at the origin and moves it back

        Notes
        -----
        the port koz is slightly trimmed to allow for square ends to the port

        """
        half_sector_degree = 0.5 * get_n_sectors(self.params.n_TF.value)[0]
        half_beta = np.deg2rad(half_sector_degree)
        cos_hb = np.cos(half_beta)
        tf_tk_in_y = self.tf_coil_thickness * cos_hb
        tk_y_prt = tf_tk_in_y + self.params.tk_upper_port_wall.value
        end_tk = 2 * self.params.tk_upper_port_wall.value

        koz_bb = self.port_koz.bounding_box

        x_min = max(
            (tk_y_prt) * np.tan(half_beta),
            koz_bb.x_min / cos_hb,
        )
        x_min_ins = x_min + end_tk
        x_max = koz_bb.x_max * cos_hb
        x_max_ins = x_max - end_tk

        outer = make_polygon({"x": [x_min, x_max], "y": 0})
        inner = make_polygon({"x": [x_min_ins, x_max_ins], "y": 0})

        outer_lower = outer.deepcopy()
        inner_lower = inner.deepcopy()
        outer.translate((0, -tf_tk_in_y, 0))
        outer_lower.translate((0, tf_tk_in_y, 0))
        inner.translate((0, -tk_y_prt, 0))
        inner_lower.translate((0, tk_y_prt, 0))

        outer.rotate(degree=half_sector_degree)
        inner.rotate(degree=half_sector_degree)
        outer_lower.rotate(degree=-half_sector_degree)
        inner_lower.rotate(degree=-half_sector_degree)

        if (
            outer.start_point()[1] - outer_lower.start_point()[1] < end_tk
            or inner.start_point()[1] < inner_lower.start_point()[1]
        ):
            delta_y = (
                ((end_tk - (outer.start_point()[1] - outer_lower.start_point()[1])) / 2)
                + outer.start_point()[1]
            )[0]
            y_plane = BluemiraPlane((0, delta_y, 0), (0, 1, 0))
            outer = split_wire(outer, slice_shape(outer, y_plane)[0])[1]

            x_plane = BluemiraPlane((outer.start_point()[0] + end_tk, 0, 0), (1, 0, 0))

            try:
                inner = split_wire(inner, slice_shape(inner, x_plane)[0])[1]
            except TypeError:
                raise GeometryError("Port dimensions too small")

            outer_lower = outer.deepcopy()
            outer_lower.rotate(direction=(1, 0, 0))

            inner_lower = inner.deepcopy()
            inner_lower.rotate(direction=(1, 0, 0))

        outer_join1 = make_polygon(
            np.array([outer.start_point(), outer_lower.start_point()])
        )
        outer_join2 = make_polygon(
            np.array([outer.end_point(), outer_lower.end_point()])
        )

        inner_join1 = make_polygon(
            np.array([inner.start_point(), inner_lower.start_point()])
        )
        inner_join2 = make_polygon(
            np.array([inner.end_point(), inner_lower.end_point()])
        )

        outer_wire = BluemiraWire((outer_join1, outer, outer_join2, outer_lower))
        inner_wire = BluemiraWire((inner_join1, inner, inner_join2, inner_lower))

        xy_face = BluemiraFace((outer_wire, inner_wire))
        xy_face.rotate(degree=half_sector_degree)

        return xy_face

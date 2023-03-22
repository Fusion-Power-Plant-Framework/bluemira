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
from typing import Dict, Tuple, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.reactor_config import ConfigParams
from bluemira.builders.tools import get_n_sectors
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import extrude_shape, make_polygon, mirror_shape
from bluemira.geometry.wire import BluemiraWire


@dataclass
class DuctBuilderParams(ParameterFrame):
    """Duct Builder Parameter Frame"""

    n_TF: Parameter[int]


class DuctBuilder(Builder):
    """Duct Builder"""

    param_cls = DuctBuilderParams

    NAME = "Upper Port Duct"

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        build_config: Dict,
        port_koz: BluemiraFace,
        port_wall_thickness: float,
        tf_coil_thickness: float,
    ):
        super().__init__(params, build_config)
        self.port_koz = port_koz.deepcopy()
        self.port_wall_thickness = port_wall_thickness
        self.tf_coil_thickness = tf_coil_thickness

    def build(self) -> Component:
        """Build upper port"""
        xy_face = self._single_xy_face()

        return self.component_tree(None, None, [self.build_xyz(xy_face)])

    def build_xyz(self, xy_face: BluemiraFace) -> PhysicalComponent:
        """Build upper port xyz"""
        port = extrude_shape(xy_face, (0, 0, self.port_koz.bounding_box.z_max))
        comp = PhysicalComponent(self.NAME, port)
        comp.display_cad_options.color = BLUE_PALETTE["VV"][0]

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
        x_double_thick = np.sqrt(3 * self.port_wall_thickness**2)

        # Inner point
        y_min = self.tf_coil_thickness
        x_min = max(self.port_koz.bounding_box.x_min, y_min / np.cos(half_beta))

        # Outer point
        o_a = y_min / self.port_koz.bounding_box.x_max
        x_a = self.port_koz.bounding_box.x_max * np.cos(np.arcsin(o_a))

        outer_line = make_polygon({"x": [x_min, x_a], "y": [y_min, y_min]})
        inner_line = make_polygon(
            {
                "x": [x_min + x_double_thick, x_a - x_double_thick],
                "y": np.full(2, y_min + self.port_wall_thickness),
            }
        )

        mirror_line = outer_line.deepcopy()
        mirror_line.rotate(degree=half_sector_degree)
        mirror_point = (
            mirror_line.bounding_box.x_max,
            mirror_line.bounding_box.y_max,
            0,
        )

        inner_bb, inner_top_bb = _mirror_line(inner_line, mirror_point)
        outer_bb, outer_top_bb = _mirror_line(outer_line, mirror_point)

        # Outer wire
        x_ow = np.array(
            [outer_bb.x_min, outer_bb.x_max, outer_top_bb.x_max, outer_top_bb.x_min]
        )
        y_ow = np.array(
            [outer_bb.y_min, outer_bb.y_max, outer_top_bb.y_max, outer_top_bb.y_min]
        )

        mirror_bug(x_ow)
        mirror_bug(y_ow)

        xy_outer_wire = make_polygon({"x": x_ow, "y": y_ow}, closed=True)

        # Inner Wire
        x_iw = np.array(
            [inner_bb.x_min, inner_bb.x_max, inner_top_bb.x_max, inner_top_bb.x_min]
        )
        y_iw = np.array(
            [inner_bb.y_min, inner_bb.y_max, inner_top_bb.y_max, inner_top_bb.y_min]
        )

        mirror_bug(x_iw)
        mirror_bug(y_iw)

        xy_inner_wire = make_polygon({"x": x_iw, "y": y_iw}, closed=True)

        xy_face = BluemiraFace((xy_outer_wire, xy_inner_wire))

        return xy_face


def _mirror_line(line: BluemiraWire, mirror_point: Tuple[float, ...]):
    new_line = mirror_shape(line, (0, 0, 0), mirror_point)
    return line.bounding_box, new_line.bounding_box


def mirror_bug(arr):
    arr[2:] = abs(arr[2:][::-1])

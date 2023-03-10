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
from bluemira.builders.tools import get_n_sectors
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire


@dataclass
class DuctBuilderParams(ParameterFrame):
    n_TF: Parameter[int]
    port_wall_thickness: Parameter[float]


class DuctBuilder(Builder):
    param_cls = DuctBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        build_config: Dict,
        vacuum_vessel: BluemiraSolid,
        cryostat: BluemiraSolid,
        port_koz: BluemiraFace,
        tf_coil_casing1: BluemiraSolid,
        tf_coil_casing2: BluemiraSolid,
    ):
        super().__init__(params, build_config)
        self.port_koz = port_koz.deepcopy()
        self.tf_coil_casing = tf_coil_casing1

    def build(self) -> Component:
        xy_wire = _single_xy_wire()

        return self.component_tree(
            [self.build_xz()], [self.build_xy(xy_wire)], [self.build_xyz(xy_wire)]
        )

    def build_xz(self) -> PhysicalComponent:
        component = PhysicalComponent(name="Duct", shape=None)

        return component

    def build_xy(self, xy_wire: BluemiraWire) -> PhysicalComponent:
        pass

    def build_xyz(self, xy_wire: BluemiraWire) -> PhysicalComponent:
        # xin = max(xin) if len(xin) != 0 else 0

        # xout = (
        #     max(xout) + self.params.R_0 if len(xout) != 0 else 10 * self.params.R_0
        # )  # big
        # ztop = max(ztop) + self.params.pf_off
        # pf1_ro = max(xin, xmin)
        # pf2_ri = min(xout, xmax)
        # tf_w = (
        #     inputs["TFsection"]["case"]["side"] * 2
        #     + inputs["TFsection"]["winding_pack"]["depth"]
        # )
        # x_b, y_b = 0.5 * tf_w / np.tan(beta), 0.5 * tf_w
        # x_bp = np.sqrt(y_b**2 + x_b**2)
        # x_oi = x_bp + self.params.g_ts_tf / np.sin(beta)
        # x_inner = pf1_ro + self.params.g_ts_pf
        # y_inner = np.tan(beta) * (x_inner - x_oi)
        # x_outer = np.cos(beta) * ((pf2_ri - self.params.g_ts_pf)) + np.sin(beta) * (
        #     tf_w / 2 + self.params.g_ts_tf
        # )  # trimmed
        # y_outer = np.tan(beta) * (x_outer - x_oi)

        pass

    def _single_xy_wire(self, width: float) -> BluemiraWire:
        """
        Creates a xy cross section of the port

        translates the port koz to the origin,
        builds the port at the origin and moves it back

        Parameters
        ----------
        width
            the width of half the tf coil and the vacuum vessel and shield

        """
        sector_degree, _ = get_n_sectors(self.params.n_TF.value)

        x_orig = self.port_koz.bounding_box.x_min

        # build at origin
        self.port_koz.translate((-x_orig, 0, 0))
        # Inner point
        x_min = self.port_koz.bounding_box.x_min
        y_min = self.port_koz.bounding_box.y_max

        # lower outer point (xy_plane)
        x_a = self.port_koz.bounding_box.x_max
        y_a = self.port_koz.bounding_box.y_max

        rotated_koz = self.port_koz.deepcopy()
        rotated_koz.rotate(degree=sector_degree)

        # upper outer point (xy plane)
        x_b = rotated_koz.bounding_box.x_max
        y_b = rotated_koz.bounding_box.y_max

        port_xy_wire = make_polygon(
            {"x": [x_min, x_a, x_b], "y": [y_min, y_a, y_b]}, closed=True
        )

        port_xy_wire.translate((x_orig, width, 0))

        return port_xy_wire

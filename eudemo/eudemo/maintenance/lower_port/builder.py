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
EUDEMO Lower Port Builder
"""

from typing import Dict, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.displayer import show_cad
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, boolean_fuse, extrude_shape, offset_wire
from bluemira.geometry.wire import BluemiraWire
from eudemo.maintenance.lower_port.parameterisations import LowerPortBuilderParams


class LowerPortBuilder(Builder):
    """
    Lower Port Builder
    """

    DUCT = "Lower Duct"
    param_cls = LowerPortBuilderParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
        lower_duct_xz: BluemiraFace,
        angled_duct_boundary: BluemiraWire,
        straight_duct_boundary: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.lower_duct_xz = lower_duct_xz
        self.angled_duct_boundary = angled_duct_boundary
        self.straight_duct_boundary = straight_duct_boundary

        self.lower_duct_wall_tk = self.params.lower_duct_wall_tk.value
        self.lower_duct_angle = self.params.lower_duct_angle.value
        self.n_TF = self.params.n_TF.value

    def build(self) -> Component:
        """
        Build the Lower Port.
        """
        return self.component_tree(
            xz=[self.build_xz()],
            xy=None,
            xyz=[self.build_xyz()],
        )

    def build_xz(self) -> Component:
        """
        Build the Lower Port in XZ.
        """
        pc = PhysicalComponent(self.DUCT, self.lower_duct_xz)
        apply_component_display_options(pc, color=BLUE_PALETTE["BB"][0])
        return pc

    def build_xyz(self) -> Component:
        """
        Build the Lower Port in XYZ.
        """
        angled_duct_face = BluemiraFace(self.angled_duct_boundary)
        angled_duct_hollow_face = self._hollow_face(
            angled_duct_face,
            thickness=self.lower_duct_wall_tk,
        )
        straight_duct_face = BluemiraFace(self.straight_duct_boundary)
        straight_duct_hollow_face = self._hollow_face(
            straight_duct_face,
            thickness=self.lower_duct_wall_tk,
        )

        duct_heading_x = np.cos(np.deg2rad(self.lower_duct_angle)) * 50
        duct_heading_z = np.sin(np.deg2rad(self.lower_duct_angle)) * 50
        angled_duct = extrude_shape(
            angled_duct_hollow_face,
            (duct_heading_x, 0, duct_heading_z),
        )

        straight_duct_backwall = extrude_shape(
            straight_duct_face,
            straight_duct_face.normal_at() * -self.lower_duct_wall_tk,
        )
        straight_duct_length = extrude_shape(
            straight_duct_hollow_face,
            straight_duct_hollow_face.normal_at() * 40,
        )
        straight_duct = boolean_fuse([straight_duct_backwall, straight_duct_length])

        angled_pieces = boolean_cut(angled_duct, [straight_duct])
        angled_top = boolean_cut(angled_duct, [angled_pieces[1]])[0]

        straight_with_hole = boolean_cut(straight_duct, [angled_top])[0]

        duct = boolean_fuse([angled_top, straight_with_hole])

        # rotate duct to correct position
        duct.rotate(degree=np.rad2deg(np.pi / self.n_TF))

        pc = PhysicalComponent(self.DUCT, duct)
        apply_component_display_options(pc, color=BLUE_PALETTE["VV"][0])

        return pc

    def _hollow_face(
        self,
        outer_face: BluemiraFace,
        thickness: float,
    ) -> BluemiraFace:
        inner = offset_wire(outer_face.wires[0], -thickness)
        inner = BluemiraFace(inner)

        resolved = boolean_cut(outer_face, [inner])
        return resolved[0]

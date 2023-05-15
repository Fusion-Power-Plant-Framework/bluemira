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
EUDEMO Lower Port Duct Builder
"""

from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, boolean_fuse, extrude_shape, offset_wire
from bluemira.geometry.wire import BluemiraWire


@dataclass
class LowerPortBuilderParams(ParameterFrame):
    """Lower Port Duct Builder Parameters"""

    n_TF: Parameter[int]

    lp_duct_angle: Parameter[float]
    lp_duct_wall_tk: Parameter[float]


class LowerPortBuilder(Builder):
    """
    Lower Port Duct Builder
    """

    param_cls = LowerPortBuilderParams
    DUCT = "Lower Port Duct"

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
        duct_xz_koz: BluemiraFace,
        duct_angled_nowall_extrude_boundary: BluemiraWire,
        duct_straight_nowall_extrude_boundary: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.duct_xz_koz = duct_xz_koz
        self.duct_angled_boundary = duct_angled_nowall_extrude_boundary
        self.duct_straight_boundary = duct_straight_nowall_extrude_boundary

        self.n_TF = self.params.n_TF.value
        self.duct_angle = self.params.lp_duct_angle.value
        self.duct_wall_tk = self.params.lp_duct_wall_tk.value

    def build(self) -> Component:
        """
        Build the Lower Port.
        """
        return self.component_tree(
            xz=self.build_xz(),
            xy=None,
            xyz=self.build_xyz(),
        )

    def build_xz(self) -> List[Component]:
        """
        Build the Lower Port in XZ.
        """
        duct_xz = PhysicalComponent(self.DUCT, self.duct_xz_koz)
        apply_component_display_options(duct_xz, color=BLUE_PALETTE["BB"][0])

        return [duct_xz]

    def build_xyz(self) -> List[Component]:
        """
        Build the Lower Port in XYZ.
        """
        return [self._build_duct()]

    def _build_duct(self) -> PhysicalComponent:
        angled_duct_hollow_face = self._hollow_face_from_inner_boundary(
            self.duct_angled_boundary,
            face_thickness=self.duct_wall_tk,
        )
        straight_duct_backwall_face = BluemiraFace(
            offset_wire(self.duct_straight_boundary, self.duct_wall_tk)
        )
        straight_duct_hollow_face = self._hollow_face_from_inner_boundary(
            self.duct_straight_boundary,
            face_thickness=self.duct_wall_tk,
        )

        if self.duct_angle < -45:
            angled_duct_extrude_extent = abs(
                (
                    self.duct_angled_boundary.bounding_box.z_max
                    - (
                        self.duct_straight_boundary.bounding_box.z_max - 1
                    )  # -1 to make sure it goes through
                )
                / np.sin(np.deg2rad(self.duct_angle))
            )
        else:
            angled_duct_extrude_extent = abs(
                (
                    self.duct_angled_boundary.bounding_box.x_min
                    - (
                        self.duct_straight_boundary.bounding_box.x_min + 1
                    )  # +1 to make sure it goes through
                )
                / np.cos(np.deg2rad(self.duct_angle))
            )
        straight_duct_extrude_extent = 20

        duct_heading_x = np.cos(np.deg2rad(self.duct_angle)) * angled_duct_extrude_extent
        duct_heading_z = np.sin(np.deg2rad(self.duct_angle)) * angled_duct_extrude_extent
        angled_duct = extrude_shape(
            angled_duct_hollow_face,
            (duct_heading_x, 0, duct_heading_z),
        )

        straight_duct_backwall = extrude_shape(
            straight_duct_backwall_face,
            straight_duct_backwall_face.normal_at() * self.duct_wall_tk,
        )
        straight_duct_length = extrude_shape(
            straight_duct_hollow_face,
            straight_duct_hollow_face.normal_at() * -straight_duct_extrude_extent,
        )
        straight_duct = boolean_fuse([straight_duct_backwall, straight_duct_length])

        angled_pieces = boolean_cut(angled_duct, [straight_duct])
        angled_top = (
            angled_pieces[0]
            if len(angled_pieces) == 1
            else boolean_cut(angled_duct, [angled_pieces[1]])[0]
        )

        straight_with_hole = boolean_cut(straight_duct, [angled_top])[0]

        duct = boolean_fuse([angled_top, straight_with_hole])

        # rotate duct to correct position
        duct.rotate(degree=np.rad2deg(np.pi / self.n_TF))

        pc = PhysicalComponent(self.DUCT, duct)
        apply_component_display_options(pc, color=BLUE_PALETTE["VV"][0])

        return pc

    @staticmethod
    def _hollow_face_from_inner_boundary(
        inner_boundary: BluemiraWire,
        face_thickness: float,
    ) -> BluemiraFace:
        return boolean_cut(
            BluemiraFace(offset_wire(inner_boundary, face_thickness)),
            [BluemiraFace(inner_boundary)],
        )[0]

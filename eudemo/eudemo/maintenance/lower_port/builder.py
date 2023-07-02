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
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

if TYPE_CHECKING:
    from bluemira.geometry.solid import BluemiraSolid

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, boolean_fuse, extrude_shape, offset_wire
from bluemira.geometry.wire import BluemiraWire
from bluemira.materials import Void


@dataclass
class TSLowerPortDuctBuilderParams(ParameterFrame):
    """Thermal Shield Lower Port Duct Builder Parameters"""

    n_TF: Parameter[int]
    lower_port_angle: Parameter[float]
    tk_ts: Parameter[float]


class TSLowerPortDuctBuilder(Builder):
    """
    Thermal Shield Lower Port Duct Builder
    """

    param_cls = TSLowerPortDuctBuilderParams

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

    def build(self) -> Component:
        """
        Build the Lower Port.
        """
        return self.component_tree(
            xz=self.build_xz(),
            xy=None,
            xyz=self.build_xyz(),
        )

    def build_xz(self) -> List[PhysicalComponent]:
        """
        Build the Lower Port in XZ.
        """
        duct_xz = PhysicalComponent(self.name, self.duct_xz_koz)
        apply_component_display_options(duct_xz, color=BLUE_PALETTE["TS"][0])

        return [duct_xz]

    def build_xyz(self) -> List[PhysicalComponent]:
        """
        Build the Lower Port in XYZ.
        """
        duct, void = build_lower_port_xyz(
            self.duct_angled_boundary,
            self.duct_straight_boundary,
            self.params.n_TF.value,
            self.params.lower_port_angle.value,
            self.params.tk_ts.value,
        )

        pc = PhysicalComponent(self.name, duct)
        void = PhysicalComponent(self.name + " voidspace", void, material=Void("vacuum"))
        apply_component_display_options(pc, color=BLUE_PALETTE["TS"][0])
        apply_component_display_options(void, color=(0, 0, 0))

        return [pc, void]


def _face_and_void_from_outer_boundary(
    outer_boundary: BluemiraWire, thickness: float
) -> Tuple[BluemiraFace]:
    inner_boundary = offset_wire(outer_boundary, -thickness)
    return BluemiraFace([outer_boundary, inner_boundary]), BluemiraFace(inner_boundary)


def build_lower_port_xyz(
    duct_angled_boundary: BluemiraWire,
    duct_straight_boundary: BluemiraWire,
    n_TF: int,
    duct_angle: float,
    wall_tk: float,
) -> Tuple[BluemiraSolid]:
    """
    Build lower port solid geometry, including void (estimate)

    Parameters
    ----------
    duct_angled_boundary:
        Outer x-y boundary wire of the angled port cross-section
    duct_straight_boundary:
        Outer x-y boundary wire of the straight port cross-section

    Returns
    -------
    duct:
        Solid of the lower port duct
    void:
        Solid of the lower port void (estimate)
    """
    straight_duct_extrude_extent = 20
    duct_angle = np.deg2rad(duct_angle)

    angled_duct_face, angled_void_face = _face_and_void_from_outer_boundary(
        duct_angled_boundary, wall_tk
    )
    straight_duct_face, straight_void_face = _face_and_void_from_outer_boundary(
        duct_straight_boundary, wall_tk
    )
    straight_duct_backwall_face = BluemiraFace(
        offset_wire(duct_straight_boundary, wall_tk)
    )

    angled_bb = duct_angled_boundary.bounding_box
    strait_bb = duct_straight_boundary.bounding_box
    if duct_angle < -45:
        # -1 to make sure it goes through
        angled_duct_extrude_extent = abs(
            (angled_bb.z_max - (strait_bb.z_max - 1)) / np.sin(duct_angle)
        )
    else:
        # +1 to make sure it goes through
        angled_duct_extrude_extent = abs(
            (angled_bb.x_min - (strait_bb.x_min + 1)) / np.cos(duct_angle)
        )

    ext_vector = angled_duct_extrude_extent * np.array(
        [np.cos(duct_angle), 0, np.sin(duct_angle)]
    )
    angled_duct = extrude_shape(angled_duct_face, ext_vector)
    angled_void = extrude_shape(angled_void_face, ext_vector)

    straight_duct_backwall = extrude_shape(
        straight_duct_backwall_face,
        straight_duct_backwall_face.normal_at() * wall_tk,
    )

    ext_vector = straight_duct_face.normal_at() * -straight_duct_extrude_extent
    straight_duct_length = extrude_shape(straight_duct_face, ext_vector)
    straight_duct_void = extrude_shape(straight_void_face, ext_vector)

    straight_duct = boolean_fuse([straight_duct_backwall, straight_duct_length])

    angled_pieces = boolean_cut(angled_duct, [straight_duct])
    angled_top = (
        angled_pieces[0]
        if len(angled_pieces) == 1
        else boolean_cut(angled_duct, [angled_pieces[1]])[0]
    )

    straight_with_hole = boolean_cut(straight_duct, [angled_top])[0]

    duct = boolean_fuse([angled_top, straight_with_hole])
    # TODO: Remember that you got lazy here. There can be an overshoot of the
    # angled void
    void = boolean_fuse([angled_void, straight_duct_void])

    # rotate pieces to correct positions
    duct.rotate(degree=180 / n_TF)
    void.rotate(degree=180 / n_TF)
    return duct, void

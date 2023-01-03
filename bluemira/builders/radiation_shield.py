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
Radiation shield builder
"""
from dataclasses import dataclass
from typing import Dict, List, Type, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import build_sectioned_xyz, make_circular_xy_ring
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, boolean_fuse, make_polygon, offset_wire


@dataclass
class RadiationShieldBuilderParams(ParameterFrame):
    """
    Radiation Shield builder parameters
    """

    n_TF: Parameter[int]
    tk_rs: Parameter[float]
    g_cr_rs: Parameter[float]


class RadiationShieldBuilder(Builder):
    """
    Radiation Shield builder
    """

    RS = "RS"
    BODY = "Body"
    param_cls: Type[RadiationShieldBuilderParams] = RadiationShieldBuilderParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
        cryo_vv: BluemiraFace,
    ):
        super().__init__(params, build_config)
        self.cryo_vv = cryo_vv

    def build(self) -> Component:
        """
        Build the radiation shield component.
        """
        rs_xz = self.build_xz()
        rs_face = rs_xz.get_component_properties("shape")

        return self.component_tree(
            xz=[rs_xz],
            xy=[self.build_xy()],
            xyz=self.build_xyz(rs_face),
        )

    def build_xz(self) -> PhysicalComponent:
        """
        Build the x-z components of the radiation shield.
        """
        cryo_vv_rot = self.cryo_vv.deepcopy()
        cryo_vv_rot.rotate(base=(0, 0, 0), direction=(0, 0, 1), degree=180)

        rs_inner = offset_wire(
            boolean_fuse([self.cryo_vv, cryo_vv_rot]).boundary[0],
            self.params.g_cr_rs.value,
        )
        rs_outer = offset_wire(rs_inner, self.params.tk_rs.value)

        # Now we slice in half
        bound_box = rs_outer.bounding_box

        x = np.zeros(4)
        x[2:] = bound_box.x_min - 1.0

        z = np.zeros(4)
        z[[0, -1]] = bound_box.z_min - 1.0
        z[[1, 2]] = bound_box.z_max + 1.0

        cutter = BluemiraFace(make_polygon({"x": x, "y": 0, "z": z}, closed=True))

        shield_body = PhysicalComponent(
            self.BODY, boolean_cut(BluemiraFace([rs_outer, rs_inner]), cutter)[0]
        )
        shield_body.plot_options.face_options["color"] = BLUE_PALETTE[self.RS][0]
        return shield_body

    def build_xy(self) -> PhysicalComponent:
        """
        Build the x-y components of the radiation shield.
        """
        r_in = self.cryo_vv.bounding_box.x_max + self.params.g_cr_rs.value
        r_out = r_in + self.params.tk_rs.value

        shield_body = PhysicalComponent(self.BODY, make_circular_xy_ring(r_in, r_out))
        shield_body.plot_options.face_options["color"] = BLUE_PALETTE[self.RS][0]

        return shield_body

    def build_xyz(
        self, rs_face: BluemiraFace, degree: float = 360.0
    ) -> List[PhysicalComponent]:
        """
        Build the x-y-z components of the radiation shield.
        """
        return build_sectioned_xyz(
            rs_face,
            self.BODY,
            self.params.n_TF.value,
            BLUE_PALETTE[self.RS][0],
            degree,
        )

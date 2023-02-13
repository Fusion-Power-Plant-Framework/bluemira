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
EU-DEMO Equatorial Port
"""
from dataclasses import dataclass
from typing import Dict, List, Type, Union

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import (
    build_sectioned_xy,
    build_sectioned_xyz,
    varied_offset,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import (
    offset_wire,
    make_polygon, 
    slice_shape,
)
from bluemira.geometry.wire import BluemiraWire


class EquatorialPort(ComponentManager):
    """
    Wrapper around a Equatorial Port component tree
    """
    @property
    def xz_boundary(self) -> BluemiraWire:
        """ Returns a wire defining the xz-plane boundary of the Equatorial Port"""
        # TODO: Implement xz_boundary functionality
        pass


@dataclass
class EquatorialPortBuilderParams(ParameterFrame):
    """
    Equatorial Port builder parameters
    """

    n_TF: Parameter[int]
    r_vv_ib_in: Parameter[float]
    r_vv_ob_in: Parameter[float]
    tk_vv_in: Parameter[float]
    tk_vv_out: Parameter[float]
    g_vv_bb: Parameter[float]
    vv_in_off_deg: Parameter[float]
    vv_out_off_deg: Parameter[float]


class EquatorialPortDesigner(Designer):
    """
    Equatorial Port Designer
    """

    param_cls = None

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        divertor_xz: BluemiraFace,
    ):
        super().__init__(params, build_config)
        self.divertor_xz = divertor_xz

    def run(self):
        """Run method of Designer"""

        # TODO return port koz

        # Task 1 create trajectory
        # step 1, what angle is the divertor taken out at
        # step 2, trace path through reactor
        # step 3, return to horizontal
        #         (at what level or just immediately outside reactor)


class EquatorialPortBuilder(Builder):
    """
    Equatorial Port Builder
    """

    VV = "VV"
    BODY = "Body"
    param_cls: Type[EquatorialPortBuilderParams] = EquatorialPortBuilderParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
        ivc_koz: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.ivc_koz = ivc_koz

    def build(self) -> Component:
        """
        Build the vacuum vessel component.
        """
        xz_vv = self.build_xz()
        vv_face = xz_vv.get_component_properties("shape")

        return self.component_tree(
            xz=[xz_vv],
            xy=self.build_xy(vv_face),
            xyz=self.build_xyz(vv_face),
        )

    def build_xz(self, ) -> PhysicalComponent:
        """
        Build the x-z components of the equatorial port
        """
        inner_vv = offset_wire(
            self.ivc_koz, self.params.g_vv_bb.value, join="arc", open_wire=False
        )
        outer_vv = varied_offset(
            inner_vv,
            self.params.tk_vv_in.value,
            self.params.tk_vv_out.value,
            self.params.vv_in_off_deg.value,
            self.params.vv_out_off_deg.value,
            num_points=300,
        )
        face = BluemiraFace([outer_vv, inner_vv])

        body = PhysicalComponent(self.BODY, face)
        body.plot_options.face_options["color"] = BLUE_PALETTE[self.VV][0]

        return body

    def build_xy(self, vv_face: BluemiraFace) -> List[PhysicalComponent]:
        """
        Build the x-y components of the equatorial port
        """
        return build_sectioned_xy(vv_face, BLUE_PALETTE[self.VV][0])

    def build_xyz(self, vv_face: BluemiraFace) -> PhysicalComponent:
        """
        Build the x-y-z components of the equatorial port
        """
        return build_sectioned_xyz(
            vv_face,
            self.BODY,
            self.params.n_TF.value,
            BLUE_PALETTE[self.VV][0],
            degree,
        )
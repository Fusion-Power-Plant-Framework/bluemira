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

    ep_x_inner: Parameter[float]
    ep_x_outer: Parameter[float]
    ep_z_min: Parameter[float]
    ep_z_max: Parameter[float]
    # ep_y_min: Parameter[float]
    # ep_y_max: Parameter[float]
    ep_thickness : Parameter[float] 
    """ May need to expand parameters further for asymmetric port cases """


class EquatorialPortBuilder(Builder):
    """
    Equatorial Port Builder
    """
    BODY = "Equatorial Port"
    param_cls: Type[EquatorialPortBuilderParams] = EquatorialPortBuilderParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
    ):
        super().__init__(params, build_config)

    def build(self) -> Component:
        """
        Build the equatorial port component.
        """
        xz_vv = self.build_xz()
        vv_face = xz_vv.get_component_properties("shape")

        return self.component_tree(
            xz=[xz_vv],
            xy=self.build_xy(vv_face),
            xyz=self.build_xyz(vv_face),
        )

    def build_xz(self) -> PhysicalComponent:
        """
        Build the x-z components of the equatorial port
        """
        # Placeholder magic numbers
        x_min = 1
        x_max = 4
        z_min = 1
        z_max = 4
        thick = 1

        z_bottom_in = x_min + thick
        z_top_in = x_max - thick

        ep_top_x = [x_min, x_max, x_max, x_min]
        ep_top_z = [z_top_in, z_top_in, z_max, z_max]
        ep_bottom_x = [x_min, x_max, x_max, x_min]
        ep_bottom_z = [z_min, z_min, z_bottom_in, z_bottom_in]
        face_top = BluemiraFace(make_polygon({"x": ep_top_x, "y": 0, "z": ep_top_z}, closed=True))
        face_bottom = BluemiraFace(make_polygon({"x": ep_bottom_x, "y": 0, "z": ep_bottom_z}, closed=True))

        body = PhysicalComponent(self.BODY, face_top)
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]

        return body

    def build_xy(self) -> PhysicalComponent:
        """
        Build the x-y components of the equatorial port
        """
        # TODO: Implement build_xy functionality
        #     return build_sectioned_xy(vv_face, BLUE_PALETTE[self.VV][0])
        pass

    def build_xyz(self, vv_face: BluemiraFace) -> PhysicalComponent:
        """
        Build the x-y-z components of the equatorial port
        """
        # TODO: Implement build_xyz functionality
        #     return build_sectioned_xyz(
        #         vv_face,
        #         self.BODY,
        #         self.params.n_TF.value,
        #         BLUE_PALETTE[self.VV][0],
        #         degree,
        #     )
        pass

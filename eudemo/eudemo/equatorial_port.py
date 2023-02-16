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
from numpy import array
from typing import Dict, List, Optional, Type, Union

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
    boolean_fuse,
    circular_pattern,
    make_polygon, 
    offset_wire,
    slice_shape,
)
from bluemira.geometry.wire import BluemiraWire


class EquatorialPort(ComponentManager):
    """
    Wrapper around a Equatorial Port component tree
    """
    
    def xz_boundary(self) -> BluemiraWire:
        """ Returns a wire defining the x-z boundary of the Equatorial Port"""
        # TODO: Implement xz_boundary functionality
        # return (
        #     self.component
        #     .get_component("xz")
        #     .get_component("koz")
        #     .shape.boundary[0]
        # )
        pass


@dataclass
class EquatorialPortDesignerParams(ParameterFrame):
    """
    Equatorial Port Designer parameters
    """

    ep_y_min: Parameter[float]
    ep_y_max: Parameter[float]
    ep_z_min: Parameter[float]
    ep_z_max: Parameter[float]


@dataclass
class EquatorialPortBuilderParams(ParameterFrame):
    """
    Equatorial Port builder parameters
    """

    ep_x_ib: Parameter[float]
    ep_x_ob: Parameter[float]
    ep_z_min: Parameter[float]
    ep_z_max: Parameter[float]
    ep_castellations: Optional[Union(list, array)]
    ep_r_corner: Optional[float]


class EquatorialPortDesigner(Designer):
    """
    Equatorial Port Designer
    """

    param_cls: Type[EquatorialPortDesignerParams] = EquatorialPortDesignerParams
    
    def __init__(
        self, 
        params: Union[Dict, ParameterFrame, EquatorialPortDesignerParams], 
        build_config: Union[Dict, None]
    ):
        super().__init__(params, build_config)

    def run(self, y: list, z: list) -> PhysicalComponent:
        """
        Design the yz in-board profile of the equatorial port
        """
        ep_boundary = BluemiraWire(make_polygon({"x": 0, "y": y, "z": z}, closed=True))
        return BluemiraFace(ep_boundary)


class EquatorialPortBuilder(Builder):
    """
    Equatorial Port Builder
    """
    BODY = "Equatorial Port"
    param_cls: Type[EquatorialPortBuilderParams] = EquatorialPortBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, EquatorialPortBuilderParams],
        build_config: Union[Dict, None],
        ep_face: BluemiraFace
    ):
        super().__init__(params, build_config)
        self.ep_face = ep_face

    def build(self) -> Component:
        """
        Build the equatorial port component.
        """
        x_min = self.params.ep_x_ib.value
        x_max = self.params.ep_x_ob.value

        return self.component_tree(
            xz= self.build_xz(self.ep_face),
            xy= self.build_xy(self.ep_face),
            xyz= self.build_xyz(self.ep_face)
        )

    def build_xz(self, ep_face: BluemiraFace) -> PhysicalComponent:
        """
        Build the x-z components of the equatorial port
        """

        r_rad = self.params.ep_r_corner.value
        n_cst = self.params.ep_castellations.value

        xz_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 0, 1])
        y_axis = (0, 1, 0)

        xz_ib = slice_shape(ep_face, xz_plane)
        xz_wires = [xz_ib]
        x_points, y_points, z_points = castellation(n_cst)
        xz_wires.append(make_polygon({"x": x_points, "y": 0, "z": z_points}))
        xz_profile = BluemiraWire(xz_wires, closed=True)
        face = BluemiraFace(xz_profile)

        body = PhysicalComponent(self.BODY, face)
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]
        return body

    def build_xy(self, ep_face: BluemiraFace) -> PhysicalComponent:
        """
        Build the x-y components of the equatorial port
        """
        # TODO: Implement build_xy functionality
        pass

    def build_xyz(self, ep_face: BluemiraFace) -> PhysicalComponent:
        """
        Build the x-y-z components of the equatorial port
        """
        # TODO: Implement build_xyz functionality
        # shape = extrude_shape(ep_face, dir)
        # body = PhysicalComponent("Equatorial Port", shape)
        # body.display_cad_options.color = BLUE_PALETTE["VV"][0]
        # return body
        pass


def castellation(self, cst_coords) -> (List[float]):
    iter = range(len(cst_coords))
    x = [cst_coords[value][0] for value in iter]
    y_min = [cst_coords[value][1] for value in iter]
    y_max = [cst_coords[value][2] for value in iter]
    z_min = [cst_coords[value][3] for value in iter]
    z_max = [cst_coords[value][4] for value in iter]

    x_points = x + list(reversed(x))
    y_points = y_min + list(reversed(y_max))
    z_points = z_min + list(reversed(z_max))
    return ()
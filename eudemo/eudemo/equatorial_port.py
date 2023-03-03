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
from numpy import ndarray, transpose
from typing import Dict, Type, Union

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_fuse,
    circular_pattern,
    extrude_shape,
    make_polygon,
)
from bluemira.geometry.wire import BluemiraWire


class EquatorialPort(ComponentManager):
    """
    Wrapper around a Equatorial Port component tree
    """

    def xz_boundary(self) -> BluemiraWire:
        """Returns a wire defining the x-z boundary of the Equatorial Port"""
        return (
            self.component.get_component("xz")
            .get_component(EquatorialPortBuilder.KOZ)
            .shape.boundary[0]
        )


@dataclass
class EquatorialPortDesignerParams(ParameterFrame):
    """
    Equatorial Port Designer parameters
    """

    x_ib: Parameter[float]
    x_ob: Parameter[float]
    ep_height: Parameter[float]


@dataclass
class EquatorialPortBuilderParams(ParameterFrame):
    """
    Equatorial Port builder parameters
    """

    n_EP: Parameter[int]
    ep_coordinates: Parameter[ndarray]
    ep_r_corner: Parameter[float]


class EquatorialPortDesigner(Designer):
    """
    Equatorial Port Designer
    """

    param_cls: Type[EquatorialPortDesignerParams] = EquatorialPortDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, EquatorialPortDesignerParams],
        build_config: Union[Dict, None],
    ):
        super().__init__(params, build_config)

    def run(self) -> BluemiraWire:
        """
        Design the xz keep-out zone profile of the equatorial port
        """

        x_i = self.params.x_ib.value
        x_o = self.params.x_ob.value
        z_h = self.params.ep_height.value

        x = (x_i, x_o, x_o, x_i)
        z = (-z_h / 2, -z_h / 2, z_h / 2, z_h / 2)

        ep_boundary = BluemiraFace(
            make_polygon({"x": x, "y": 0, "z": z}, label="koz", closed=True)
        )
        return ep_boundary


class EquatorialPortBuilder(Builder):
    """
    Equatorial Port Builder
    """

    NAME = "Equatorial Port"
    KOZ = "koz"
    param_cls: Type[EquatorialPortBuilderParams] = EquatorialPortBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, EquatorialPortBuilderParams],
        build_config: Union[Dict, None],
        xz_profile: BluemiraFace,
    ):
        super().__init__(params, build_config)
        self.xz_profile = xz_profile

    def build(self) -> Component:
        """
        Build the equatorial port component.
        """

        # TODO: Integrate corner radii
        self.r_rad = self.params.ep_r_corner.value
        self.coords = self.params.ep_coordinates.value
        self.n_ep = self.params.n_EP.value

        self.x_points, self.y_points, self.z_points = castellation(self.coords)

        return self.component_tree(
            xz=self.build_xz(self.xz_profile),
            xy=self.build_xy(self.x_points, self.y_points, self.n_ep),
            xyz=self.build_xyz(self.coords, self.n_ep),
        )

    def build_xz(self, xz_profile: BluemiraWire) -> PhysicalComponent:
        """
        Build the x-z components of the equatorial port
        """

        body = PhysicalComponent(self.KOZ, xz_profile)
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]
        return body

    def build_xy(self, x: list, y: list, n_ep: int = 10) -> PhysicalComponent:
        """
        Build the x-y components of the equatorial port
        """

        ep_boundary = BluemiraWire(make_polygon({"x": x, "y": y, "z": 0}, closed=True))
        body = PhysicalComponent(self.NAME, BluemiraFace(ep_boundary))
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]
        return circular_pattern(body, n_shapes=n_ep)

    def build_xyz(
        self, coords: Union[list, ndarray], n_ep: int = 10
    ) -> PhysicalComponent:
        """
        Build the x-y-z components of the equatorial port
        """
        sections = []
        x_ob = coords[0][-1]  # Outboard x = the last listed x-value
        new_coords = transpose(coords)

        for r in new_coords:
            x = r[0]
            y = [r[1], r[1], r[2], r[2]]
            z = [r[3], r[4], r[4], r[3]]
            ext_vec = (x_ob - x, 0, 0)
            section = BluemiraFace(make_polygon({"x": x, "y": y, "z": z}, closed=True))
            sections.append(extrude_shape(section, ext_vec))

        ep_shape = boolean_fuse(sections)
        return circular_pattern(ep_shape, n_shapes=n_ep)


def castellation(cst_coords: list):
    iter = range(len(cst_coords))
    x = [cst_coords[value][0] for value in iter]
    y_min = [cst_coords[value][1] for value in iter]
    y_max = [cst_coords[value][2] for value in iter]
    z_min = [cst_coords[value][3] for value in iter]
    z_max = [cst_coords[value][4] for value in iter]

    x_points = x + list(reversed(x))
    y_points = y_min + list(reversed(y_max))
    z_points = z_min + list(reversed(z_max))
    return x_points, y_points, z_points

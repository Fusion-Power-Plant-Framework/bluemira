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

from numpy import array, ndarray

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import circular_pattern_component
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.plane import BluemiraFace, BluemiraPlane
from bluemira.geometry.tools import (
    boolean_fuse,
    extrude_shape,
    make_polygon,
    slice_shape,
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

    ep_height: Parameter[float]


@dataclass
class EquatorialPortBuilderParams(ParameterFrame):
    """
    Equatorial Port builder parameters
    """

    n_ep: Parameter[int]
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
        x_ib: float,
        x_ob: float,
    ):
        super().__init__(params, build_config)
        self.x_ib = x_ib
        self.x_ob = x_ob

    def run(self) -> BluemiraWire:
        """
        Design the xz keep-out zone profile of the equatorial port
        """
        x_ib = self.x_ib
        x_ob = self.x_ob
        z_h = self.params.ep_height.value

        x = (x_ib, x_ob, x_ob, x_ib)
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
    param_cls: Type[EquatorialPortBuilderParams] = EquatorialPortBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, EquatorialPortBuilderParams],
        build_config: Union[Dict, None],
        xz_profile: BluemiraFace,
        y_width: float,
        x_offsets: list,
        castellation_offsets: list,
    ):
        super().__init__(params, build_config)
        self.xz_profile = xz_profile
        self.y_width = y_width
        self.x_off = x_offsets
        self.cst = castellation_offsets

    def build(self) -> Component:
        """
        Build the equatorial port component.
        """
        # TODO: Implement corner radii
        self.r_rad = self.params.ep_r_corner.value
        self.n_ep = self.params.n_ep.value

        # Set-up coordinates for geometry building
        xz_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 0, 1])
        intersections = slice_shape(self.xz_profile.boundary[0], xz_plane)
        self.x_ib = min(intersections[:, 0])
        self.x_ob = max(intersections[:, 0])
        self.z_lo = min(intersections[:, 2])
        self.z_hi = max(intersections[:, 2])

        x_points, y_points, z_points, xyz_coords = self.build_coordinates(
            self.x_off, self.cst
        )

        return self.component_tree(
            xz=[self.build_xz(x_points, z_points)],
            xy=self.build_xy(x_points, y_points),
            xyz=self.build_xyz(xyz_coords, n_ep=10),
        )

    def build_xz(self, x: list, z: list) -> PhysicalComponent:
        """
        Build the xz components of the equatorial port
        """
        xz_profile = BluemiraWire(make_polygon({"x": x, "y": 0, "z": z}, closed=True))
        body = PhysicalComponent("koz", xz_profile)
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]
        return body

    def build_xy(self, x: list, y: list, n_ep: int = 10) -> List[PhysicalComponent]:
        """
        Build the xy components of the equatorial port
        """
        ep_xy_profile = BluemiraWire(make_polygon({"x": x, "y": y, "z": 0}, closed=True))
        body = PhysicalComponent(self.NAME, BluemiraFace(ep_xy_profile))
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]
        return circular_pattern_component(body, n_children=n_ep)

    def build_xyz(self, coords: ndarray, n_ep: int = 10) -> List[PhysicalComponent]:
        """
        Build the xyz components of the equatorial port
        """
        sections = []
        coords = coords.transpose()[:-1]

        for r in coords:
            x = r[0]
            y = [r[1], r[1], r[2], r[2]]
            z = [r[3], r[4], r[4], r[3]]
            ext_vec = (self.x_ob - x, 0, 0)
            section = BluemiraFace(make_polygon({"x": x, "y": y, "z": z}, closed=True))
            sections.append(extrude_shape(section, ext_vec))

        ep_shape = PhysicalComponent(self.NAME, boolean_fuse(sections))
        ep_shape.display_cad_options.color = BLUE_PALETTE["VV"][0]
        return circular_pattern_component(ep_shape, n_children=n_ep)

    def build_coordinates(self, x_offs: list, c_offs: list):
        """
        Returns coordinate lists for building the Eq. Port Bluemira geometry

        Parameters
        ----------
        x_offsets: list
            list of x positions of castellations
        castellation_offsets: list
            list of castellations offsets
        """
        offsets = [0, 0] + [z for i in zip(c_offs, c_offs) for z in i]
        x_offsets = [x for i in zip(x_offs, x_offs) for x in i]

        x_outwards = (
            [self.x_ib] + [self.x_ib + x_offset for x_offset in x_offsets] + [self.x_ob]
        )
        x_points = x_outwards + list(reversed(x_outwards))

        y_pos_half = [self.y_width / 2 + offset for offset in offsets]
        y_neg_half = [-y for y in y_pos_half]
        y_points = y_pos_half + list(reversed(y_neg_half))

        z_upper_half = [self.z_hi + offset for offset in offsets]
        z_lower_half = [self.z_lo - offset for offset in offsets]
        z_points = z_upper_half + list(reversed(z_lower_half))

        xyz_coords = array(
            [x_outwards, y_pos_half, y_neg_half, z_upper_half, z_lower_half], float
        )
        return x_points, y_points, z_points, xyz_coords

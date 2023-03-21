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
from bluemira.geometry.solid import BluemiraSolid, BluemiraWire
from bluemira.geometry.tools import (
    boolean_fuse,
    extrude_shape,
    make_polygon,
    offset_wire,
    slice_shape,
)


class EquatorialPort(ComponentManager):
    """
    Wrapper around a Equatorial Port component tree
    """

    def xz_boundary(self) -> BluemiraWire:
        """Returns a wire defining the x-z boundary of the Equatorial Port"""
        return (
            self.component.get_component("xz")
            .get_component(EquatorialPortBuilder.NAME)
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
        koz_z_offset: float,
        x_ib: float,
        x_ob: float,
    ):
        super().__init__(params, build_config)
        self.koz_offset = koz_z_offset
        self.x_ib = x_ib
        self.x_ob = x_ob

    def run(self) -> BluemiraWire:
        """
        Design the xz keep-out zone profile of the equatorial port
        """
        x_ib = self.x_ib
        x_ob = self.x_ob
        z_h = self.params.ep_height.value + (2 * self.koz_offset)

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
        x_ib: float,
        x_ob: float,
        yz_profile: BluemiraWire,
        extrude_vec: tuple,
        x_offsets: list,
        castellation_offsets: list,
    ):
        super().__init__(params, build_config)
        self.x_ib = x_ib
        self.x_ob = x_ob
        self.yz_profile = yz_profile
        self.vec = extrude_vec
        self.x_off = x_offsets
        self.cst = castellation_offsets

    def build(self) -> Component:
        """
        Build the equatorial port component.
        """
        # TODO: Implement corner radii using PR #1992
        self.r_rad = self.params.ep_r_corner.value
        self.n_ep = self.params.n_ep.value

        xyz_solid = self.build_castellations(
            self.yz_profile, self.vec, self.x_ob - self.x_ib, self.x_off, self.cst
        )
        xz_plane = BluemiraPlane(axis=(0, 1, 0))
        xy_plane = BluemiraPlane(axis=(0, 0, 1))
        xz_slice = BluemiraFace(slice_shape(xyz_solid, xz_plane))
        xy_slice = BluemiraFace(slice_shape(xyz_solid, xy_plane))

        return self.component_tree(
            xz=[self.build_xz(xz_slice)],
            xy=self.build_xy(xy_slice),
            xyz=self.build_xyz(xyz_solid, n_ep=10),
        )

    def build_xz(self, xz: BluemiraFace) -> PhysicalComponent:
        """
        Build the xz components of the equatorial port
        """
        body = PhysicalComponent(self.NAME, xz)
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]
        return body

    def build_xy(self, xy: BluemiraFace, n_ep: int = 10) -> List[PhysicalComponent]:
        """
        Build the xy components of the equatorial port
        """
        body = PhysicalComponent(self.NAME, xy)
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]
        return circular_pattern_component(body, n_children=n_ep)

    def build_xyz(self, xyz: BluemiraSolid, n_ep: int = 10) -> List[PhysicalComponent]:
        """
        Build the xyz representation of the equatorial port
        """
        ep_shape = PhysicalComponent(self.NAME, xyz)
        ep_shape.display_cad_options.color = BLUE_PALETTE["VV"][0]
        return circular_pattern_component(ep_shape, n_children=n_ep)

    def build_castellations(
        self,
        wire: BluemiraFace,
        vec: tuple,
        length: float,
        distances: List[float],
        offsets: List[float],
    ) -> BluemiraSolid:
        """
        Returns BluemiraSolid that is a BluemiraFace castellated along a given vector

        Parameters
        ----------
        wire: BluemiraWire
            profile boundary to be castellated
        vec: tuple (x,y,z)
            unit vector along which to extrude
        length: float
            total length of castellated BluemiraSolid in vec direction
        distances: List[float]
            list of distances along vec describing castellation positions (relative to starting wire)
        offsets: List[float]
            list of castellations offsets corresponding to each position in 'distances' parameter
        """
        base = wire
        sections = []

        # Normalise and Scale vec
        vec_mag = (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5
        if vec_mag != 1.0:
            vec = (vec[0] / vec_mag, vec[1] / vec_mag, vec[2] / vec_mag)
        print("Extrude by ", vec)
        make_vec = lambda scale: (vec[0] * scale, vec[1] * scale, vec[2] * scale)

        parameter_array = list(zip(distances, offsets))
        parameter_array.append((length, 1.0))
        _prev_dist = 0
        for dist_off in parameter_array:
            dist = dist_off[0] - _prev_dist
            ext = make_vec(dist)
            part = extrude_shape(base, ext)
            sections.append(part)
            _prev_dist = dist_off[0]
            x = [i + ext[0] for i in base.vertexes.x]
            y = [i + ext[1] for i in base.vertexes.y]
            z = [i + ext[2] for i in base.vertexes.z]
            base = BluemiraWire(make_polygon([x, y, z], closed=True))
            base = offset_wire(base, dist_off[1])
            base = BluemiraFace(base)
        return boolean_fuse(sections)

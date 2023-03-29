# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short, I. Chiang
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

import numpy as np

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
            .get_component(CastellationBuilder.NAME)
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
    Castellation Builder parameters
    """

    ep_height: Parameter[float]
    cst_r_corner: Parameter[float]


@dataclass
class CastellationBuilderParams(ParameterFrame):
    """
    Castellation Builder parameters
    """

    n_components: Parameter[int]
    cst_r_corner: Parameter[float]


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
        outer_profile: BluemiraWire,
        length: float,
        wall_thickness: float,
    ):
        super().__init__(params, build_config)
        self.outer = outer_profile
        self.length = length
        self.offset = float(0 - wall_thickness)

    def build(self) -> Component:
        """Build the Equatorial Port"""
        self.z_h = self.params.ep_height.value
        self.r_rad = self.params.cst_r_corner.value
        hole = offset_wire(self.outer.deepcopy(), self.offset)
        self.profile = BluemiraFace([self.outer, hole])
        self.port = extrude_shape(self.profile, (self.length, 0, 0))

        return self.component_tree(
            xz=[self.build_xz()],
            xy=self.build_xy(),
            xyz=self.build_xyz(),
        )

    def build_xz(self) -> PhysicalComponent:
        """
        Build the xy representation of the Equatorial Port
        """
        port = slice_shape(
            extrude_shape(BluemiraFace(self.outer), (self.length, 0, 0)),
            BluemiraPlane(axis=(0, 1, 0)),
        )
        body = PhysicalComponent(self.NAME, BluemiraFace(port))
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]
        return body

    def build_xy(self, n: int = 10) -> PhysicalComponent:
        """
        Build the cross-sectional representation of the Equatorial Port
        """
        body = PhysicalComponent(self.NAME, self.profile)
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]
        return circular_pattern_component(body, n_children=n)

    def build_xyz(self, n: int = 10) -> PhysicalComponent:
        """
        Build the 3D representation of the Equatorial Port
        """
        body = PhysicalComponent(self.NAME, self.port)
        body.display_cad_options.color = BLUE_PALETTE["VV"][0]
        return circular_pattern_component(body, n_children=n)


class CastellationBuilder(Builder):
    """
    Castellation Builder
    """

    NAME = "Castellation"
    param_cls: Type[CastellationBuilderParams] = CastellationBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, CastellationBuilderParams],
        build_config: Union[Dict, None],
        length: float,
        start_profile: BluemiraFace,
        extrude_vec: tuple,
        offsets: list,
        castellation_offsets: list,
    ):
        super().__init__(params, build_config)
        self.length = length
        self.face = start_profile
        self.vec = np.array(extrude_vec)
        self.off = offsets
        self.cst = castellation_offsets

    def build(self) -> Component:
        """
        Build the castellated component.
        """
        # TODO: Implement corner radii using PR #1992
        self.r_rad = self.params.cst_r_corner.value
        self.n = self.params.n_components.value

        xyz_solid = self.build_castellations(
            self.face, self.vec, self.length, self.off, self.cst
        )
        xz_plane = BluemiraPlane(axis=(0, 1, 0))
        xy_plane = BluemiraPlane(axis=(0, 0, 1))
        xz_slice = BluemiraFace(slice_shape(xyz_solid, xz_plane))
        xy_slice = BluemiraFace(slice_shape(xyz_solid, xy_plane))

        return self.component_tree(
            xz=[self.build_xz(xz_slice)],
            xy=self.build_xy(xy_slice),
            xyz=self.build_xyz(xyz_solid),
        )

    def build_xz(self, xz: BluemiraFace) -> PhysicalComponent:
        """
        Build the xz representation of the castellated component
        """
        body = PhysicalComponent(self.NAME, xz)
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]
        return body

    def build_xy(self, xy: BluemiraFace, n: int = 10) -> List[PhysicalComponent]:
        """
        Build the xy representation of the castellated component
        """
        body = PhysicalComponent(self.NAME, xy)
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]
        return circular_pattern_component(body, n_children=n)

    def build_xyz(self, xyz: BluemiraSolid, n: int = 10) -> List[PhysicalComponent]:
        """
        Build the xyz representation of the castellated component
        """
        cst_shape = PhysicalComponent(self.NAME, xyz)
        cst_shape.display_cad_options.color = BLUE_PALETTE["VV"][0]
        return circular_pattern_component(cst_shape, n_children=n)

    def build_castellations(
        self,
        face: BluemiraFace,
        vec: tuple,
        length: float,
        distances: List[float],
        offsets: List[float],
    ) -> BluemiraSolid:
        """
        Returns BluemiraSolid for a BluemiraFace castellated along a given vector

        Parameters
        ----------
        wire: BluemiraWire
            profile boundary to be castellated
        vec: tuple (x,y,z)
            unit vector along which to extrude
        length: float
            total length of castellated BluemiraSolid in vec direction
        distances: List[float]
            castellation positions along vec (relative to starting wire)
        offsets: List[float]
            castellations offsets for each position
        """
        base = face
        sections = []

        # Normalise vec
        vec_mag = np.linalg.norm(vec)
        if vec_mag != 1.0:
            vec /= vec_mag

        parameter_array = list(zip(distances, offsets))
        parameter_array.append((length, 1.0))
        _prev_dist = 0
        for dist, offset in parameter_array:
            ext_vec = np.array(vec) * (dist - _prev_dist)
            sections.append(extrude_shape(base, ext_vec))
            base.translate(ext_vec)
            base = BluemiraFace(offset_wire(base.deepcopy(), offset))
            _prev_dist = dist
        return boolean_fuse(sections)

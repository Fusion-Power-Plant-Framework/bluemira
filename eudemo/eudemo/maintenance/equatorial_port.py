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
from typing import Dict, Iterable, List, Optional, Type, Union

import numpy as np

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import (
    apply_component_display_options,
    circular_pattern_component,
)
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
            .get_component(EquatorialPortDuctBuilder.NAME)
            .shape.boundary[0]
        )


@dataclass
class EquatorialPortKOZDesignerParams(ParameterFrame):
    """
    Equatorial Port Designer parameters
    """

    ep_height: Parameter[float]


class EquatorialPortKOZDesigner(Designer):
    """
    Equatorial Port Keep-out Zone Designer
    - Builds a rectangular horizontal keep-out zone
    offset out from the equatorial port x-z profile
    """

    param_cls: Type[EquatorialPortKOZDesignerParams] = EquatorialPortKOZDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, EquatorialPortKOZDesignerParams],
        build_config: Union[Dict, None],
        koz_z_offset: float,
        x_ib: float,
        x_ob: float,
        z_pos: float = 0.0,
    ):
        """
        Parameters:
        -----------
        params: Union[Dict, ParameterFrame, EquatorialPortKOZDesignerParams]
        build_config: Union[Dict, None]
        koz_z_offset: float
            offset distance for the KOZ around the equatorial port
        x_ib: float
            in-board x-position of the KOZ
        x_ob: float
            out-board x-position of the KOZ
        z_pos: float
            z-positional height of the KOZ x-y midplane, default: 0.0
        """
        super().__init__(params, build_config)
        self.koz_offset = koz_z_offset
        self.x_ib = x_ib
        self.x_ob = x_ob
        self.z_pos = z_pos

    def run(self) -> BluemiraWire:
        """
        Design the xz keep-out zone profile of the equatorial port
        """
        z_h = (self.params.ep_height.value / 2.0) + self.koz_offset
        z_o = self.z_pos

        x = (self.x_ib, self.x_ob, self.x_ob, self.x_ib)
        z = (z_o - z_h, z_o - z_h, z_o + z_h, z_o + z_h)

        ep_boundary = BluemiraFace(
            make_polygon({"x": x, "y": 0, "z": z}, closed=True),
            label="equatorial_port_koz",
        )
        return ep_boundary


@dataclass
class EquatorialPortDuctBuilderParams(ParameterFrame):
    """
    Castellation Builder parameters
    """

    ep_height: Parameter[float]
    cst_r_corner: Parameter[float]


class EquatorialPortDuctBuilder(Builder):
    """
    Equatorial Port Duct Builder
    """

    NAME = "Equatorial Port Duct"
    param_cls: Type[EquatorialPortDuctBuilderParams] = EquatorialPortDuctBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, EquatorialPortDuctBuilderParams],
        build_config: Union[Dict, None],
        outer_profile: BluemiraWire,
        length: float,
        equatorial_port_wall_thickness: float,
    ):
        super().__init__(params, build_config)
        self.outer = outer_profile
        self.length = length
        self.offset = equatorial_port_wall_thickness

    def build(self) -> Component:
        """Build the Equatorial Port"""
        self.z_h = self.params.ep_height.value
        self.r_rad = self.params.cst_r_corner.value
        hole = offset_wire(self.outer, -self.offset)
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
        apply_component_display_options(body, BLUE_PALETTE["VV"][0])
        return body

    def build_xy(self, n: int = 10) -> PhysicalComponent:
        """
        Build the cross-sectional representation of the Equatorial Port
        """
        body = PhysicalComponent(self.NAME, self.profile)
        apply_component_display_options(body, BLUE_PALETTE["VV"][0])
        return circular_pattern_component(body, n_children=n)

    def build_xyz(self, n: int = 10) -> PhysicalComponent:
        """
        Build the 3D representation of the Equatorial Port
        """
        body = PhysicalComponent(self.NAME, self.port)
        apply_component_display_options(body, BLUE_PALETTE["VV"][0])
        return circular_pattern_component(body, n_children=n)


@dataclass
class CastellationBuilderParams(ParameterFrame):
    """
    Castellation Builder parameters
    """

    n_components: Parameter[int]
    cst_r_corner: Parameter[float]


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
        depth: float,
        start_profile: BluemiraFace,
        extrude_direction: Iterable,
        offsets: Union[float, Iterable],
        depth_offsets: Union[float, Iterable],
        n_castellation: Optional[int] = None,
    ):
        super().__init__(params, build_config)
        self.length = depth
        self.face = start_profile
        self.vec = np.array(extrude_direction)
        self.off = offsets
        self.cst = depth_offsets
        self.n_cst = n_castellation

    def build(self) -> Component:
        """
        Build the castellated component.
        """
        # TODO: Implement corner radii using PR #1992
        self.r_rad = self.params.cst_r_corner.value
        self.n = self.params.n_components.value

        xyz_solid = self.build_castellations(
            self.face, self.vec, self.length, self.off, self.cst, self.n_cst
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
        apply_component_display_options(body, BLUE_PALETTE["VV"][0])
        return body

    def build_xy(self, xy: BluemiraFace, n: int = 10) -> List[PhysicalComponent]:
        """
        Build the xy representation of the castellated component
        """
        body = PhysicalComponent(self.NAME, xy)
        apply_component_display_options(body, BLUE_PALETTE["VV"][0])
        return circular_pattern_component(body, n_children=n)

    def build_xyz(self, xyz: BluemiraSolid, n: int = 10) -> List[PhysicalComponent]:
        """
        Build the xyz representation of the castellated component
        """
        cst_shape = PhysicalComponent(self.NAME, xyz)
        apply_component_display_options(cst_shape, BLUE_PALETTE["VV"][0])
        return circular_pattern_component(cst_shape, n_children=n)

    def build_castellations(
        self,
        face: BluemiraFace,
        vec: tuple,
        length: float,
        distances: Optional[Union[float, Iterable]],
        offsets: Union[float, Iterable],
        num_castellation: Optional[int],
    ) -> BluemiraSolid:
        """
        Returns BluemiraSolid for a BluemiraFace castellated along a given vector

        Parameters
        ----------
        face: BluemiraFace
            starting profile to be castellated
        vec: tuple (x,y,z)
            unit vector along which to extrude
        length: float
            total length of castellated BluemiraSolid in vec direction
        distances: Optional[Union[float, Iterable]]
            (optional) parameter for manually spaced castellations
        offsets: Union[float, Iterable]
            castellations offsets for each position
        num_castellation: Optional[int]
            (optional) parameter for equally spaced castellations
        """
        base = face
        sections = []

        # Normalise vec
        vec_mag = np.linalg.norm(vec)
        if vec_mag != 1.0:
            vec /= vec_mag

        if num_castellation is not (None or 0):
            interval = length / (num_castellation + 1)
            offset_iterable = [interval * i for i in range(num_castellation + 1)]
        else:
            offset_iterable = offsets
        parameter_array = list(zip(distances, offset_iterable))
        parameter_array.append((length, 0.0))
        _prev_dist = 0
        for dist, offset in parameter_array:
            ext_vec = np.array(vec) * (dist - _prev_dist)
            sections.append(extrude_shape(base, ext_vec))
            base.translate(ext_vec)
            base = BluemiraFace(offset_wire(BluemiraWire(base.wires), offset))
            _prev_dist = dist
        return boolean_fuse(sections)

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
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import apply_component_display_options
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
        params:
            Parameters for the equatorial port designer
        build_config:
            Build config for the equatorial port designer
        koz_z_offset:
            offset distance for the KOZ around the equatorial port
        x_ib:
            in-board x-position of the KOZ
        x_ob:
            out-board x-position of the KOZ
        z_pos:
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
            xy=[self.build_xy()],
            xyz=[self.build_xyz()],
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

    def build_xy(self) -> PhysicalComponent:
        """
        Build the cross-sectional representation of the Equatorial Port
        """
        body = PhysicalComponent(self.NAME, self.profile)
        apply_component_display_options(body, BLUE_PALETTE["VV"][0])
        return body

    def build_xyz(self) -> PhysicalComponent:
        """
        Build the 3D representation of the Equatorial Port
        """
        body = PhysicalComponent(self.NAME, self.port)
        apply_component_display_options(body, BLUE_PALETTE["VV"][0])
        return body


@dataclass
class CastellationBuilderParams(ParameterFrame):
    """
    Castellation Builder parameters
    """

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
        depth_offsets: Optional[Union[float, Iterable]] = None,
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

        xyz_solid = self.build_castellations(
            self.face, self.vec, self.length, self.off, self.cst, self.n_cst
        )
        xz_plane = BluemiraPlane(axis=(0, 1, 0))
        xy_plane = BluemiraPlane(axis=(0, 0, 1))
        xz_slice = BluemiraFace(slice_shape(xyz_solid, xz_plane))
        xy_slice = BluemiraFace(slice_shape(xyz_solid, xy_plane))

        return self.component_tree(
            xz=[self.build_xz(xz_slice)],
            xy=[self.build_xy(xy_slice)],
            xyz=[self.build_xyz(xyz_solid)],
        )

    def build_xz(self, xz: BluemiraFace) -> PhysicalComponent:
        """
        Build the xz representation of the castellated component
        """
        body = PhysicalComponent(self.NAME, xz)
        apply_component_display_options(body, BLUE_PALETTE["VV"][0])
        return body

    def build_xy(self, xy: BluemiraFace) -> List[PhysicalComponent]:
        """
        Build the xy representation of the castellated component
        """
        body = PhysicalComponent(self.NAME, xy)
        apply_component_display_options(body, BLUE_PALETTE["VV"][0])
        return body

    def build_xyz(self, xyz: BluemiraSolid) -> List[PhysicalComponent]:
        """
        Build the xyz representation of the castellated component
        """
        cst_shape = PhysicalComponent(self.NAME, xyz)
        apply_component_display_options(cst_shape, BLUE_PALETTE["VV"][0])
        return cst_shape

    def build_castellations(
        self,
        face: BluemiraFace,
        vec: Tuple[float, float, float],
        length: float,
        offsets: Union[float, Iterable],
        distances: Optional[Iterable] = None,
        n_cast: Optional[int] = None,
    ) -> BluemiraSolid:
        """
        Returns BluemiraSolid for a BluemiraFace castellated along a given vector

        Parameters
        ----------
        face:
            starting profile to be castellated
        vec:
            unit vector along which to extrude
        length:
            total length of castellated BluemiraSolid in vec direction
        offsets:
            castellations offset(s) for each position
        distances:
            (optional) parameter for manually spaced castellations
        n_cast:
            (optional) parameter for equally spaced castellations
        """
        base = face
        sections = []

        # Normalise vec
        vec_mag = np.linalg.norm(vec)
        if vec_mag != 1.0:
            vec /= vec_mag

        # Check/Set-up distances iterable
        if not ((n_cast is None) or (n_cast == 0)):
            interval = length / (n_cast + 1)
            dist_iter = [interval * i for i in range(1, n_cast + 1)]
        else:
            if distances is not None:
                dist_iter = distances
            else:
                raise ValueError("Both distance and n_cast parameters are None")

        # Check/Set-up offsets iterable
        if type(offsets) == float:
            off_iter = [offsets] * len(dist_iter)
        else:
            if len(offsets) == len(dist_iter):
                off_iter = offsets
            else:
                raise ValueError("Length of offsets doesn't match distances/n_cast")

        parameter_array = list(zip(dist_iter, off_iter))
        parameter_array.append((length, 0.0))
        _prev_dist = 0
        for dist, off in parameter_array:
            ext_vec = np.array(vec) * (dist - _prev_dist)
            sections.append(extrude_shape(base, ext_vec))
            base.translate(ext_vec)
            base = BluemiraFace(offset_wire(BluemiraWire(base.wires), off))
            _prev_dist = dist
        return boolean_fuse(sections)

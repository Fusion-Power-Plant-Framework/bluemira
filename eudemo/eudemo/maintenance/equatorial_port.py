# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
EU-DEMO Equatorial Port
"""

from dataclasses import dataclass

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.reactor import ComponentManager
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.solid import BluemiraWire
from bluemira.geometry.tools import extrude_shape, make_polygon, offset_wire, slice_shape


class EquatorialPort(ComponentManager):
    """
    Wrapper around a Equatorial Port component tree
    """

    def xz_boundary(self) -> BluemiraWire:
        """
        Returns
        -------
        :
            A wire defining the x-z boundary of the Equatorial Port
        """
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

    R_0: Parameter[float]
    """Gap between VV and TS"""
    g_vv_ts: Parameter[float]
    """TS thickness"""
    tk_ts: Parameter[float]
    """Gap between TS and TF (used for short gap to PF)"""
    g_ts_tf: Parameter[float]
    """Gap between PF coil and support"""
    pf_s_g: Parameter[float]
    """PF coil support thickness"""
    pf_s_tk_plate: Parameter[float]
    tk_vv_single_wall: Parameter[float]

    ep_z_position: Parameter[float]
    ep_height: Parameter[float]


class EquatorialPortKOZDesigner(Designer):
    """
    Equatorial Port Keep-out Zone Designer
    - Builds a rectangular horizontal keep-out zone
    offset out from the equatorial port x-z profile
    """

    param_cls: type[EquatorialPortKOZDesignerParams] = EquatorialPortKOZDesignerParams

    def __init__(
        self,
        params: dict | ParameterFrame | EquatorialPortKOZDesignerParams,
        build_config: dict | None,
        x_ob: float,
    ):
        """
        Parameters
        ----------
        params:
            Parameters for the equatorial port designer
        build_config:
            Build config for the equatorial port designer
        x_ob:
            out-board x-position of the KOZ
        """
        super().__init__(params, build_config)
        self.koz_offset = (
            self.params.tk_vv_single_wall.value
            + self.params.g_vv_ts.value
            + self.params.tk_ts.value
            + self.params.g_ts_tf.value
            + self.params.pf_s_tk_plate.value
            + self.params.pf_s_g.value
        )
        self.x_ib = self.params.R_0.value
        self.x_ob = x_ob
        self.z_pos = self.params.ep_z_position.value

    def run(self) -> BluemiraWire:
        """
        Design the xz keep-out zone profile of the equatorial port

        Returns
        -------
        :
            The xz keep out zone
        """
        z_h = 0.5 * self.params.ep_height.value + self.koz_offset
        z_o = self.z_pos

        x = (self.x_ib, self.x_ob, self.x_ob, self.x_ib)
        z = (z_o - z_h, z_o - z_h, z_o + z_h, z_o + z_h)

        return BluemiraFace(
            make_polygon({"x": x, "y": 0, "z": z}, closed=True),
            label="equatorial_port_koz",
        )


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
    param_cls: type[EquatorialPortDuctBuilderParams] = EquatorialPortDuctBuilderParams

    def __init__(
        self,
        params: dict | ParameterFrame | EquatorialPortDuctBuilderParams,
        build_config: dict | None,
        outer_profile: BluemiraWire,
        length: float,
        equatorial_port_wall_thickness: float,
    ):
        super().__init__(params, build_config)
        self.outer = outer_profile
        self.length = length
        self.offset = equatorial_port_wall_thickness

    def build(self) -> Component:
        """
        Build the Equatorial Port

        Returns
        -------
        :
            The equatorial port component tree
        """
        self.z_h = self.params.ep_height.value
        self.r_rad = self.params.cst_r_corner.value
        hole = offset_wire(self.outer, -self.offset)
        self.profile = BluemiraFace([self.outer, hole])
        self.port = extrude_shape(self.profile, (self.length, 0, 0))

        return self.component_tree(
            xz=[self.build_xz()], xy=[self.build_xy()], xyz=[self.build_xyz()]
        )

    def build_xz(self) -> PhysicalComponent:
        """
        Build the xz representation of the Equatorial Port

        Returns
        -------
        :
            The xz of the equatorial port
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

        Returns
        -------
        :
            The xy of the equatorial port
        """
        body = PhysicalComponent(self.NAME, self.profile)
        apply_component_display_options(body, BLUE_PALETTE["VV"][0])
        return body

    def build_xyz(self) -> PhysicalComponent:
        """
        Build the 3D representation of the Equatorial Port

        Returns
        -------
        :
            The xyz of the equatorial port
        """
        body = PhysicalComponent(self.NAME, self.port)
        apply_component_display_options(body, BLUE_PALETTE["VV"][0])
        return body

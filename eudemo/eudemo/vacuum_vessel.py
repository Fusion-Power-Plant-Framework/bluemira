# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Builder for making a parameterised EU-DEMO vacuum vessel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.reactor import ComponentManager
from bluemira.builders.tools import (
    apply_component_display_options,
    build_sectioned_xy,
    build_sectioned_xyz,
    varied_offset,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    _offset_wire_discretised,  # noqa: PLC2701
    boolean_fuse,
    force_wire_to_spline,
)
from bluemira.materials.basic import Void
from eudemo.comp_managers import PortManagerMixin
from eudemo.maintenance.duct_connection import pipe_pipe_join

if TYPE_CHECKING:
    from bluemira.geometry.wire import BluemiraWire


class VacuumVessel(PortManagerMixin, ComponentManager):
    """
    Wrapper around a Vacuum Vessel component tree.
    """

    @property
    def xz_boundary(self) -> BluemiraWire:
        """
        Returns
        -------
        :
            A wire giving the vessel's boundary in the xz plane.

        """
        return (
            self.component()
            .get_component("xz")
            .get_component(VacuumVesselBuilder.BODY)
            .shape.boundary[0]
        )

    def add_ports(self, ports: Component | list[Component], n_TF: int):
        """
        Add a series of ports to the vacuum vessel component tree.
        """
        component = self.component()
        xyz = component.get_component("xyz")
        vv_xyz = xyz.get_component("Sector 1")
        target_void = vv_xyz.get_component("Vessel voidspace 1").shape
        vv_body = vv_xyz.get_component("Body 1")
        target_shape = vv_body.shape

        if isinstance(ports, Component):
            ports = [ports]

        tool_voids = []
        new_shape_pieces = []
        for i, port in enumerate(ports):
            if i > 0:
                target_shape = boolean_fuse(new_shape_pieces)

            port_xyz = port.get_component("xyz")
            tool_shape = port_xyz.get_component(port.name).shape
            tool_void = port_xyz.get_component(port.name + " voidspace").shape
            tool_voids.append(tool_void)
            new_shape_pieces = pipe_pipe_join(
                target_shape, target_void, tool_shape, tool_void
            )

        final_shape = boolean_fuse(new_shape_pieces)
        final_void = boolean_fuse([target_void, *tool_voids])

        sector_body = PhysicalComponent(
            VacuumVesselBuilder.BODY, final_shape, material=vv_body.material
        )
        sector_void = PhysicalComponent(
            VacuumVesselBuilder.VOID, final_void, material=Void("vacuum")
        )

        self._orphan_old_components(component)
        self._create_new_components(sector_body, sector_void, n_TF)

    def _create_new_components(self, sector_body, sector_void, n_TF: int):
        angle = 180 / n_TF
        component = self.component()
        apply_component_display_options(sector_body, color=BLUE_PALETTE["VV"][0])
        apply_component_display_options(sector_void, color=(0, 0, 0))
        Component(
            "xyz",
            children=[Component("Sector 1", children=[sector_body, sector_void])],
            parent=component,
        )

        self._make_2d_views(
            component,
            sector_body,
            sector_void,
            angle,
            BLUE_PALETTE["TS"][0],
            void_color=(0, 0, 0),
        )


@dataclass
class VacuumVesselBuilderParams(ParameterFrame):
    """
    Vacuum Vessel builder parameters
    """

    n_TF: Parameter[int]
    r_vv_ib_in: Parameter[float]
    r_vv_ob_in: Parameter[float]
    tk_vv_in: Parameter[float]
    tk_vv_out: Parameter[float]
    g_vv_bb: Parameter[float]
    vv_in_off_deg: Parameter[float]
    vv_out_off_deg: Parameter[float]


class VacuumVesselBuilder(Builder):
    """
    Vacuum Vessel builder
    """

    VV = "VV"
    BODY = "Body"
    VOID = "Vessel voidspace"
    param_cls: type[VacuumVesselBuilderParams] = VacuumVesselBuilderParams

    def __init__(
        self,
        params: ParameterFrame | dict,
        build_config: dict,
        ivc_koz: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.ivc_koz = ivc_koz

    def build(self) -> Component:
        """
        Build the vacuum vessel component.

        Returns
        -------
        :
            The built component tree
        """
        xz_vv, xz_vacuum = self.build_xz()
        vv_face = xz_vv.get_component_properties("shape")
        vacuum_face = xz_vacuum.get_component_properties("shape")

        return self.component_tree(
            xz=[xz_vv, xz_vacuum],
            xy=self.build_xy(vv_face),
            xyz=self.build_xyz(vv_face, vacuum_face, degree=0),
        )

    def build_xz(
        self,
    ) -> tuple[PhysicalComponent, ...]:
        """
        Build the x-z components of the vacuum vessel.

        Returns
        -------
        :
            The xz component parts
        """
        inner_vv = _offset_wire_discretised(
            self.ivc_koz,
            self.params.g_vv_bb.value,
            join="arc",
            open_wire=False,
            ndiscr=600,
        )

        outer_vv = varied_offset(
            inner_vv,
            self.params.tk_vv_in.value,
            self.params.tk_vv_out.value,
            self.params.vv_in_off_deg.value,
            self.params.vv_out_off_deg.value,
            num_points=300,
        )
        inner_vv = force_wire_to_spline(inner_vv, n_edges_max=100)
        outer_vv = force_wire_to_spline(outer_vv, n_edges_max=100)
        face = BluemiraFace([outer_vv, inner_vv])

        body = PhysicalComponent(
            self.BODY,
            face,
            material=self.get_material(),
        )
        vacuum = PhysicalComponent(
            self.VOID, BluemiraFace(inner_vv), material=Void("vacuum")
        )
        apply_component_display_options(body, color=BLUE_PALETTE[self.VV][0])
        apply_component_display_options(vacuum, color=(0, 0, 0))

        return body, vacuum

    def build_xy(self, vv_face: BluemiraFace) -> list[PhysicalComponent]:
        """
        Build the x-y components of the vacuum vessel.

        Returns
        -------
        :
            The xy component parts
        """
        return build_sectioned_xy(vv_face, BLUE_PALETTE[self.VV][0])

    def build_xyz(
        self, vv_face: BluemiraFace, vacuum_face: BluemiraFace, degree: float = 360.0
    ) -> list[PhysicalComponent]:
        """
        Build the x-y-z components of the vacuum vessel.

        Returns
        -------
        :
            The xyz component parts
        """
        return build_sectioned_xyz(
            [vv_face, vacuum_face],
            [self.BODY, self.VOID],
            self.params.n_TF.value,
            [BLUE_PALETTE[self.VV][0], (0, 0, 0)],
            degree,
            material=[
                self.get_material(),
                Void("vacuum"),
            ],
        )

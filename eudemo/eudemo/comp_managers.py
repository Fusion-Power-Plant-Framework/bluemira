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
EUDEMO thermal shield classes
"""

from typing import List

from bluemira.base.builder import ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.builders.cryostat import CryostatBuilder
from bluemira.builders.thermal_shield import CryostatTSBuilder, VVTSBuilder
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.tools import boolean_cut, boolean_fuse
from bluemira.geometry.wire import BluemiraWire
from bluemira.materials import Void
from eudemo.maintenance.duct_connection import pipe_pipe_join


class VacuumVesselThermalShield(ComponentManager):
    """
    Wrapper around a VVTS component tree.
    """

    def xz_boundary(self) -> BluemiraWire:
        """Return a wire representing the VVTS poloidal silhouette."""
        return (
            self.component()
            .get_component("xz")
            .get_component(VVTSBuilder.VVTS)
            .shape.boundary[0]
        )


class CryostatThermalShield(ComponentManager):
    """
    Wrapper around a VVTS component tree.
    """

    def xz_boundary(self) -> BluemiraWire:
        """Return a wire representing the VVTS poloidal silhouette."""
        return (
            self.component()
            .get_component("xz")
            .get_component(CryostatTSBuilder.CRYO_TS)
            .shape.boundary[0]
        )


class ThermalShield(ComponentManager):
    """
    Wrapper around a Thermal Shield component tree.
    """

    def vacuum_vessel_thermal_shield(self) -> Component:
        return self.component().get_component("VVTS")

    def cryostat_thermal_shield(self) -> Component:
        return self.component().get_component("CryostatTS")

    def add_ports(self, ports: List[Component]):
        vvts = self.vacuum_vessel_thermal_shield()
        cts = self.cryostat_thermal_shield()

        vvts_xyz = vvts.get_component("xyz")
        vv_xyz = vvts_xyz.get_component("Sector 1")
        vvts_target_name = f"{VVTSBuilder.VOID} 1"
        vvts_void_name = f"{VVTSBuilder.VVTS} 1"
        vvts_target_void = vv_xyz.get_component(vvts_target_name).shape
        vvts_target_shape = vv_xyz.get_component(vvts_void_name).shape

        cts_xyz = cts.get_component("xyz")
        cr_xyz = cts_xyz.get_component("Sector 1")
        cts_target_name = f"{CryostatTSBuilder.VOID} 1"
        cts_void_name = f"{CryostatTSBuilder.CRYO_TS} 1"
        cts_target_void = cr_xyz.get_component(cts_target_name).shape
        cts_target_shape = cr_xyz.get_component(cts_void_name).shape

        if isinstance(ports, Component):
            ports = [ports]

        tool_voids = []
        new_shape_pieces = []
        for i, port in enumerate(ports):
            port_xyz = port.get_component("xyz")
            tool_shape = port_xyz.get_component(port.name).shape
            tool_void = port_xyz.get_component(port.name + " voidspace").shape
            tool_voids.append(tool_void)
            result_pieces = pipe_pipe_join(
                vvts_target_shape, vvts_target_void, tool_shape, tool_void
            )
            # Assume the body is the biggest piece
            result_pieces.sort(key=lambda solid: -solid.volume)
            vvts_target_shape = result_pieces[0]
            new_shape_pieces.extend(result_pieces[1:])

        final_shape = boolean_fuse([vvts_target_shape] + new_shape_pieces)
        final_void = boolean_fuse([vvts_target_void] + tool_voids)

        temp = boolean_cut(final_shape, cts_target_shape)
        temp.sort(key=lambda solid: -solid.volume)
        final_shape = temp[0]

        cts_target_shape = boolean_cut(cts_target_shape, tool_voids)[0]

        vvts_sector_body = PhysicalComponent(vvts_target_name, final_shape)
        vvts_sector_void = PhysicalComponent(
            vvts_void_name, final_void, material=Void("vacuum")
        )
        apply_component_display_options(vvts_sector_body, color=BLUE_PALETTE["TS"][0])
        apply_component_display_options(vvts_sector_void, color=(0, 0, 0))

        cts_sector_body = PhysicalComponent(cts_target_name, cts_target_shape)
        cts_sector_void = PhysicalComponent(
            cts_void_name, cts_target_void, material=Void("vacuum")
        )
        apply_component_display_options(cts_sector_body, color=BLUE_PALETTE["TS"][0])
        apply_component_display_options(cts_sector_void, color=(0, 0, 0))

        # Orphan and kill old shapes
        vvts_xyz.parent = None
        del vvts_xyz
        cts_xyz.parent = None
        del cts_xyz
        # Replace xyz components
        Component(
            "xyz",
            children=[
                Component("Sector 1", children=[vvts_sector_body, vvts_sector_void])
            ],
            parent=vvts,
        )
        Component(
            "xyz",
            children=[
                Component("Sector 1", children=[cts_sector_body, cts_sector_void])
            ],
            parent=cts,
        )


class Cryostat(ComponentManager):
    """
    Wrapper around a VVTS component tree.
    """

    def xz_boundary(self) -> BluemiraWire:
        """Return a wire representing the VVTS poloidal silhouette."""
        return (
            self.component()
            .get_component("xz")
            .get_component(CryostatBuilder.CRYO)
            .shape.boundary[0]
        )


class RadiationShield(ComponentManager):
    """
    Wrapper around a VVTS component tree.
    """

    def xz_boundary(self) -> BluemiraWire:
        """Return a wire representing the VVTS poloidal silhouette."""
        return self.component().get_component("xz").shape.boundary[0]


class CoilStructures(ComponentManager):
    """
    Wrapper around the coil structures component tree
    """

    pass

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
Plasma builder.
"""

from typing import Dict

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, revolve_shape
from bluemira.geometry.wire import BluemiraWire


class Plasma(ComponentManager):
    """
    Wrapper around a plasma component tree.
    """

    def lcfs(self) -> BluemiraWire:
        """Return a wire representing the last-closed flux surface."""
        return (
            self.component()
            .get_component("xz")
            .get_component(PlasmaBuilder.LCFS)
            .shape.boundary[0]
        )


class PlasmaBuilder(Builder):
    """
    Builder for a poloidally symmetric plasma.
    """

    LCFS = "LCFS"

    # This builder has no parameters
    param_cls = None

    def __init__(
        self,
        build_config: Dict,
        xz_lcfs: BluemiraWire,
    ):
        super().__init__(None, build_config)
        self.xz_lcfs = xz_lcfs

    def build(self) -> Component:
        """
        Build the plasma component.
        """
        return self.component_tree(
            xz=[self.build_xz(self.xz_lcfs)],
            xy=[self.build_xy(self.xz_lcfs)],
            xyz=[self.build_xyz(self.xz_lcfs)],
        )

    def build_xz(self, lcfs: BluemiraWire) -> PhysicalComponent:
        """
        Build the x-z components of the plasma.

        Parameters
        ----------
        lcfs: BluemiraWire
            LCFS wire
        """
        face = BluemiraFace(lcfs, self.name)
        component = PhysicalComponent(self.LCFS, face)
        component.plot_options.face_options["color"] = BLUE_PALETTE["PL"]
        return component

    def build_xy(self, lcfs: BluemiraWire) -> PhysicalComponent:
        """
        Build the x-y components of the plasma.

        Parameters
        ----------
        lcfs: BluemiraWire
            LCFS wire
        """
        inner = make_circle(lcfs.bounding_box.x_min, axis=[0, 1, 0])
        outer = make_circle(lcfs.bounding_box.x_max, axis=[0, 1, 0])
        face = BluemiraFace([outer, inner], self.name)
        component = PhysicalComponent(self.LCFS, face)
        component.plot_options.face_options["color"] = BLUE_PALETTE["PL"]
        return component

    def build_xyz(self, lcfs: BluemiraWire, degree: float = 360.0) -> PhysicalComponent:
        """
        Build the x-y-z components of the plasma.

        Parameters
        ----------
        lcfs: BluemiraWire
            LCFS wire
        degree: float
            degrees to sweep the shape
        """
        shell = revolve_shape(lcfs, direction=(0, 0, 1), degree=degree)
        component = PhysicalComponent(self.LCFS, shell)
        component.display_cad_options.color = BLUE_PALETTE["PL"]
        return component

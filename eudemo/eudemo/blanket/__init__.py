# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Builders, designers, and components for an EUDEMO blanket."""

from bluemira.base.components import Component
from bluemira.base.reactor import ComponentConstructionType, ComponentManager
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire
from eudemo.blanket.builder import BlanketBuilder
from eudemo.blanket.designer import BlanketDesigner

__all__ = ["Blanket", "BlanketBuilder", "BlanketDesigner"]


class Blanket(ComponentManager):
    """Wrapper around a Blanket component tree."""

    def __init__(
        self, component_tree: Component, panel_points: Coordinates, r_inner_cut: float
    ):
        self.r_inner_cut = r_inner_cut
        self._panel_points = panel_points
        super().__init__(component_tree)

    @property
    def construction_type(self) -> ComponentConstructionType:  # noqa: D102
        return ComponentConstructionType.COMPOUND

    def panel_points(self) -> Coordinates:
        """The panel points of the blanket."""
        return self._panel_points

    def inboard_xz_face(self) -> BluemiraFace:
        """The poloidal plane face of the inboard blanket segment."""
        return (
            self.component().get_component("xz").get_component(BlanketBuilder.IBS).shape
        )

    def outboard_xz_face(self) -> BluemiraFace:
        """The poloidal plane face of the outboard blanket segment."""
        return (
            self.component().get_component("xz").get_component(BlanketBuilder.OBS).shape
        )

    def inboard_xz_boundary(self) -> BluemiraWire:
        """The toroidal plane silhouette of the inboard blanket segment."""
        return self.inboard_xz_face().boundary[0]

    def outboard_xz_boundary(self) -> BluemiraWire:
        """The poloidal plane silhouette of the outboard blanket segment."""
        return self.outboard_xz_face().boundary[0]

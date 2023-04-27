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
"""Builders, designers, and components for an EUDEMO blanket."""

from bluemira.base.reactor import ComponentManager
from bluemira.geometry.face import BluemiraFace
from eudemo.blanket.builder import BlanketBuilder
from eudemo.blanket.designer import BlanketDesigner

__all__ = ["Blanket", "BlanketBuilder", "BlanketDesigner"]


class Blanket(ComponentManager):
    """Wrapper around a Blanket component tree."""

    def inboard_xz_silhouette(self) -> BluemiraFace:
        """The poloidal plane silhouette of the inboard blanket segment."""
        return (
            self.component().get_component("xz").get_component(BlanketBuilder.IBS).shape
        )

    def outboard_xz_silhouette(self) -> BluemiraFace:
        """The poloidal plane silhouette of the outboard blanket segment."""
        return (
            self.component().get_component("xz").get_component(BlanketBuilder.OBS).shape
        )

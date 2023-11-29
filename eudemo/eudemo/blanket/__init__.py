# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
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

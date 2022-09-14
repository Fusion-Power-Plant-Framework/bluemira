# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Wrapper for FreeCAD Part.Face objects
"""

from __future__ import annotations

# import from freecad
import bluemira.codes._freecadapi as cadapi

# import from bluemira
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire

__all__ = ["BluemiraShell"]


class BluemiraShell(BluemiraGeo):
    """Bluemira Shell class."""

    def __init__(self, shape, label: str = ""):
        shape_classes = [cadapi.apiShell]
        super().__init__(shape, label, shape_classes)

    @property
    def vertexes(self):
        """
        The vertexes of the shell.
        """
        return Coordinates(cadapi.vertexes(self.shape))

    @property
    def edges(self):
        """
        The edges of the shell.
        """
        return [BluemiraWire(cadapi.apiWire(o)) for o in cadapi.edges(self.shape)]

    @property
    def wires(self):
        """
        The wires of the shell.
        """
        return [BluemiraWire(o) for o in cadapi.wires(self.shape)]

    @property
    def faces(self):
        """
        The faces of the shell.
        """
        return [BluemiraFace(o) for o in cadapi.faces(self.shape)]

    @property
    def shells(self):
        """
        The shells of the shell. By definition a list of itself.
        """
        return [self]

    @property
    def solids(self):
        """
        The solids of the shell. By definition and empty list.
        """
        return []

    @property
    def boundary(self):
        """
        The boundaries of the shell.
        """
        return self.faces

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
Wrapper for FreeCAD Part.Compounds objects
"""
# Note: this class is mainly used in the mesh module to allow the mesh of Components.
#       Indeed, Component shape for meshing purpose is considered as the compound of
#       all the component's children shapes.
#       Please note that information as length, area, and volume, could not be relevant.
#       They could be set to None or reimplemented, in case.

from __future__ import annotations

from typing import Tuple

import bluemira.codes._freecadapi as cadapi
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.wire import BluemiraWire


class BluemiraCompound(BluemiraGeo):
    """Bluemira Compound class."""

    def __init__(self, boundary, label: str = ""):
        boundary_classes = [BluemiraGeo]
        super().__init__(boundary, label, boundary_classes)

    def _create_shape(self) -> cadapi.apiCompound:
        """apiCompound: shape of the object as a single compound"""
        return cadapi.apiCompound([s.shape for s in self.boundary])

    @property
    def vertexes(self) -> Coordinates:
        """
        The ordered vertexes of the compound.
        """
        return Coordinates(cadapi.ordered_vertexes(self.shape))

    @property
    def edges(self) -> Tuple[BluemiraWire]:
        """
        The ordered edges of the compound.
        """
        return tuple(
            [BluemiraWire(cadapi.apiWire(o)) for o in cadapi.ordered_edges(self.shape)]
        )

    @property
    def wires(self) -> Tuple[BluemiraWire]:
        """
        The wires of the compound.
        """
        return tuple([BluemiraWire(o) for o in cadapi.wires(self.shape)])

    @property
    def faces(self) -> Tuple[BluemiraFace]:
        """
        The faces of the compound.
        """
        return tuple([BluemiraFace(o) for o in cadapi.faces(self.shape)])

    @property
    def shells(self) -> Tuple[BluemiraShell]:
        """
        The shells of the compound.
        """
        return tuple([BluemiraShell(o) for o in cadapi.shells(self.shape)])

    @property
    def solids(self) -> Tuple[BluemiraSolid]:
        """
        The solids of the compound.
        """
        return tuple([BluemiraSolid(o) for o in cadapi.solids(self.shape)])

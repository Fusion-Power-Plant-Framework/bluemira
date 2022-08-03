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

from typing import List

import numpy as np

import bluemira.codes._freecadapi as cadapi

# import from bluemira
from bluemira.geometry.base import BluemiraGeo

# import from error
from bluemira.geometry.error import DisjointedFace, NotClosedWire
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.coordinates import Coordinates

__all__ = ["BluemiraFace"]


class BluemiraFace(BluemiraGeo):
    """Bluemira Face class."""

    def __init__(self, shape, label: str = ""):
        shape_classes = [cadapi.apiFace]
        super().__init__(shape, label, shape_classes)

    @property
    def vertexes(self):
        return Coordinates(cadapi.vertexes(self.shape))

    @property
    def edges(self):
        return [BluemiraWire(cadapi.apiWire(o)) for o in cadapi.edges(self.shape)]

    @property
    def wires(self):
        return [BluemiraWire(o) for o in cadapi.wires(self.shape)]

    @property
    def faces(self):
        return [self]

    @property
    def shells(self):
        return []

    @property
    def solids(self):
        return []

    @property
    def boundary(self):
        return self.wires
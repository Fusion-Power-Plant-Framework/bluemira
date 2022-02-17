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
Wrapper for FreeCAD Part.Compounds objects
"""
# Note: this class is mainly used in the mesh module to allow the mesh of Components.
#       Indeed, Component shape for meshing purpose is considered as the compound of
#       all the component's children shapes.
#       Please note that information as length, area, and volume, could not be relevant.
#       They could be set to None or reimplemented, in case.

from __future__ import annotations

from bluemira.codes._freecadapi import apiCompound
from bluemira.geometry.base import BluemiraGeo


class BluemiraCompound(BluemiraGeo):
    """Bluemira Compound class."""

    def __init__(self, boundary, label: str = ""):
        boundary_classes = [BluemiraGeo]
        super().__init__(boundary, label, boundary_classes)

    @property
    def _shape(self) -> apiCompound:
        """apiCompound: shape of the object as a single compound"""
        return apiCompound([s._shape for s in self.boundary])

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

import bluemira.codes._freecadapi as cadapi
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.wire import BluemiraWire


class BluemiraCompound(BluemiraGeo):
    """Bluemira Compound class."""

    def __init__(self, shape, label: str = ""):
        shape_classes = [BluemiraGeo]
        super().__init__(shape, label, shape_classes)

    def _check_shape(self, shape):
        """
        Check if shape is a valid object to be wrapped in BluemiraCompound
        """
        if isinstance(shape, self.__class__):
            return shape._shape

        is_list = isinstance(shape, list)
        if not is_list:
            shape = [shape]

        check = False

        for ind, s in enumerate(shape):
            if isinstance(s, self.__class__):
                shape[ind] = s.shape

        for c in self._shape_classes:
            check = check or all(isinstance(s, c) for s in shape)

        if check:
            if len(shape) == 1:
                return shape[0]
            else:
                return cadapi.apiCompound(shape)
        raise TypeError(
            f"Only {self._shape_classes} objects can be used for {self.__class__}"
        )

    @property
    def vertexes(self):
        """
        The vertexes of the compound.
        """
        return Coordinates(cadapi.vertexes(self.shape))

    @property
    def edges(self):
        """
        The edges of the compound.
        """
        return [BluemiraWire(cadapi.apiWire(o)) for o in cadapi.edges(self.shape)]

    @property
    def wires(self):
        """
        The wires of the compound.
        """
        return [BluemiraWire(o) for o in cadapi.wires(self.shape)]

    @property
    def faces(self):
        """
        The faces of the compound.
        """
        return [BluemiraFace(o) for o in cadapi.faces(self.shape)]

    @property
    def shells(self):
        """
        The shells of the compound.
        """
        return [BluemiraShell(o) for o in cadapi.shells(self.shape)]

    @property
    def solids(self):
        """
        The solids of the compound.
        """
        return [BluemiraSolid(o) for o in cadapi.solids(self.shape)]

    @property
    def boundary(self):
        """
        The boundaries of the compound. Ill-defined, so None.
        """
        return None

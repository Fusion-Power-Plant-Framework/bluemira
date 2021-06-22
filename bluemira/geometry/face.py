# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

from typing import Union, List

# import from freecad
import freecad
import Part

# import from bluemira
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.wire import BluemiraWire

# import from error
from bluemira.geometry.error import NotClosedWire, DisjointedFace


class BluemiraFace(BluemiraGeo):
    """Bluemira Face class."""
    def __init__(
            self,
            boundary,
            label: str = "",
            lcar: Union[float, List[float]] = 0.1
    ):
        boundary_classes = [BluemiraWire]
        super().__init__(boundary, label, lcar, boundary_classes)

    def _check_boundary(self, objs):
        """Check if objects in objs are of the correct type for this class"""
        if not hasattr(objs, '__len__'):
            objs = [objs]
        check = False
        for c in self._boundary_classes:
            check = check or (all(isinstance(o, c) for o in objs))
            if check:
                if all(o.is_closed() for o in objs):
                    return objs
                else:
                    raise NotClosedWire("Only closed BluemiraWire are accepted.")
        raise TypeError("Only {} objects can be used for {}".format(
            self._boundary_classes, self.__class__))

    @BluemiraGeo.boundary.setter
    def boundary(self, objs):
        self._boundary = self._check_boundary(objs)
        # The face is created here to have consistency between boundary and face.
        self._face = self._createFace()

    def _createFace(self):
        """ """
        external: BluemiraWire = self.boundary[0]
        face = Part.Face(external._shape)
        if len(self.boundary) > 1:
            fholes = [Part.Face(h._shape) for h in self.boundary[1:]]
            face = face.cut(fholes)
            if len(face.Faces) == 1:
                face = face.Faces[0]
            else:
                raise DisjointedFace("Any or more than one face has been created.")
        return face

    @property
    def _shape(self):
        """Part.Wire: shape of the object as a single wire"""
        return self._face

    @property
    def Wires(self) -> List[Part.Wire]:
        """list(Part.Wire): list of wires of which the shape consists of."""

        # Note:the method is recursively implemented considering that
        # Part.Wire has a similar Wires property.

        wires = []
        for o in self.boundary:
            wires += o.Wires
        return wires

    @staticmethod
    def create(cls, obj: Part.Face):
        if isinstance(obj, Part.Face):
            wires = obj.Wires
            bmwire = BluemiraWire(wires)
            bmface = BluemiraFace(bmwire)
            return bmface
        raise TypeError("Only Part.Face objects can be used to create a {} "
                        "instance".format(cls))

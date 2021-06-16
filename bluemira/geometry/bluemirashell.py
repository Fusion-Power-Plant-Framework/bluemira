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
from bluemira.geometry.bluemirageo import BluemiraGeo
from bluemira.geometry.bluemirawire import BluemiraWire
from bluemira.geometry.bluemiraface import BluemiraFace

# import from error
from bluemira.geometry.error import NotClosedWire, DisjointedFace


class BluemiraShell(BluemiraGeo):
    """Bluemira Solid class."""
    def __init__(
            self,
            boundary,
            label: str = "",
            lcar: Union[float, List[float]] = 0.1
    ):
        boundary_classes = [BluemiraFace]
        super().__init__(boundary, label, lcar, boundary_classes)

    def _check_boundary(self, objs):
        """Check if objects in objs are of the correct type for this class"""
        return super()._check_boundary(objs)

    @BluemiraGeo.boundary.setter
    def boundary(self, objs):
        self._boundary = self._check_boundary(objs)
        # The shell is created here to have consistency between boundary and face.
        self._shell = self._createShell()

    def _createShell(self):
        """ Creation of the shell"""
        faces = [f.shape for f in self.boundary]
        return Part.makeShell(faces)

    @property
    def shape(self):
        """Part.Wire: shape of the object as a single wire"""
        return self._shell

    @staticmethod
    def create(cls, obj: Part.Shell):
        if isinstance(obj, Part.Shell):
            faces = obj.Faces
            bmfaces = []
            for f in faces:
                wires = face.Wires
                bmwire = BluemiraWire(wires)
                bmfaces.append(BluemiraFace(bmwire))
            bmshell = BluemiraShell(bmfaces)
            return bmshell
        raise TypeError("Only Part.Shell objects can be used to create a {} "
                        "instance".format(cls))
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
import freecad  # noqa: F401
import Part

# import from bluemira
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.face import BluemiraFace


class BluemiraShell(BluemiraGeo):
    """Bluemira Shell class."""

    def __init__(self, boundary, label: str = ""):
        boundary_classes = [BluemiraFace]
        super().__init__(boundary, label, boundary_classes)

    def _check_boundary(self, objs):
        """Check if objects in objs are of the correct type for this class"""
        return super()._check_boundary(objs)

    def _create_shell(self):
        """Creation of the shell"""
        faces = [f._shape for f in self.boundary]
        return Part.makeShell(faces)

    @property
    def _shape(self):
        """Part.Shell: shape of the object as a primitive shell"""
        return self._create_shell()

    @classmethod
    def _create(cls, obj: Part.Shell):
        if isinstance(obj, Part.Shell):
            faces = obj.Faces
            bmfaces = []
            for face in faces:
                bmfaces.append(BluemiraFace._create(face))
            bmshell = BluemiraShell(bmfaces)
            return bmshell
        raise TypeError(
            f"Only Part.Shell objects can be used to create a {cls} instance"
        )

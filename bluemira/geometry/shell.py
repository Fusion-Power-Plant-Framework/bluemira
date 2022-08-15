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
from bluemira.geometry.face import BluemiraFace

__all__ = ["BluemiraShell"]

SHELL = []


class BluemiraShell(BluemiraGeo):
    """Bluemira Shell class."""

    def __init__(self, boundary, label: str = ""):
        boundary_classes = [BluemiraFace]
        super().__init__(boundary, label, boundary_classes)

    def _create_shell(self, check_reverse=True):
        """Creation of the shell"""
        faces = [f._create_face(check_reverse=True) for f in self.boundary]
        shell = cadapi.apiShell(faces)
        SHELL.append(shell)
        if check_reverse:
            return self._check_reverse(shell)
        else:
            return shell

    @property
    def _shape(self):
        """Part.Shell: shape of the object as a primitive shell"""
        return self._create_shell()

    @classmethod
    def _create(cls, obj: cadapi.apiShell, label=""):
        if isinstance(obj, cadapi.apiShell):
            orientation = obj.Orientation
            faces = obj.Faces
            bmfaces = []
            for face in faces:
                bmfaces.append(BluemiraFace._create(face))
            bmshell = BluemiraShell(bmfaces, label=label)
            bmshell._orientation = orientation
            return bmshell
        raise TypeError(
            f"Only Part.Shell objects can be used to create a {cls} instance"
        )

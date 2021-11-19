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
import bluemira.geometry._freecadapi as cadapi

# import from bluemira
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.shell import BluemiraShell

from bluemira.geometry.error import DisjointedSolid


class BluemiraSolid(BluemiraGeo):
    """Bluemira Solid class."""

    def __init__(self, boundary, label: str = ""):
        boundary_classes = [BluemiraShell]
        super().__init__(boundary, label, boundary_classes)

    def _create_solid(self):
        """Creation of the solid"""
        new_shell = self.boundary[0]._shape
        # for o in self.boundary[1:]:
        #     new_shell = new_shell.fuse(o._shape)
        return cadapi.make_solid(new_shell)

    @property
    def _shape(self):
        """Part.Solid: shape of the object as a single solid"""
        return self._create_solid()

    @classmethod
    def _create(cls, obj: cadapi.apiSolid, label=""):
        if isinstance(obj, cadapi.apiSolid):
            shells = obj.Shells
            if len(shells) == 1:
                bmshell = BluemiraShell._create(shells[0])
                return cls(bmshell, label=label)

            else:
                raise DisjointedSolid("Disjointed solids are not accepted.")
        raise TypeError(
            f"Only Part.Solid objects can be used to create a {cls} instance"
        )

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
Wrapper for FreeCAD Part.Compound objects
"""
from bluemira.geometry.error import GeometryError
import bluemira.geometry._freecadapi as cadapi

from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.solid import BluemiraSolid


class BluemiraCompound(BluemiraGeo):
    """BluemiraCompound class"""

    def __init__(self, boundary, label: str = ""):
        boundary_classes = [BluemiraSolid]
        super().__init__(boundary, label, boundary_classes)

    def _create_compound(self):
        solids = [f._shape for f in self.boundary]
        return cadapi.make_compound(solids)

    @property
    def _shape(self):
        """Part.Solid: shape of the object as a single solid"""
        return self._create_compound()

    @classmethod
    def _create(cls, obj: cadapi.apiCompound, label=""):
        if not isinstance(obj, cadapi.apiCompound):
            raise TypeError(
                f"Only Part.Compound objects can be used to create a {cls} instance"
            )

        solids = obj.Solids
        if not solids:
            raise GeometryError(
                "BluemiraCompound can only be made from solid compounds."
            )

        return cls(solids, label=label)

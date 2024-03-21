# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Wrapper for FreeCAD Part.Face objects
"""

from __future__ import annotations

# import from freecad
import bluemira.codes._freecadapi as cadapi

# import from bluemira
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import DisjointedSolidError, GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.wire import BluemiraWire

__all__ = ["BluemiraSolid"]


class BluemiraSolid(BluemiraGeo):
    """
    Bluemira Solid class.

    Parameters
    ----------
    boundary:
        List of shells from  which to make the BluemiraSolid
    label:
        Label to assign to the solid
    """

    def __init__(self, boundary: list[BluemiraShell], label: str = ""):
        boundary_classes = [BluemiraShell]
        super().__init__(boundary, label, boundary_classes)

    def _create_solid(self, check_reverse: bool = True):
        """Creation of the solid"""
        new_shell = self.boundary[0]._create_shell(check_reverse=False)
        solid = cadapi.apiSolid(new_shell)

        if len(self.boundary) > 1:
            shell_holes = [cadapi.apiSolid(s.shape) for s in self.boundary[1:]]
            solid = solid.cut(shell_holes)
            if len(solid.Solids) == 1:
                solid = solid.Solids[0]
            else:
                raise DisjointedSolidError("Disjointed solids are not accepted.")

        if check_reverse:
            return self._check_reverse(cadapi.apiSolid(solid))
        return solid

    def _create_shape(self):
        """Part.Solid: shape of the object as a single solid"""
        return self._create_solid()

    @classmethod
    def _create(cls, obj: cadapi.apiSolid, label: str = "") -> BluemiraSolid:
        if isinstance(obj, cadapi.apiSolid):
            if len(obj.Solids) > 1:
                raise DisjointedSolidError("Disjointed solids are not accepted.")

            if not obj.isValid():
                # cadapi.save_as_STP(obj, "object_not_valid")
                raise GeometryError(f"Solid {obj} is not valid.")

            bm_shells = [BluemiraShell._create(shell) for shell in obj.Shells]

            # create an empty BluemiraSolid
            bmsolid = cls(None, label=label)
            # assign shape, boundary, and orientation
            bmsolid._set_shape(obj)
            bmsolid._boundary = bm_shells
            bmsolid._orientation = obj.Orientation
            return bmsolid

        raise TypeError(
            f"Only Part.Solid objects can be used to create a {cls} instance"
        )

    @property
    def vertexes(self) -> Coordinates:
        """
        The vertexes of the solid.
        """
        return Coordinates(cadapi.vertexes(self.shape))

    @property
    def edges(self) -> tuple[BluemiraWire]:
        """
        The edges of the solid.
        """
        return tuple([BluemiraWire(cadapi.apiWire(o)) for o in cadapi.edges(self.shape)])

    @property
    def wires(self) -> tuple[BluemiraWire]:
        """
        The wires of the solid.
        """
        return tuple([BluemiraWire(o) for o in cadapi.wires(self.shape)])

    @property
    def faces(self) -> tuple[BluemiraFace]:
        """
        The faces of the solid.
        """
        return tuple([BluemiraFace._create(o) for o in cadapi.faces(self.shape)])

    @property
    def shells(self) -> tuple[BluemiraShell]:
        """
        The shells of the solid.
        """
        return tuple([BluemiraShell._create(o) for o in cadapi.shells(self.shape)])

    @property
    def solids(self) -> tuple[BluemiraSolid]:
        """
        The solids of the solid. By definition a tuple of itself.
        """
        return (self,)

# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Wrapper for FreeCAD Part.Compounds objects
"""
# Note: this class is mainly used in the mesh module to allow the mesh of Components.
#       Indeed, Component shape for meshing purpose is considered as the compound of
#       all the component's children shapes.
#       Please note that information as length, area, and volume, could not be relevant.
#       They could be set to None or reimplemented, in case.

from __future__ import annotations

import bluemira.codes.cadapi as cadapi
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.wire import BluemiraWire


class BluemiraCompound(BluemiraGeo):
    """
    Bluemira Compound class.

    Parameters
    ----------
    boundary:
        List of BluemiraGeo objects to include in the compound
    label:
        Label to assign to the compound
    _compound_obj:
        A pre-constructed compound object to use for the shape. This should only be used
        by the _create classmethod.
    """

    def __init__(
        self,
        boundary: list[BluemiraGeo],
        label: str = "",
        *,
        _compound_obj: cadapi.apiCompound | None = None,
    ):
        boundary_classes = [BluemiraGeo]
        self._compound_obj = _compound_obj
        super().__init__(boundary, label, boundary_classes)

    def _create_shape(self) -> cadapi.apiCompound:
        """
        Returns
        -------
        apiCompound:
            Shape of the object as a single compound.
        """
        if self._compound_obj:
            return self._compound_obj
        return cadapi.apiCompound([s.shape for s in self.boundary])

    @classmethod
    def _create(cls, obj: cadapi.apiCompound, label="") -> BluemiraCompound:
        if not isinstance(obj, cadapi.apiCompound):
            raise TypeError(
                f"Only apiCompound objects can be used to create a {cls} instance"
            )
        if not obj.isValid():
            raise GeometryError(f"Compound {obj} is not valid.")

        if cadapi.solids(obj):
            topo_compound_shapes = [
                BluemiraSolid._create(solid) for solid in cadapi.solids(obj)
            ]
        elif cadapi.shells(obj):
            topo_compound_shapes = [
                BluemiraShell._create(shell) for shell in cadapi.shells(obj)
            ]
        elif cadapi.faces(obj):
            topo_compound_shapes = [
                BluemiraFace._create(face) for face in cadapi.faces(obj)
            ]
        elif cadapi.wires(obj):
            topo_compound_shapes = [BluemiraWire(wire) for wire in cadapi.wires(obj)]
        else:
            topo_compound_shapes = [
                BluemiraWire(wire)
                for wire in [cadapi.apiWire(o) for o in cadapi.edges(obj)]
            ]

        return cls(topo_compound_shapes, label=label, _compound_obj=obj)

    @property
    def vertexes(self) -> Coordinates:
        """
        The vertexes of the compound.
        """
        return Coordinates(cadapi.vertexes(self.shape))

    @property
    def edges(self) -> tuple[BluemiraWire, ...]:
        """
        The edges of the compound.
        """
        return tuple(BluemiraWire(cadapi.apiWire(o)) for o in cadapi.edges(self.shape))

    @property
    def wires(self) -> tuple[BluemiraWire, ...]:
        """
        The wires of the compound.
        """
        return tuple(BluemiraWire(o) for o in cadapi.wires(self.shape))

    @property
    def faces(self) -> tuple[BluemiraFace, ...]:
        """
        The faces of the compound.
        """
        return tuple(BluemiraFace._create(o) for o in cadapi.faces(self.shape))

    @property
    def shells(self) -> tuple[BluemiraShell, ...]:
        """
        The shells of the compound.
        """
        return tuple(BluemiraShell._create(o) for o in cadapi.shells(self.shape))

    @property
    def solids(self) -> tuple[BluemiraSolid, ...]:
        """
        The solids of the compound.
        """
        return tuple(BluemiraSolid._create(o) for o in cadapi.solids(self.shape))

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

from typing import TYPE_CHECKING

import bluemira.codes._freecadapi as cadapi
from bluemira.geometry.base import BluemiraShape
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.wire import BluemiraWire

if TYPE_CHECKING:
    from collections.abc import Iterable


class BluemiraCompound(BluemiraShape):
    """
    Bluemira Compound class.

    Parameters
    ----------
    boundary:
        List of BluemiraGeo objects to include in the compound
    label:
        Label to assign to the compound
    """

    def __init__(
        self,
        compound_obj: cadapi.apiCompound,
        label: str = "",
        *,
        constituents: Iterable[BluemiraShape] | None = None,
    ):
        self.label = label
        self._constituents = constituents
        super().__init__(compound_obj)

    @classmethod
    def _create(
        cls,
        obj: cadapi.apiCompound,
        label="",
        *,
        constituents: Iterable[BluemiraShape] | None = None,
    ) -> BluemiraCompound:
        if not isinstance(obj, cadapi.apiCompound):
            raise TypeError(
                f"Only apiCompound objects can be used to create a {cls} instance"
            )
        if not obj.isValid():
            raise GeometryError(f"Compound {obj} is not valid.")

        return cls(obj, label, constituents=constituents)

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

    @property
    def constituents(self) -> tuple[BluemiraShape, ...]:
        """
        The constituents of the compound.
        """
        if self._constituents:
            return tuple(self._constituents)
        return self.solids + self.shells + self.faces + (self.wires or self.edges)

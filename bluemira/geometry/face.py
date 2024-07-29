# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Wrapper for FreeCAD Part.Face objects
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bluemira.codes._freecadapi as cadapi

# import from bluemira
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.coordinates import Coordinates

# import from error
from bluemira.geometry.error import DisjointedFaceError, NotClosedWireError
from bluemira.geometry.wire import BluemiraWire

if TYPE_CHECKING:
    import numpy as np

__all__ = ["BluemiraFace"]


class BluemiraFace(BluemiraGeo):
    """
    Bluemira Face class.

    Parameters
    ----------
    boundary:
        List of BluemiraWires to use to make the face
    label:
        Label to assign to the BluemiraFace
    """

    def __init__(self, boundary: BluemiraWire | list[BluemiraWire], label: str = ""):
        boundary_classes = [BluemiraWire]
        super().__init__(boundary, label, boundary_classes)

    @staticmethod
    def _converter(func):
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            if isinstance(output, cadapi.apiWire):
                output = BluemiraWire(output)
            if isinstance(output, cadapi.apiFace):
                output = BluemiraFace._create(output)
            return output

        return wrapper

    def copy(self):
        """Make a copy of the BluemiraFace"""
        return BluemiraFace(self.boundary, self.label)

    def deepcopy(self, label: str | None = None) -> BluemiraFace:
        """Make a copy of the BluemiraFace"""
        boundary = []
        for o in self.boundary:
            boundary += [o.deepcopy(o.label)]
        geo_copy = BluemiraFace(boundary)
        if label is not None:
            geo_copy.label = label
        else:
            geo_copy.label = self.label
        return geo_copy

    def _check_boundary(self, objs):
        """Check if objects in objs are of the correct type for this class

        Raises
        ------
        TypeError
            Only wires are allowed as boundaries
        NotClosedWireError
            not all boundary wires are closed
        """
        if objs is None:
            return objs

        if not hasattr(objs, "__len__"):
            objs = [objs]
        check = False
        for c in self._boundary_classes:
            for o in objs:
                check = check or isinstance(o, c)
            if check:
                if all(o.is_closed() for o in objs):
                    return objs
                raise NotClosedWireError("Only closed BluemiraWire are accepted.")
        raise TypeError(
            f"Only {self._boundary_classes} objects can be used for {self.__class__}"
        )

    def _create_face(self, *, check_reverse: bool = True):
        """Create the primitive face

        Raises
        ------
        DisjointedFaceError
            More than 1 face created
        """
        external: BluemiraWire = self.boundary[0]
        face = cadapi.apiFace(external._create_wire(check_reverse=False))

        if len(self.boundary) > 1:
            fholes = [cadapi.apiFace(h.shape) for h in self.boundary[1:]]
            face = face.cut(fholes)
            if len(face.Faces) == 1:
                face = face.Faces[0]
            else:
                raise DisjointedFaceError("Any or more than one face has been created.")

        if check_reverse:
            return self._check_reverse(face)
        return face

    def _create_shape(self) -> cadapi.apiFace:
        """Part.Face: shape of the object as a primitive face"""
        return self._create_face()

    @classmethod
    def _create(cls, obj: cadapi.apiFace, label="") -> BluemiraFace:
        if isinstance(obj, cadapi.apiFace):
            bmwires = []
            for w in obj.Wires:
                w_orientation = w.Orientation
                bm_wire = BluemiraWire(w)
                bm_wire._orientation = w_orientation
                if cadapi.is_closed(w):
                    bm_wire.close()
                bmwires += [bm_wire]

            bmface = cls(None, label=label)
            bmface._set_shape(obj)
            bmface._boundary = bmwires
            bmface._orientation = obj.Orientation

            return bmface

        raise TypeError(f"Only Part.Face objects can be used to create a {cls} instance")

    def discretise(
        self, ndiscr: int = 100, *, byedges: bool = False, dl: float | None = None
    ) -> np.ndarray:
        """
        Make an array of the geometry.

        Parameters
        ----------
        ndiscr:
            Number of points in the array
        dl:
            Optional length discretisation (overrides ndiscr)
        byedges:
            Whether or not to discretise by edges

        Returns
        -------
        (M, (3, N)) array of point coordinates where M is the number of boundaries
        and N the number of discretisation points.
        """
        points = []
        for w in self.shape.Wires:
            if byedges:
                points.append(cadapi.discretise_by_edges(w, ndiscr=ndiscr, dl=dl))
            else:
                points.append(cadapi.discretise(w, ndiscr=ndiscr, dl=dl))
        return points

    def normal_at(self, alpha_1: float = 0.0, alpha_2: float = 0.0) -> np.ndarray:
        """
        Get the normal vector of the face at a parameterised point in space. For
        planar faces, the normal is the same everywhere.
        """
        return cadapi.normal_at(self.shape, alpha_1, alpha_2)

    @property
    def vertexes(self) -> Coordinates:
        """
        The vertexes of the face.
        """
        return Coordinates(cadapi.vertexes(self.shape))

    @property
    def edges(self) -> tuple[BluemiraWire]:
        """
        The edges of the face.
        """
        return tuple([BluemiraWire(cadapi.apiWire(o)) for o in cadapi.edges(self.shape)])

    @property
    def wires(self) -> tuple[BluemiraWire]:
        """
        The wires of the face.
        """
        return tuple([BluemiraWire(o) for o in cadapi.wires(self.shape)])

    @property
    def faces(self) -> tuple[BluemiraFace]:
        """
        The faces of the face. By definition a tuple of itself.
        """
        return (self,)

    @property
    def shells(self) -> tuple:
        """
        The shells of the face. By definition an empty tuple.
        """
        return ()

    @property
    def solids(self) -> tuple:
        """
        The solids of the face. By definition an empty tuple.
        """
        return ()

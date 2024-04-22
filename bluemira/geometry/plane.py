# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Wrapper for FreeCAD Plane objects
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bluemira.geometry.placement import BluemiraPlacement

import bluemira.codes._freecadapi as cadapi
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.face import BluemiraFace

__all__ = ["BluemiraPlane"]


class BluemiraPlane:
    """
    Bluemira Plane class.

    Parameters
    ----------
    base:
        Plane reference point
    axis:
        normal vector dto the plane
    label:
        Label of the plane
    """

    def __init__(
        self,
        base: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        label: str = "",
    ):
        if np.allclose(np.array(axis), np.array([0, 0, 0])):
            raise ValueError("Axis must to be a vector with non zero norm.")
        self._shape = cadapi.make_plane(base, axis)
        self.label = label

    @classmethod
    def from_3_points(
        cls,
        point_1: Iterable[float],
        point_2: Iterable[float],
        point_3: Iterable[float],
        label: str = "",
    ) -> BluemiraPlane:
        """
        Instantiate a BluemiraPlane from three points.

        Parameters
        ----------
        point_1:
            First point
        point_2:
            Second Point
        point_3:
            Third point
        label:
            Label of the plane
        """
        plane = BluemiraPlane()
        plane._shape = cadapi.make_plane_from_3_points(point_1, point_2, point_3)
        plane.label = label
        return plane

    @property
    def base(self) -> np.ndarray:
        """Plane's reference point"""
        return cadapi.vector_to_numpy(self._shape.Position)

    @base.setter
    def base(self, value: Iterable[float]):
        """
        Set a new plane base

        Parameters
        ----------
        value:
            Base vector
        """
        self._shape.Position = cadapi.Base.Vector(value)

    @property
    def axis(self) -> np.ndarray:
        """Plane's normal vector"""
        return cadapi.vector_to_numpy(self._shape.Axis)

    @axis.setter
    def axis(self, value: Iterable[float]):
        """
        Set a new plane axis

        Parameters
        ----------
        value:
            Axis vector
        """
        self._shape.Axis = cadapi.Base.Vector(value)

    def move(self, vector: Iterable[float]):
        """Moves the Plane along the given vector"""
        self.base += np.array(vector)

    def __repr__(self) -> str:
        """
        Plane __repr__
        """
        return (
            f"([{type(self).__name__}] = Label: {self.label},"
            f" base: {self.base},"
            f" axis: {self.axis})"
        )

    def copy(self, label: str | None = None) -> BluemiraPlane:
        """
        Make a copy of the BluemiraGeo.
        """
        plane_copy = copy.copy(self)
        if label is not None:
            plane_copy.label = label
        else:
            plane_copy.label = self.label
        return plane_copy

    def deepcopy(self, label: str | None = None) -> BluemiraPlane:
        """Make a deepcopy of the BluemiraPlane"""
        plane_copy = BluemiraPlane(self.base, self.axis)
        if label is not None:
            plane_copy.label = label
        else:
            plane_copy.label = self.label
        return plane_copy

    def to_face(
        self, width: float = VERY_BIG, height: float = VERY_BIG, label: str = ""
    ) -> BluemiraFace:
        """
        Convert the plane to a face with dimension (width, height) and centred into
        the plane base position.
        """
        face = cadapi.face_from_plane(self._shape, width, height)
        return BluemiraFace._create(face, label)

    def to_placement(self) -> BluemiraPlacement:
        """
        Convert the plane into a placement
        """
        from bluemira.geometry.placement import BluemiraPlacement  # noqa: PLC0415

        return BluemiraPlacement._create(cadapi.placement_from_plane(self._shape))

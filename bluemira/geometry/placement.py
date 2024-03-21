# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Wrapper for FreeCAD Placement objects
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import bluemira.codes._freecadapi as cadapi
from bluemira.geometry.error import GeometryError
from bluemira.geometry.plane import BluemiraPlane

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["BluemiraPlacement"]


class BluemiraPlacement:
    """
    Bluemira Placement class.

    Parameters
    ----------
    base:
        Placement origin
    axis:
        vector describing the axis of rotation
    angle:
        angle of rotation in degree
    label:
        Label of the placement
    """

    def __init__(
        self,
        base: Iterable[float] = [0.0, 0.0, 0.0],
        axis: Iterable[float] = [0.0, 0.0, 1.0],
        angle: float = 0.0,
        label: str = "",
    ):
        self._shape = cadapi.make_placement(base, axis, angle)
        self.label = label

    @classmethod
    def from_3_points(
        cls,
        point_1: Iterable[float],
        point_2: Iterable[float],
        point_3: Iterable[float],
        label: str = "",
    ) -> BluemiraPlacement:
        """
        Instantiate a BluemiraPlacement from three points.

        Parameters
        ----------
        point_1:
            First point
        point_2:
            Second Point
        point_3:
            Third point
        label:
            Label of the placement
        """
        p1 = np.array(point_1)
        p2 = np.array(point_2)
        p3 = np.array(point_3)
        v1, v2 = p3 - p1, p2 - p1
        v3 = np.cross(v2, v1)
        if np.all(v3 == 0):
            raise GeometryError("Cannot make a BluemiraPlacement from co-linear points.")

        normal = v3 / np.sqrt(v3.dot(v3))
        return cls(point_1, normal, 0.0, label=label)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, label: str = "") -> BluemiraPlacement:
        """
        Instantiate a BluemiraPlacement from a 4 x 4 matrix

        Parameters
        ----------
        matrix:
            4 x 4 matrix from which to make the placement
        label:
            Label of the placement
        """
        obj = cls.__new__(cls)
        obj._shape = cadapi.make_placement_from_matrix(matrix)
        obj.label = label
        return obj

    @property
    def base(self) -> np.ndarray:
        """Placement's local origin"""
        return cadapi.vector_to_numpy(self._shape.Base)

    @base.setter
    def base(self, value: Iterable[float]):
        """
        Set a new placement base

        Parameters
        ----------
        value:
            Base vector
        """
        self._shape.Base = cadapi.Base.Vector(value)

    @property
    def axis(self) -> np.ndarray:
        """Placement's rotation matrix"""
        return self._shape.Rotation.Axis

    @axis.setter
    def axis(self, value: Iterable[float]):
        """
        Set a new placement axis

        Parameters
        ----------
        value:
            Axis vector
        """
        self._shape.Axis = cadapi.Base.Vector(value)

    @property
    def angle(self) -> float:
        """Placement's angle of rotation"""
        return np.rad2deg(self._shape.Rotation.Angle)

    @angle.setter
    def angle(self, value: float):
        """
        Set a new placement angle of rotation

        Parameters
        ----------
        value:
            Angle value in degree
        """
        self._shape.Angle = value

    def to_matrix(self) -> np.ndarray:
        """Returns a matrix (quaternion) representing the Placement's transformation"""
        return np.array(self._shape.Matrix.A).reshape(4, 4)

    def inverse(self) -> BluemiraPlacement:
        """Returns the inverse placement"""
        return BluemiraPlacement._create(
            self._shape.inverse(), label=self.label + "_inverse"
        )

    def move(self, vector: Iterable[float]):
        """Moves the Placement along the given vector"""
        cadapi.move_placement(self._shape, vector)

    def __repr__(self):  # noqa: D105
        return (
            f"([{type(self).__name__}] = Label: {self.label}, "
            f"base: {self.base}, "
            f"axis: {self.axis}, "
            f"angle: {self.angle})"
        )

    def copy(self, label: str | None = None) -> BluemiraPlacement:
        """Make a copy of the BluemiraPlacement"""
        placement_copy = BluemiraPlacement(self.base, self.axis, self.angle)
        if label is not None:
            placement_copy.label = label
        else:
            placement_copy.label = self.label
        return placement_copy

    def deepcopy(self, label: str | None = None) -> BluemiraPlacement:  # noqa: ARG002
        """Make a deepcopy of the BluemiraPlacement"""
        return self.copy()

    @classmethod
    def _create(cls, obj: cadapi.apiPlacement, label: str = "") -> BluemiraPlacement:
        """Create a placement from a cadapi Placement"""
        if isinstance(obj, cadapi.apiPlacement):
            placement = BluemiraPlacement(label=label)
            placement._shape = obj
            return placement

        raise TypeError(
            f"Only Base.Placement objects can be used to create a {cls} instance"
        )

    def mult_vec(self, vec: Iterable[float]) -> np.ndarray:
        """Transform a vector into the local placement"""
        return cadapi.vector_to_numpy(self._shape.multVec(cadapi.Base.Vector(vec)))

    def extract_plane(
        self, v1: Iterable[float], v2: Iterable[float], base: float | None = None
    ) -> BluemiraPlane:
        """
        Return a plane identified by two vector given in the self placement

        Parameters
        ----------
        v1:
            first reference vector
        v2:
            second reference vector
        base:
            output plane origin

        Returns
        -------
        A BluemiraPlane
        """
        if base is None:
            base = self.base

        p1 = self.mult_vec(v1)
        p2 = self.mult_vec(v2)

        return BluemiraPlane.from_3_points(base, p1, p2)

    def xy_plane(self):
        """Returns the corresponding placement xy plane"""
        return self.extract_plane(v1=np.array([1, 0, 0]), v2=np.array([0, 1, 0]))

    def yz_plane(self):
        """Returns the corresponding placement yz plane"""
        return self.extract_plane(v1=np.array([0, 1, 0]), v2=np.array([0, 0, 1]))

    def xz_plane(self):
        """Returns the corresponding placement xz plane"""
        return self.extract_plane(v1=np.array([1, 0, 0]), v2=np.array([0, 0, 1]))


XYZ = BluemiraPlacement(label="xyz")
YZX = BluemiraPlacement.from_matrix(
    matrix=np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    label="yzx",
)
XZY = BluemiraPlacement(axis=(1.0, 0.0, 0.0), angle=-90.0, label="xzy")

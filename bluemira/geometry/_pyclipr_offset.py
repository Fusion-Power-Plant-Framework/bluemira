# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Discretised offset operations used in case of failure in primitive offsetting.
"""

from __future__ import annotations

from copy import deepcopy
from enum import Enum, auto

import numpy as np
import numpy.typing as npt
import pyclipr

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.coordinates import Coordinates, rotation_matrix_v1v2
from bluemira.geometry.error import GeometryError

__all__ = ["offset_clipper"]


class OffsetClipperMethodType(Enum):
    """Enumeration of types of offset methods."""

    SQUARE = auto()
    ROUND = auto()
    MITER = auto()

    @classmethod
    def _missing_(cls, value: str | OffsetClipperMethodType) -> OffsetClipperMethodType:
        try:
            return cls[value.upper()]
        except KeyError:
            raise GeometryError(
                f"{cls.__name__} has no method {value}."
                f"please select from {(*cls._member_names_,)}"
            ) from None


def pyclippath_to_coordinates(path: np.ndarray) -> Coordinates:
    """
    Transforms a pyclipper path into a bluemira Coordinates object

    Parameters
    ----------
    path:
        The vertex polygon path formatting used in pyclipper

    Returns
    -------
    The Coordinates from the path object
    """
    # p2 = scale_from_clipper(np.array(path).T)
    return Coordinates({"x": path[0], "y": 0, "z": path[1]})


class PyCliprOffsetter:
    def __init__(
        self,
        coordinates: Coordinates,
        method: OffsetClipperMethodType,
        miter_limit: float = 2.0,
    ):
        if not coordinates.is_planar:
            raise GeometryError("Cannot offset non-planar coordinates.")

        if not coordinates.closed:
            raise GeometryError(
                "Open Coordinates are not supported by PyCliprOffsetter."
            )

        coordinates = deepcopy(coordinates)
        com = coordinates.center_of_mass

        t_coordinates = transform_coordinates_to_xz(
            coordinates, tuple(-np.array(com)), np.array([0.0, 1.0, 0.0])
        )
        clipr_path = t_coordinates.xz.T

        self._coord_scale = self._calculate_scale(clipr_path, coordinates)
        self._coordinates = coordinates

        # Create an offsetting object
        pco = pyclipr.ClipperOffset()

        pco.miterLimit = miter_limit
        # Set the scale factor to convert to internal integer representation
        pco.scaleFactor = 1000  # ?

        match method:
            case OffsetClipperMethodType.SQUARE:
                pco.addPaths(
                    [clipr_path], pyclipr.JoinType.Square, pyclipr.EndType.Polygon
                )
            case OffsetClipperMethodType.ROUND:
                pco.addPaths(
                    [clipr_path], pyclipr.JoinType.Round, pyclipr.EndType.Polygon
                )
            case OffsetClipperMethodType.MITER:
                pco.addPaths(
                    [clipr_path], pyclipr.JoinType.Miter, pyclipr.EndType.Polygon
                )
        self._pco = pco

    @staticmethod
    def _calculate_scale(path: np.ndarray, coordinates: Coordinates) -> float:
        """
        Calculate the pyclipper scaling to integers
        """
        # Find the first non-zero dimension (low number of iterations)
        for i in range(len(path) - 1):
            if path[i][0] != 0:
                return path[i][0] / coordinates.x[i]
            if path[i][1] != 0:
                return path[i][1] / coordinates.z[i]
        raise GeometryError(
            "Could not calculate scale factor for pyclipper. "
            "Path is empty or only (0, 0)'s."
        )

    def _transform_offset_result(
        self, result: list[npt.NDArray[np.float64]]
    ) -> list[Coordinates]:
        """
        Transforms the offset solution into a Coordinates object
        """
        if not result:
            raise GeometryError("Offset operation resulted in no geometry.")

        com = self._coordinates.center_of_mass
        norm_v = self._coordinates.normal_vector

        res_coords_t = [pyclippath_to_coordinates(p) for p in result]
        return [transform_coordinates_to_original(c, com, norm_v) for c in res_coords_t]

    def perform(self, delta: float) -> list[Coordinates]:
        delta = int(round(delta * self._coord_scale))
        offset_result = self._pco.execute(delta)
        return self._transform_offset_result(offset_result)


def offset_clipper(
    coordinates: Coordinates,
    delta: float,
    method: str = "square",
    miter_limit: float = 2.0,
) -> Coordinates:
    """
    Carries out an offset operation on the Coordinates using the ClipperLib library.
    Only supports closed Coordinates.

    Parameters
    ----------
    coordinates:
        The Coordinates upon which to perform the offset operation
    delta:
        The value of the offset [m]. Positive for increasing size, negative for
        decreasing
    method:
        The type of offset to perform ['square', 'round', 'miter']
    miter_limit:
        The ratio of delta to use when mitering acute corners. Only used if
        method == 'miter'

    Returns
    -------
    The offset Coordinates result

    Raises
    ------
    GeometryError:
        If the Coordinates are not planar
        If the Coordinates are not closed
    """
    tool = PyCliprOffsetter(coordinates, OffsetClipperMethodType(method), miter_limit)
    result = tool.perform(delta)

    if len(result) > 1:
        bluemira_warn(
            f"Offset operation with delta={delta} has produced multiple 'islands'; only"
            " returning the biggest one!"
        )

    return result[0]


def transform_coordinates_to_xz(
    coordinates: Coordinates, base: tuple[float, float, float], direction: np.ndarray
) -> Coordinates:
    """
    Rotate coordinates to the x-z plane.
    """
    coordinates.translate(base)
    if abs(coordinates.normal_vector[1]) == 1.0:
        return coordinates

    r = rotation_matrix_v1v2(coordinates.normal_vector, np.array(direction))
    x, y, z = r.T @ coordinates

    return Coordinates({"x": x, "y": y, "z": z})


def transform_coordinates_to_original(
    coordinates: Coordinates,
    base: tuple[float, float, float],
    original_normal: np.ndarray,
) -> Coordinates:
    """
    Rotate coordinates back to original plane
    """
    r = rotation_matrix_v1v2(coordinates.normal_vector, np.array(original_normal))
    x, y, z = r.T @ coordinates
    coordinates = Coordinates({"x": x, "y": y, "z": z})
    coordinates.translate(base)
    return coordinates

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
from pyclipr import ClipperOffset, EndType, JoinType

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


def offset_clipper(
    coordinates: Coordinates,
    delta: float,
    method: str | OffsetClipperMethodType = "square",
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
    method = OffsetClipperMethodType(method)
    tool = PyCliprOffsetter(method, miter_limit)
    return tool.offset(coordinates, delta)


class PyCliprOffsetter:
    def __init__(
        self,
        method: OffsetClipperMethodType,
        miter_limit: float = 2.0,
    ):
        self.miter_limit = miter_limit
        self.offset_scale = 1  # ? what to set to
        match method:
            case OffsetClipperMethodType.SQUARE:
                self._jt = JoinType.Square
                self._et = EndType.Joined
            case OffsetClipperMethodType.ROUND:
                self._jt = JoinType.Round
                self._et = EndType.Round
            case OffsetClipperMethodType.MITER:
                self._jt = JoinType.Miter
                self._et = EndType.Joined

    @staticmethod
    def _transform_coords_to_path(coords: Coordinates) -> np.ndarray:
        com = coords.center_of_mass

        coords_t = transform_coordinates_to_xz(
            coords, tuple(-np.array(com)), np.array([0.0, 1.0, 0.0])
        )

        return coordinates_to_pyclippath(coords_t)

    @staticmethod
    def _transform_offset_result_to_orig(
        orig_coords: Coordinates, result: npt.NDArray[np.float64]
    ) -> Coordinates:
        """
        Transforms the offset solution into a Coordinates object
        """
        orig_com = orig_coords.center_of_mass
        orig_norm_v = orig_coords.normal_vector

        res_coords_t = pyclippath_to_coordinates(result)

        return transform_coordinates_to_original(res_coords_t, orig_com, orig_norm_v)

    def _perform_offset(
        self, path: npt.NDArray[np.float64], delta: float
    ) -> npt.NDArray[np.float64]:
        # Create an offsetting object
        pco = ClipperOffset()

        # causes it to error
        # pco.miterLimit = self.miter_limit
        pco.scaleFactor = int(self.offset_scale)

        pco.addPath(path, self._jt, self._et)
        offset_result = pco.execute(delta)

        if len(offset_result) == 1:
            offset_result = offset_result[0]
        elif len(offset_result) > 1:
            bluemira_warn(
                f"Offset operation with delta={delta} has produced multiple 'islands';"
                " using the biggest one!"
            )
            offset_result = max(offset_result, key=len)
        else:
            raise GeometryError("Offset operation failed to produce any geometry.")

        return offset_result

    def offset(self, orig_coords: Coordinates, delta: float) -> Coordinates:
        if not orig_coords.is_planar:
            raise GeometryError("Cannot offset non-planar coordinates.")

        if not orig_coords.closed:
            raise GeometryError(
                "Open Coordinates are not supported by PyCliprOffsetter."
            )

        used_coords = deepcopy(orig_coords)

        path = self._transform_coords_to_path(used_coords)
        offset_path = self._perform_offset(path, delta)

        return self._transform_offset_result_to_orig(orig_coords, offset_path)


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


def pyclippath_to_coordinates(path: np.ndarray, *, close=True) -> Coordinates:
    """
    Transforms a pyclipper path into a bluemira Coordinates object

    Parameters
    ----------
    path:
        The vertex polygon path formatting used in pyclipper
    close:
        Whether to close the path

    Returns
    -------
    The Coordinates from the path object
    """
    x, z = path.T
    if close:
        x = np.append(x, x[0])
        z = np.append(z, z[0])
    return Coordinates({"x": x, "y": np.zeros(x.shape), "z": z})


def coordinates_to_pyclippath(coords: Coordinates) -> np.ndarray:
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
    return coords.xz.T

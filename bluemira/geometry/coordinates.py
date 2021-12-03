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
Utility for sets of coordinates
"""

from typing import Iterable
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.constants import EPS
from bluemira.geometry.error import CoordinatesError
from bluemira.geometry._deprecated_tools import (
    get_perimeter_3d,
    get_area_3d,
    get_centroid_3d,
)


class Coordinates:
    """
    Coordinates object for storing ordered sets of coordinates.

    The following are enforced by default:
    * Shape of (3, N)
    * Counter-clockwise direction

    Parameters
    ----------
    xyz_array: Union[np.ndarray, dict, Iterable[Iterable]]

    Notes
    -----
    This is a utility class for dealing with sets of coordinates in a number of different
    contexts. It should not be used for the creation of CAD geometries.
    """

    __slots__ = ("_array",)
    # =============================================================================
    # Instantiation
    # =============================================================================
    def __init__(self, xyz_array, enforce_ccw=False):
        self._array = self._parse_input(xyz_array)

        # if not self.ccw and enforce_ccw:
        #     self.reverse()

    def _parse_input(self, xyz_array):
        if isinstance(xyz_array, np.ndarray):
            xyz_array = self._parse_array(xyz_array)
        elif isinstance(xyz_array, dict):
            xyz_array = self._parse_dict(xyz_array)
        elif isinstance(xyz_array, Iterable):
            xyz_array = self._parse_iterable(xyz_array)
        else:
            raise CoordinatesError(
                f"Cannot instantiate Coordinates with: {type(xyz_array)}"
            )
        return xyz_array

    def _parse_array(self, xyz_array):
        try:
            xyz_array = np.array(np.atleast_2d(xyz_array), dtype=np.float64)
        except ValueError:
            raise CoordinatesError(
                "Cannot instantiate Coordinates with a ragged (3, N | M) array."
            )

        shape = xyz_array.shape
        if len(shape) > 2:
            raise NotImplementedError

        n, m = shape
        if n == 3:
            if m == 3:
                bluemira_warn(
                    "You are creating Coordinates with a (3, 3) array, defaulting to (3, N)."
                )

        elif m == 3:
            xyz_array = xyz_array.T

        else:
            raise CoordinatesError(
                "Cannot instantiate Coordinates where either n or m != 3."
            )

        if not np.allclose([len(xyz_array[i]) for i in range(3)], len(xyz_array[0])):
            raise CoordinatesError(
                "Cannot instantiate Coordinates with a ragged (3, N | M) array."
            )

        return xyz_array

    def _parse_dict(self, xyz_dict):
        x = np.atleast_1d(xyz_dict.get("x", 0))
        y = np.atleast_1d(xyz_dict.get("y", 0))
        z = np.atleast_1d(xyz_dict.get("z", 0))

        shape_lengths = np.array([len(c.shape) for c in [x, y, z]])

        if np.any(shape_lengths > 1):
            raise CoordinatesError(
                "Cannot instantiate Coordinates from dict with coordinate vectors that are not 1-D."
            )

        lengths = [len(c) for c in [x, y, z]]

        usable_lengths = []
        for length in lengths:
            if length != 1:
                usable_lengths.append(length)

        if not np.allclose(usable_lengths, usable_lengths[0]):
            raise CoordinatesError(
                "Cannot instantiate Coordinate from dict with a ragged set of vectors."
            )

        # Backfill single-value coordinates
        actual_length = usable_lengths[0]
        if len(x) == 1:
            x = x[0] * np.ones(actual_length)
        if len(y) == 1:
            y = y[0] * np.ones(actual_length)
        if len(z) == 1:
            z = z[0] * np.ones(actual_length)

        return np.array([x, y, z])

    def _parse_iterable(self, xyz_iterable):
        # We temporarily set the dtype to object to avoid a VisibleDeprecationWarning
        xyz_array = np.array(xyz_iterable, dtype=object)
        return self._parse_array(xyz_array)

    def _parse_dtype(self, xyz_array):
        return np.array(xyz_array, dtype=np.float64)

    # =============================================================================
    # Checks
    # =============================================================================

    def _check_ccw(self):
        pass

    # =============================================================================
    # Property access
    # =============================================================================

    @property
    def x(self):
        return self._array[0]

    @property
    def y(self):
        return self._array[1]

    @property
    def z(self):
        return self._array[2]

    @property
    def xy(self):
        return self._array[[0, 1], :]

    @property
    def xz(self):
        return self._array[[0, 2], :]

    @property
    def yz(self):
        return self._array[[1, 2], :]

    # =========================================================================
    # Conversions
    # =========================================================================

    def as_dict(self):
        """
        Cast the Coordinates as a dictionary.

        Returns
        -------
        d: dict
            Dictionary with {'x': [], 'y': [], 'z':[]}
        """
        return {"x": self.x, "y": self.y, "z": self.z}

    # =============================================================================
    # Useful properties
    # =============================================================================

    @property
    def closed(self) -> bool:
        """
        Whether or not this is a closed set of Coordinates.
        """
        if len(self) > 2:
            if np.allclose(self[:, 0], self[:, -1], rtol=0, atol=EPS):
                return True
        return False

    @property
    def ccw(self) -> bool:
        """
        Whether or not the Coordinates are ordered in the counter-clockwise direction.
        """
        return True

    @property
    def length(self) -> float:
        """
        Perimeter length of the coordinates.
        """
        return get_perimeter_3d(*self._array)

    @property
    def area(self) -> float:
        """
        Enclosed area of the Coordinates. 0 if the Coordinates are open.
        """
        if not self.closed:
            return 0.0
        return get_area_3d(*self._array)

    @property
    def center_of_mass(self) -> tuple:
        """
        Geometrical centroid of the Coordinates.
        """
        # [sic] coordinates do not have a "mass", but named such for consistency with
        # other geometry objects.
        return tuple(get_centroid_3d(*self._array))

    # =============================================================================
    # Array-like behaviour
    # =============================================================================

    @property
    def T(self):  # noqa(N802)
        """
        Transpose of the Coordinates
        """
        return self._array.T

    @property
    def shape(self):
        """
        Shape of the Coordinates
        """
        return self._array.shape

    # =============================================================================
    # Modification
    # =============================================================================
    def reverse(self):
        """
        Reverse the direction of the Coordinates.
        """
        self._array = self._array[:, ::-1]

    def open(self):
        """
        Open the Coordinates (if they are closed)
        """
        if self.closed:
            self._array = self._array[:, :-1]

    def close(self):
        """
        Close the Coordinates (if they are open)
        """
        if not self.closed:
            self._array = np.vstack((self._array.T, self._array[:, 0])).T

    def rotate(self, base, direction, degree):
        """
        Rotate the Coordinates.

        Parameters
        ----------
        base: tuple (x,y,z)
            Origin location of the rotation
        direction: tuple (x,y,z)
            The direction vector
        degree: float
            rotation angle
        """
        pass

    def translate(self, vector):
        """
        Translate this shape with the vector. This function modifies the self
        object.
        """
        pass

    # =============================================================================
    # Dunders (with different behaviour to array)
    # =============================================================================

    def __eq__(self, other):
        """
        Check the Coordinates for equality with other Coordinates.

        Parameters
        ----------
        other: Coordinates
            The other Coordinates to compare against

        Returns
        -------
        equal: bool
            Whether or not the Coordinates are identical

        Notes
        -----
        Coordinates with identical coordinates but different orderings will not be
        counted as identical.
        """
        if isinstance(other, self.__class__):
            return np.all(np.allclose(self._array, other._array, rtol=0, atol=EPS))
        return False

    def __len__(self):
        """
        The number of points in the Coordinates.
        """
        return len(self._array[0])

    # =============================================================================
    # Array-like dunders
    # =============================================================================

    def __repr__(self):
        r = self._array.__repr__()
        return f"{self.__class__.__name__}{r[5:]}"

    def __getitem__(self, *args, **kwargs):
        return self._array.__getitem__(*args, **kwargs)

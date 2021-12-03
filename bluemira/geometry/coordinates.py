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


class Coordinates:
    """
    Coordinates object for storing ordered sets of coordinates.

    The following are enforced by default:
    * Shape of (3, N)
    * Counter-clockwise direction

    Parameters
    ----------

    Notes
    -----
    This is a utility class for dealing with sets of coordinates in a number of different
    contexts. It should not be used for the creation of CAD geometries.
    """

    __slots__ = ("_array",)
    # =============================================================================
    # Instantiation
    # =============================================================================
    def __init__(self, xyz_array, enforce_ccw=True):
        self._array = self._parse_input(xyz_array)

        # if not self._check_ccw() and enforce_ccw:
        #    self.reverse()

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
        shape = xyz_array.shape
        if len(shape) > 2:
            raise NotImplementedError

        n, m = shape
        if n == 3:
            if m == 3:
                bluemira_warn(
                    "You are creating Coordinates with a (3, 3) array, defaulting to (3, N)."
                )
            return xyz_array

        if m == 3:
            return xyz_array.T

    def _parse_dict(self, xyz_dict):
        x = np.atleast_1d(xyz_dict.get("x", 0))
        y = np.atleast_1d(xyz_dict.get("y", 0))
        z = np.atleast_1d(xyz_dict.get("z", 0))

        shape_lengths = [len(c.shape) for c in [x, y, z]]

        if any(shape_lengths > 1):
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

        actual_length = usable_lengths[0]
        if len(x) == 1:
            x = x[0] * np.ones(actual_length)
        if len(y) == 1:
            y = y[0] * np.ones(actual_length)
        if len(z) == 1:
            z = z[0] * np.ones(actual_length)

        return np.array([x, y, z])

    def _parse_iterable(self, xyz_iterable):
        xyz_array = np.array(xyz_iterable)
        return self._parse_array(xyz_array)

    # =============================================================================
    # Checks
    # =============================================================================

    def _check_dimension(self):
        pass

    def _check_ccw(self):
        pass

    def _check_closed(self):
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
    def length(self) -> float:
        pass

    @property
    def area(self) -> float:
        pass

    @property
    def center_of_mass(self) -> tuple:
        pass

    # =============================================================================
    # Array-like behaviour
    # =============================================================================

    @property
    def T(self):
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
        pass

    def open(self):
        pass

    def close(self):
        pass

    def rotate(self, base, direction, degree):
        pass

    def translate(self, dx=0, dy=0, dz=0):
        pass

    # =============================================================================
    # Dunders
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
        if isinstance(other, self.__class):
            return np.all(np.allclose(self._array, other._array), rtol=0, atol=EPS)
        return False

    def __repr__(self):
        r = self._array.__repr__()
        return f"{self.__class__.__name__}{r[5:]}"

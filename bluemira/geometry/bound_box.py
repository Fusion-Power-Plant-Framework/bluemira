# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Bounding box object
"""
from dataclasses import dataclass

import numpy as np

__all__ = ["BoundingBox"]


@dataclass
class BoundingBox:
    """
    Bounding box class

    Parameters
    ----------
    x_min: float
        Minimum x coordinate
    x_max: float
        Maximum x coordinate
    y_min: float
        Minimum y coordinate
    y_max: float
        Maximum y coordinate
    z_min: float
        Minimum z coordinate
    z_max: float
        Maximum z coordinate
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self):
        """
        Perform some silent sanity checks.
        """
        if self.x_min > self.x_max:
            self.x_min, self.x_max = self.x_max, self.x_min
        if self.y_min > self.y_max:
            self.y_min, self.y_max = self.y_max, self.y_min
        if self.z_min > self.z_max:
            self.z_min, self.z_max = self.z_max, self.z_min

    @classmethod
    def from_xyz(cls, x, y, z):
        """
        Create a BoundingBox from a set of coordinates

        Parameters
        ----------
        x: np.ndarray
            x coordinates from which to create the bounding box
        y: np.ndarray
            y coordinates from which to create the bounding box
        z: np.ndarray
            z coordinates from which to create the bounding box
        """
        x_max, x_min = np.max(x), np.min(x)
        y_max, y_min = np.max(y), np.min(y)
        z_max, z_min = np.max(z), np.min(z)
        return cls(x_min, x_max, y_min, y_max, z_min, z_max)

    def get_box_arrays(self):
        """
        Get the x, y, z arrays of the BoundingBox in space.

        Returns
        -------
        x_b: np.ndarray
            x coordinates of the BoundingBox in space
        y_b: np.ndarray
            y coordinates of the BoundingBox in space
        z_b: np.ndarray
            z coordinates of the BoundingBox in space
        """
        size = max(
            [self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min]
        )

        x_b = 0.5 * size * np.array([-1, -1, -1, -1, 1, 1, 1, 1]) + 0.5 * (
            self.x_max + self.x_min
        )
        y_b = 0.5 * size * np.array([-1, -1, 1, 1, -1, -1, 1, 1]) + 0.5 * (
            self.y_max + self.y_min
        )
        z_b = 0.5 * size * np.array([-1, 1, -1, 1, -1, 1, -1, 1]) + 0.5 * (
            self.z_max + self.z_min
        )
        return x_b, y_b, z_b

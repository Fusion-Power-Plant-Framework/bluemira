# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Bounding box object
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BoundingBox"]


@dataclass
class BoundingBox:
    """
    Bounding box class

    Parameters
    ----------
    x_min:
        Minimum x coordinate
    x_max:
        Maximum x coordinate
    y_min:
        Minimum y coordinate
    y_max:
        Maximum y coordinate
    z_min:
        Minimum z coordinate
    z_max:
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
    def from_xyz(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> BoundingBox:
        """
        Create a BoundingBox from a set of coordinates

        Parameters
        ----------
        x:
            x coordinates from which to create the bounding box
        y:
            y coordinates from which to create the bounding box
        z:
            z coordinates from which to create the bounding box
        """
        x_max, x_min = np.max(x), np.min(x)
        y_max, y_min = np.max(y), np.min(y)
        z_max, z_min = np.max(z), np.min(z)
        return cls(x_min, x_max, y_min, y_max, z_min, z_max)

    def get_box_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the x, y, z arrays of the BoundingBox in space.

        Returns
        -------
        x_b:
            x coordinates of the BoundingBox in space
        y_b:
            y coordinates of the BoundingBox in space
        z_b:
            z coordinates of the BoundingBox in space
        """
        size = max([
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        ])

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

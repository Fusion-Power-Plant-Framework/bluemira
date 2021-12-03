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

import numpy as np


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

    def __init__(self, array):
        pass

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
        pass

    @property
    def y(self):
        pass

    @property
    def z(self):
        pass

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

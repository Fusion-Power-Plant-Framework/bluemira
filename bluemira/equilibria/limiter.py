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
Limiter object class
"""

from itertools import cycle
from typing import Optional, Union

import numpy as np
from matplotlib.pyplot import Axes

from bluemira.equilibria.plotting import LimiterPlotter

__all__ = ["Limiter"]


class Limiter:
    """
    A set of discrete limiter points.

    Parameters
    ----------
    x:
        The x coordinates of the limiter points
    z:
        The z coordinates of the limiter points
    """

    __slots__ = ["x", "z", "xz", "_i"]

    def __init__(self, x: Union[float, np.ndarray], z: Union[float, np.ndarray]):
        self.x = x
        self.z = z
        self.xz = cycle(np.array([x, z]).T)
        self._i = 0

    def __iter__(self):
        """
        Hacky phoenix iterator
        """
        i = 0
        while i < len(self):
            yield next(self.xz)
            i += 1

    def __len__(self) -> int:
        """
        The length of the limiter.
        """
        return len(self.x)

    def __next__(self):
        """
        Hacky phoenix iterator
        """
        if self._i >= len(self):
            raise StopIteration
        self._i += 1
        return next(self.xz[self._i - 1])

    def plot(self, ax: Optional[Axes] = None):
        """
        Plots the Limiter object
        """
        return LimiterPlotter(self, ax)

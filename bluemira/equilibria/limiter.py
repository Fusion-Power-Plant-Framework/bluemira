# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Limiter object class
"""

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING

import numpy as np

from bluemira.equilibria.plotting import LimiterPlotter

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.pyplot import Axes

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

    __slots__ = ("_i", "x", "xz", "z")

    def __init__(self, x: npt.ArrayLike, z: npt.ArrayLike):
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

    def plot(self, ax: Axes | None = None) -> LimiterPlotter:
        """
        Plots the Limiter object
        """
        return LimiterPlotter(self, ax)

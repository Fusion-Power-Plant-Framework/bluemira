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
    from collections.abc import Iterator

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

    x: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    xz: Iterator[npt.NDArray[np.float64]]
    _i: int

    __slots__ = ("_i", "x", "xz", "z")

    def __init__(self, x: npt.ArrayLike, z: npt.ArrayLike):
        self.x = np.asarray(x, dtype=np.float64)
        self.z = np.asarray(z, dtype=np.float64)
        self.xz = cycle(np.array([self.x, self.z]).T)
        self._i = 0

    def __iter__(self) -> Iterator[npt.NDArray]:
        """
        Hacky phoenix iterator

        Yields
        ------
        :
            next element of xz
        """
        i = 0
        while i < len(self):
            yield next(self.xz)
            i += 1

    def __len__(self) -> int:
        """
        The length of the limiter.
        """  # noqa: DOC201
        return len(self.x)

    def __next__(self):
        """
        Hacky phoenix iterator

        Returns
        -------
        :
            The xz coordinates

        Raises
        ------
        StopIteration
            stop iterating at end of object
        """
        if self._i >= len(self):
            raise StopIteration
        self._i += 1
        return next(self.xz)

    def plot(self, ax: Axes | None = None) -> LimiterPlotter:
        """
        Plots the Limiter object

        Returns
        -------
        :
            The plot axis
        """
        return LimiterPlotter(self, ax)

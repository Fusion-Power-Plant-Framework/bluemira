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
Limiter object class
"""
from BLUEPRINT.equilibria.plotting import LimiterPlotter
from itertools import cycle
from copy import deepcopy


class Limiter:
    """
    Represents a set of limiters

    Parameters
    ----------
    locations: List of (x, z) tuples
        The locations of the limiters
    """

    def __init__(self, locations):
        x, z = [], []
        for loc in locations:
            x.append(loc[0])
            z.append(loc[1])
        self.x = x
        self.z = z
        self.len = len(locations)
        self.locs = cycle(locations)
        self._i = 0

    def __iter__(self):
        """
        Hacky phoenix iterator
        """
        i = 0
        while i < self.len:
            yield next(self.locs)
            i += 1
        i = 0

    def __len__(self):
        """
        The length of the limiter.
        """
        return self.len

    def __next__(self):
        """
        Hacky phoenix iterator
        """
        if self._i >= len(self.locs):
            raise StopIteration
        self._i += 1
        return next(self.locs[self._i - 1])

    def plot(self, ax=None):
        """
        Plots the Limiter object
        """
        return LimiterPlotter(self, ax)

    def copy(self):
        """
        Get a deep copy of the Limiter object.
        """
        return deepcopy(self)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()

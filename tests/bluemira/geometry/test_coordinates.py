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

import numpy as np

from bluemira.geometry.error import CoordinatesError
from bluemira.geometry.coordinates import Coordinates


class TestCoordinates:
    def test_array_init(self):
        xyz = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 2, 3]])

        c1 = Coordinates(xyz)

        c2 = Coordinates(xyz.T)

        assert c1 == c2

    def test_dict_init(self):
        pass

    def test_iterable_init(self):
        pass

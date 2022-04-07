# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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

import json
import os

import matplotlib.pyplot as plt
import pytest

import tests
from bluemira.base.file import get_bluemira_path
from bluemira.geometry._deprecated_offset import offset_clipper
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError


class TestClipperOffset:
    plot = tests.PLOTTING

    # fmt: off
    x_open = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2]
    y_open = [0, -2, -4, -3, -4, -2, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 4, 3, 2, 1, 2, 2, 1]
    x_closed = x_open + [-3, -4, -3]
    y_closed = y_open + [1, 0, -2]
    # fmt: on

    @pytest.mark.parametrize("x, y", [(x_open, y_open), (x_closed, y_closed)])
    def test_complex_open(self, x, y):
        coordinates = Coordinates({"x": x, "y": y, "z": 0})
        c = offset_clipper(coordinates, 1)
        if self.plot:
            f, ax = plt.subplots()
            ax.plot(x, y, "k")
            ax.plot(c.x, c.y, "r", marker="o")
            ax.set_aspect("equal")

    def test_blanket_offset(self):
        fp = get_bluemira_path("bluemira/geometry/test_data", subfolder="tests")
        fn = os.sep.join([fp, "bb_offset_test.json"])
        with open(fn, "rb") as file:
            data = json.load(file)
        coordinates = Coordinates(data)
        offsets = []
        for m in ["square", "miter"]:  # round very slow...
            offset_coordinates = offset_clipper(coordinates, 1.5, method=m)
            offsets.append(offset_coordinates)

        if self.plot:
            f, ax = plt.subplots()
            ax.plot(coordinates.x, coordinates.z, color="k")
            colors = ["r", "g", "y"]
            for offset_coordinates, c in zip(offsets, colors):
                ax.plot(offset_coordinates.x, offset_coordinates.z, color=c)

    def test_raise_error(self):
        coordinates = Coordinates({"x": [0, 1, 2], "y": [0, 0, 0]})
        with pytest.raises(GeometryError):
            offset_clipper(coordinates, 1, method="fail")

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
Created on Fri Aug  2 07:33:39 2019

@author: matti
"""
import os
import pickle  # noqa :S403

import numpy as np
import pytest
from matplotlib import pyplot as plt

import tests
from bluemira.base.file import get_bluemira_path
from bluemira.geometry.error import GeometryError
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.offset import offset, offset_clipper


class TestOffset:
    plot = tests.PLOTTING

    @classmethod
    def setup_class(cls):
        pass

    def test_rectangle(self):
        # Rectangle - positive offset
        x = [1, 3, 3, 1, 1, 3]
        y = [1, 1, 3, 3, 1, 1]
        o = offset(x, y, 0.25)
        assert sum(o[0] - np.array([0.75, 3.25, 3.25, 0.75, 0.75])) == 0
        assert sum(o[1] - np.array([0.75, 0.75, 3.25, 3.25, 0.75])) == 0
        if self.plot:
            f, ax = plt.subplots()
            ax.plot(x, y, "k")
            ax.plot(*o, "r", marker="o")
            ax.set_aspect("equal")

    def test_triangle(self):
        x = [1, 2, 1.5, 1, 2]
        y = [1, 1, 4, 1, 1]
        t = offset(x, y, -0.25)
        assert (
            abs(sum(t[0] - np.array([1.29511511, 1.70488489, 1.5, 1.29511511])) - 0)
            < 1e-3
        )
        assert abs(sum(t[1] - np.array([1.25, 1.25, 2.47930937, 1.25])) - 0) < 1e-3
        if self.plot:
            f, ax = plt.subplots()
            ax.plot(x, y, "k")
            ax.plot(*t, "r", marker="o")
            ax.set_aspect("equal")

    def test_complex_open(self):
        x = [
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
            -1,
            -2,
        ]
        y = [
            0,
            -2,
            -4,
            -3,
            -4,
            -2,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            6,
            6,
            6,
            4,
            3,
            2,
            1,
            2,
            2,
            1,
        ]
        c = offset(x, y, 1)
        if self.plot:
            f, ax = plt.subplots()
            ax.plot(x, y, "k")
            ax.plot(*c, "r", marker="o")
            ax.set_aspect("equal")

    def test_complex_closed(self):
        x = [
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
            -1,
            -2,
            -3,
            -4,
            -3,
        ]
        y = [
            0,
            -2,
            -4,
            -3,
            -4,
            -2,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            6,
            6,
            6,
            4,
            3,
            2,
            1,
            2,
            2,
            1,
            1,
            0,
            -2,
        ]
        c = offset(x, y, 1)
        if self.plot:
            f, ax = plt.subplots()
            ax.plot(x, y, "k")
            ax.plot(*c, "r", marker="o")
            ax.set_aspect("equal")


class TestClipperOffset:
    plot = tests.PLOTTING

    def test_complex_open(self):
        x = [
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
            -1,
            -2,
        ]
        y = [
            0,
            -2,
            -4,
            -3,
            -4,
            -2,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            6,
            6,
            6,
            4,
            3,
            2,
            1,
            2,
            2,
            1,
        ]
        loop = Loop(x=x, y=y)
        c = offset_clipper(loop, 1)
        if self.plot:
            f, ax = plt.subplots()
            ax.plot(x, y, "k")
            ax.plot(c.x, c.y, "r", marker="o")
            ax.set_aspect("equal")

    def test_complex_closed(self):
        x = [
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
            -1,
            -2,
            -3,
            -4,
            -3,
        ]
        y = [
            0,
            -2,
            -4,
            -3,
            -4,
            -2,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            6,
            6,
            6,
            4,
            3,
            2,
            1,
            2,
            2,
            1,
            1,
            0,
            -2,
        ]
        loop = Loop(x=x, y=y)
        c = offset_clipper(loop, 1)
        if self.plot:
            f, ax = plt.subplots()
            ax.plot(x, y, "k")
            ax.plot(c.x, c.y, "r", marker="o")
            ax.set_aspect("equal")

    def test_blanket_offset(self):
        fp = get_bluemira_path("BLUEPRINT/geometry/test_data", subfolder="tests")
        fn = os.sep.join([fp, "bb_offset_test.pkl"])
        with open(fn, "rb") as file:
            d = pickle.load(file)  # noqa :S301
        loop = Loop(**d)
        offsets = []
        for m in ["square", "miter"]:  # round very slow...
            offset_loop = offset_clipper(loop, 1.5, method=m)
            offsets.append(offset_loop)
        f, ax = plt.subplots()
        loop.plot(ax)
        colors = ["r", "g", "y"]
        for offset_loop, c in zip(offsets, colors):
            offset_loop.plot(ax, facecolor=c)

    def test_raise(self):
        x = [
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
            -1,
            -2,
            -3,
            -4,
            -3,
        ]
        y = [
            0,
            -2,
            -4,
            -3,
            -4,
            -2,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            6,
            6,
            6,
            4,
            3,
            2,
            1,
            2,
            2,
            1,
            1,
            0,
            -2,
        ]
        loop = Loop(x=x, y=y)
        with pytest.raises(GeometryError):
            offset_clipper(loop, 1, method="fail")

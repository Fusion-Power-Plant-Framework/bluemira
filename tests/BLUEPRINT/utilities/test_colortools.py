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
Created on Thu Aug  1 21:50:44 2019

@author: matti
"""

import pytest
from matplotlib import pyplot as plt

import tests
from BLUEPRINT.utilities.colortools import shift_rgb_color


@pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
class TestshiftRGB:
    def test_shift(self):
        from BLUEPRINT.geometry.loop import Loop

        f, ax = plt.subplots()
        loop1 = Loop(x=[0, 1, 1, 0, 0], z=[0, 0, 1, 1, 0])
        loop2 = loop1.translate([1, 0, 0], update=False)
        loop3 = loop1.translate([2, 0, 0], update=False)
        loop4 = loop1.translate([0, 0, 1], update=False)
        loop5 = loop4.translate([1, 0, 0], update=False)
        loop6 = loop4.translate([2, 0, 0], update=False)
        c1 = (0.2909803921568628, 0.5945098039215686, 0.7890196078431373)
        c2 = shift_rgb_color(c1, 0.1)
        c3 = shift_rgb_color(c1, 0.2)
        c4 = c1
        c5 = shift_rgb_color(c4, -0.1)
        c6 = shift_rgb_color(c4, -0.2)
        loop1.plot(ax, facecolor=c1)
        loop2.plot(ax, facecolor=c2)
        loop3.plot(ax, facecolor=c3)
        loop4.plot(ax, facecolor=c4)
        loop5.plot(ax, facecolor=c5)
        loop6.plot(ax, facecolor=c6)

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
Created on Fri Aug  2 20:29:07 2019

@author: matti
"""

import pytest

from bluemira.base.look_and_feel import _print_color, bluemira_print, bluemira_warn
from BLUEPRINT.base.palettes import LONDON
from BLUEPRINT.utilities.colortools import map_palette


class TestLookAndFeel:
    def test_map_pal(self):
        pal = {
            "Plasma": "Hammersmith and City",
            "BB": ["Bakerloo", "Circle"],
            "DIV": "Overground",
            "VV": "District",
            "TS": "DLR",
            "TF": ["Piccadilly", "Victoria"],
            "PF": ["Metropolitan", "Central"],
            "CR": "Jubilee",
            "RS": "Tramlink",
        }
        palnew = map_palette(pal, LONDON)
        assert palnew == {
            "BB": ["#B36305", "#FFD300"],
            "CR": "#A0A5A9",
            "DIV": "#EE7C0E",
            "PF": ["#9B0056", "#E32017"],
            "Plasma": "#F3A9BB",
            "RS": "#84B817",
            "TF": ["#003688", "#0098D4"],
            "TS": "#00A4A7",
            "VV": "#00782A",
        }


@pytest.mark.longrun
class TestColors:
    def test_single(self):
        print("\n")
        colors = [
            "white",
            "red",
            "green",
            "orange",
            "blue",
            "purple",
            "cyan",
            "lightgrey",
            "darkgrey",
            "lightred",
            "lightgreen",
            "yellow",
            "lightblue",
            "pink",
            "lightcyan",
        ]
        for color in colors:
            print(_print_color(color, color))

    def test_warn(self):
        print("\n")
        bluemira_warn("test warning")

    def test_bluemira_print(self):
        print("\n")
        bluemira_print("test normal blue")

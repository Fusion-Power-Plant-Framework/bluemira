# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

import os
import pytest
from BLUEPRINT.utilities.colortools import map_palette
from BLUEPRINT.base.palettes import LONDON
from BLUEPRINT.base.file import get_BP_path, get_BP_root, get_PROCESS_root
from BLUEPRINT.base.lookandfeel import (
    banner,
    user_banner,
    count_slocs,
    get_git_branch,
    _print_color,
    bpwarn,
    bpinfo,
    bprint,
)


class TestLookAndFeel:
    @pytest.mark.longrun
    def test_banner(self):
        banner()

    @pytest.mark.longrun
    def test_user_banner(self):
        user_banner()

    @pytest.mark.longrun
    def test_count_slocs(self):
        directory = get_BP_root()
        count_slocs(directory, get_git_branch(directory))

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
        bpwarn("test warning")

    def test_into(self):
        print("\n")
        bpinfo("test info")

    def test_bprint(self):
        print("\n")
        bprint("test normal blue")


class TestMisc:
    def test_BP_root(self):  # noqa (N802)
        assert os.path.isdir(get_BP_root())

    def test_BP_path(self):  # noqa (N802)
        folders = ["systems", "materials", "geometry", "nova"]
        for p in folders:
            assert os.path.isdir(get_BP_path(p))
        fails = ["wrongwrong", "nopenope"]
        for f in fails:
            with pytest.raises(ValueError):
                get_BP_path(f)

    def test_PROCESS_path(self):  # noqa (N802)
        pw = pytest.importorskip(
            "process_io_lib.mfile"
        )  # Skip the test if PROCESS not installed
        assert os.path.isdir(get_PROCESS_root())


if __name__ == "__main__":
    pytest.main([__file__])

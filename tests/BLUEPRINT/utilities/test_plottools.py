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
Generic plot utilities, figure and gif operations
"""

import pytest
import os
import filecmp
import matplotlib.pyplot as plt
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.base.lookandfeel import plot_defaults
from BLUEPRINT.utilities.plottools import mathify, gsymbolify, SuperSankey


class TestMathify:
    def test_single(self):
        result = mathify("PF_1")
        assert result == "$PF_{1}$"

        result = mathify("I_m_p")
        assert result == "$I_{m_{p}}$"


class TestGsymbolify:
    def test_lowercase(self):
        string = gsymbolify("beta")
        assert string == "\\beta"

    def test_uppercase(self):
        string = gsymbolify("Beta")
        assert string == "\\Beta"

    def test_nothing(self):
        string = gsymbolify("nothing")
        assert string == "nothing"


class TestSuperSankey:
    def test_sankey_ring(self):
        plot_defaults(True)

        scale = 0.001
        gap = 0.25
        trunk_length = 0.0007 / scale
        l_standard = 0.0006 / scale  # standard arrow length
        l_medium = 0.001 / scale  # medium arrow length
        sankey = SuperSankey(scale=scale, gap=gap)
        sankey.add(
            "1",
            [1000, 500, -1500],
            orientations=[0, -1, 0],
            pathlengths=[l_medium, l_standard, l_medium],
        )
        sankey.add(
            "2",
            [1500, -1500],
            orientations=[0, -1],
            prior=0,
            connect=(2, 0),
            pathlengths=[l_medium, l_standard],
        )
        sankey.add(
            "3",
            [1500, -1000, -500],
            orientations=[0, 0, -1],
            prior=1,
            connect=(1, 0),
            pathlengths=[l_medium, l_standard, l_medium],
        )
        sankey.add(
            "4",
            [500, -500],
            orientations=[0, -1],
            prior=2,
            future=0,
            pathlengths=[l_medium, l_standard],
            connect=[(2, 0), (1, 1)],
        )

        sankey.finish()
        figure = plt.gcf()

        path = get_BP_path("BLUEPRINT/utilities/test_data", subfolder="tests")
        name_new = os.sep.join([path, "sankey_test_new.png"])
        figure.save_figure(name_new)
        name_old = os.sep.join([path, "sankey_test.png"])

        assert filecmp.cmp(name_new, name_old, shallow=False)


if __name__ == "__main__":
    pytest.main([__file__])

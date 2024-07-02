# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

from matplotlib.testing import compare as mpl_compare

from bluemira.balance_of_plant.plotting import SuperSankey
from bluemira.base.file import get_bluemira_path
from bluemira.display.auto_config import plot_defaults


class TestSuperSankey:
    def test_sankey_ring(self, tmp_path):
        plot_defaults(force=True)

        scale = 0.001
        gap = 0.25
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
        figure = sankey.ax.figure
        new_file = tmp_path / "sankey_test.png"
        figure.savefig(new_file)

        path = get_bluemira_path("balance_of_plant/test_data", subfolder="tests")
        reference_file = Path(path, "sankey_test.png")
        assert mpl_compare.compare_images(reference_file, new_file, 0.005) is None

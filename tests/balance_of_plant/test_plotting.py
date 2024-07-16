# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from numpy.testing import assert_allclose, assert_array_equal

from bluemira.balance_of_plant.plotting import SuperSankey
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
        sf = sankey.finish()[0]

        assert_array_equal(sf.flows, [1000, 500, -1500])
        assert sf.angles == [0, 1, 0]
        assert all(
            text.get_text() == f"{res}"
            for res, text in zip([1000, 500, 1500], sf.texts, strict=False)
        ), [
            text.get_text()
            for res, text in zip([1000, 500, 1500], sf.texts, strict=False)
        ]
        assert_allclose(
            sf.tips,
            [(-1.58045022, 0.25), (-0.75, -1.14022509), (1.90449771, 0.0)],
        )

        figure = sankey.ax.figure
        new_file = tmp_path / "sankey_test.svg"
        figure.savefig(new_file)

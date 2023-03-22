# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
from random import uniform

import pytest

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from eudemo.blanket import BlanketBuilder


class TestDivertorBuilder:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "n_bb_inboard": {"value": 2, "unit": "m"},
            "n_bb_outboard": {"value": 3, "unit": "m"},
            "c_rm": {"value": 0.02, "unit": "m"},
            "n_TF": {"value": 12, "unit": ""},
        }
        cls.silhouette = BluemiraFace(
            make_polygon(
                [
                    [1, 0, -2],
                    [1, 0, 10],
                    [5, 0, 10],
                    [5, 0, -2],
                    [4, 0, -2],
                    [4, 0, 9],
                    [2, 0, 9],
                    [2, 0, -2],
                ],
                closed=True,
            )
        )

    @pytest.mark.parametrize("cut_angle", [uniform(0, 90) for _ in range(2)])
    @pytest.mark.parametrize("r_inner_cut", [uniform(2, 4) for _ in range(2)])
    def test_components_and_segments(self, r_inner_cut, cut_angle):
        builder = BlanketBuilder(
            self.params, {}, self.silhouette, r_inner_cut, cut_angle
        )
        blanket = builder.build()

        assert blanket.get_component("xz")
        assert blanket.get_component("xy")

        xyz = blanket.get_component("xyz")
        assert xyz
        assert (
            len(xyz.leaves)
            == self.params["n_bb_inboard"]["value"]
            + self.params["n_bb_outboard"]["value"]
        )

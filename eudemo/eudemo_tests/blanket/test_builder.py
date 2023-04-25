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
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from eudemo.blanket.builder import BlanketBuilder


class TestBlanketBuilder:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "n_bb_inboard": {"value": 2, "unit": "m"},
            "n_bb_outboard": {"value": 3, "unit": "m"},
            "c_rm": {"value": 0.02, "unit": "m"},
            "n_TF": {"value": 12, "unit": ""},
        }
        cls.ib_silhouette = BluemiraFace(
            make_polygon(
                [
                    [1, 0, -2],
                    [1, 0, 10],
                    [2.9, 0, 10],
                    [2.9, 0, 9],
                    [2, 0, 9],
                    [2, 0, -2],
                    [1, 0, -2],
                ],
                closed=True,
            )
        )
        cls.ob_silhouette = BluemiraFace(
            make_polygon(
                [
                    [5, 0, -2],
                    [5, 0, 10],
                    [3.1, 0, 10],
                    [3.1, 0, 9],
                    [4, 0, 9],
                    [4, 0, -2],
                    [5, 0, -2],
                ],
                closed=True,
            )
        )

    def test_components_and_segments(self):
        builder = BlanketBuilder(
            self.params,
            build_config={},
            ib_silhouette=self.ib_silhouette,
            ob_silhouette=self.ob_silhouette,
        )
        blanket = builder.build()

        assert blanket.get_component("xz")
        assert blanket.get_component("xy")
        xyz = blanket.get_component("xyz")
        assert xyz
        xyz.show_cad()
        expected_num_leaves = (
            self.params["n_bb_inboard"]["value"] + self.params["n_bb_outboard"]["value"]
        )
        assert len(xyz.leaves) == expected_num_leaves

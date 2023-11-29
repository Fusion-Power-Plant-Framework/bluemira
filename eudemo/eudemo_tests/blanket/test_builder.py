# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
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

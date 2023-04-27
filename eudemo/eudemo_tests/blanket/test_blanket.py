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
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>
from eudemo.blanket import Blanket, BlanketBuilder
from eudemo_tests.blanket.tools import make_simple_blanket


def make_blanket_component():
    params = {
        "n_bb_inboard": {"value": 2, "unit": "m"},
        "n_bb_outboard": {"value": 3, "unit": "m"},
        "c_rm": {"value": 0.02, "unit": "m"},
        "n_TF": {"value": 12, "unit": ""},
    }
    segments = make_simple_blanket()
    builder = BlanketBuilder(
        params,
        build_config={},
        ib_silhouette=segments.inboard,
        ob_silhouette=segments.outboard,
    )
    return segments, builder.build()


class TestBlanket:
    def test_inboard_xz_silhouette_face_from_BlanketBuilder_component_tree(self):
        segments, component = make_blanket_component()

        blanket = Blanket(component_tree=component)
        ib_face = blanket.inboard_xz_silhouette()

        assert ib_face is segments.inboard

    def test_outboard_xz_silhouette_face_from_BlanketBuilder_component_tree(self):
        segments, component = make_blanket_component()

        blanket = Blanket(component_tree=component)
        ob_face = blanket.outboard_xz_silhouette()

        assert ob_face is segments.outboard

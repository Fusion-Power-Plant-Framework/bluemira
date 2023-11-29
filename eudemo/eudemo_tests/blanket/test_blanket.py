# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

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

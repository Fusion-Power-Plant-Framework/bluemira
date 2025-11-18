# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from eudemo.blanket import BlanketBuilder
from eudemo_tests.blanket.tools import make_simple_blanket


def make_blanket_component():
    params = {
        "n_bb_inboard": {"value": 2, "unit": "m"},
        "n_bb_outboard": {"value": 3, "unit": "m"},
        "tk_bb_fw_ib": {"value": 0.02, "unit": "m"},
        "tk_bb_fw_ob": {"value": 0.02, "unit": "m"},
        "tk_bb_bz_ib": {"value": 0.02, "unit": "m"},
        "tk_bb_bz_ob": {"value": 0.02, "unit": "m"},
        "c_rm": {"value": 0.02, "unit": "m"},
        "n_TF": {"value": 12, "unit": ""},
    }
    panel_points, segments = make_simple_blanket()
    builder = BlanketBuilder(
        params,
        build_config={},
        ib_silhouette=segments.inboard,
        ob_silhouette=segments.outboard,
        panel_points=panel_points,
    )
    return params, segments, builder.build()

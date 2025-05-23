# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from copy import deepcopy

import pytest

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import extrude_shape, make_polygon
from bluemira.radiation_transport.neutronics.dagmc import (
    DAGMCConverterFastCTDConfig,
    save_cad_to_dagmc,
)


class TestDAGMCConverter:
    @pytest.mark.parametrize(
        ("converter_config"),
        [
            DAGMCConverterFastCTDConfig(
                fix_step_to_brep_geometry=True,
            ),
            None,
        ],
    )
    @pytest.mark.parametrize(
        ("translate_x", "translate_y"),
        [
            (0.6, 0.6),
            (0.3, 0.5),
            (2.5, 2.5),
        ],
    )
    def test_dagmc_converter_fast_ctd(
        self,
        tmp_path,
        converter_config: DAGMCConverterFastCTDConfig | None,
        translate_x: float,
        translate_y: float,
    ) -> None:
        """
        Test the DAGMC converter using fast_ctd.
        """
        pytest.importorskip("fast_ctd")

        box_a = BluemiraFace(
            make_polygon(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], closed=True
            )
        )
        box_a = extrude_shape(box_a, [0, 0, 1])
        box_b = deepcopy(box_a)
        box_b.translate([translate_x, translate_y, 1])
        box_c = deepcopy(box_a)
        box_c.translate([-translate_x, -translate_y, 1])

        shapes = [box_a, box_b, box_c]
        names = ["box_a", "box_b", "box_c"]

        save_cad_to_dagmc(
            shapes=shapes,
            names=names,
            filename=tmp_path / "test.h5m",
            comp_mat_mapping={
                "box_a": "mat_a",
                "box_b": "mat_b",
                "box_c": "mat_c",
            },
            converter_config=converter_config,
        )

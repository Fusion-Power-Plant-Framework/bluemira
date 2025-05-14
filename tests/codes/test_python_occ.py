# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from copy import deepcopy

import pytest

from bluemira.codes import python_occ
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import extrude_shape, make_polygon


class TestImprintSolids:
    @pytest.mark.parametrize(
        ("translate_x", "translate_y", "t_imprints", "a_faces", "bc_faces"),
        [
            (0.6, 0.6, 2, 8, 7),
            (0.3, 0.5, 3, 9, 8),
            (2.5, 2.5, 0, 6, 6),
        ],
    )
    def test_imprint_solids(
        self,
        translate_x: float,
        translate_y: float,
        t_imprints: int,
        a_faces: int,
        bc_faces: int,
    ):
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

        pre_imps = [box_a, box_b, box_c]
        imp_result = python_occ.imprint_solids(pre_imps, use_cgal=True)

        imps = imp_result.imprintables
        imp_solids = imp_result.solids

        assert imp_result.total_imprints == t_imprints

        assert len(imp_solids) == 3

        assert len(imp_solids[0].faces) == a_faces
        assert imps[0]._has_imprinted if t_imprints > 0 else not imps[0]._has_imprinted
        assert len(imp_solids[1].faces) == bc_faces
        assert imps[1]._has_imprinted if t_imprints > 0 else not imps[0]._has_imprinted
        assert len(imp_solids[2].faces) == bc_faces
        assert imps[2]._has_imprinted if t_imprints > 0 else not imps[0]._has_imprinted

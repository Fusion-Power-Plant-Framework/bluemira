# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from copy import deepcopy

from bluemira.codes import python_occ
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import extrude_shape, make_polygon


class TestImprintSolids:
    def test_imprint_solids(self):
        box_a = BluemiraFace(
            make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
        )
        box_a = extrude_shape(box_a, [0, 0, 1])
        box_b = deepcopy(box_a)
        box_b.translate([-0.6, -0.6, 1])
        box_c = deepcopy(box_a)
        box_c.translate([0.6, 0.6, 1])

        pre_imps = [box_a, box_b, box_c]
        imp_result = python_occ.imprint_solids(pre_imps)

        imps = imp_result.imprintables
        imp_solids = imp_result.solids

        assert len(imp_solids) == 3
        assert len(imp_solids[0].faces) == 8
        assert imps[0]._has_imprinted
        assert len(imp_solids[1].faces) == 7
        assert imps[1]._has_imprinted
        assert len(imp_solids[2].faces) == 7
        assert imps[2]._has_imprinted

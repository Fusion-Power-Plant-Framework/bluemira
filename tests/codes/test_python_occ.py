# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from copy import deepcopy
from unittest import mock

import pytest

from bluemira.codes import python_occ
from bluemira.codes.python_occ import imprintable_solid as _imp
from bluemira.codes.python_occ.imprintable_solid import ImprintableSolid
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import extrude_shape, make_polygon


class TestImprintSolids:
    @pytest.mark.parametrize("use_cgal", [True, False])
    @pytest.mark.parametrize(
        ("translate_x", "translate_y", "t_imprints", "a_faces", "bc_faces"),
        [(0.6, 0.6, 2, 8, 7), (0.3, 0.5, 3, 9, 8), (2.5, 2.5, 0, 6, 6)],
    )
    def test_imprint_solids(
        self,
        use_cgal: bool,  # noqa: FBT001
        translate_x: float,
        translate_y: float,
        t_imprints: int,
        a_faces: int,
        bc_faces: int,
    ):
        pytest.importorskip("OCC")

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
        pre_imps_labels = ["box_a", "box_b", "box_c"]
        imp_result = python_occ.imprint_solids(
            pre_imps, labels=pre_imps_labels, use_cgal=use_cgal
        )

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

        # Rerun the tests mocking cgal as not available
        if use_cgal:
            with mock.patch(
                "bluemira.geometry.overlap_checking.cgal.cgal_available",
                return_value=False,
            ):
                imp_result = python_occ.imprint_solids(
                    pre_imps, labels=pre_imps_labels, use_cgal=use_cgal
                )

        assert imp_result.total_imprints == t_imprints
        assert len(imp_solids) == 3
        assert len(imp_solids[0].faces) == a_faces
        assert imps[0]._has_imprinted if t_imprints > 0 else not imps[0]._has_imprinted
        assert len(imp_solids[1].faces) == bc_faces
        assert imps[1]._has_imprinted if t_imprints > 0 else not imps[0]._has_imprinted
        assert len(imp_solids[2].faces) == bc_faces
        assert imps[2]._has_imprinted if t_imprints > 0 else not imps[0]._has_imprinted


@pytest.fixture
def bm_solid():
    face = BluemiraFace(
        make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], closed=True)
    )
    return extrude_shape(face, [0, 0, 1])


class TestImprintableSolid:
    """Unit tests for ``ImprintableSolid`` and its module-level helpers."""

    def test_from_bluemira_solid_rejects_wrong_type(self):
        pytest.importorskip("OCC")
        with pytest.raises(TypeError, match="bm_solid must be a BluemiraSolid"):
            ImprintableSolid.from_bluemira_solid("bad", "not a solid")

    def test_bm_shape_to_occ_solid_rejects_unknown_type(self):
        pytest.importorskip("OCC")
        with pytest.raises(TypeError, match="Cannot convert"):
            _imp._bm_shape_to_occ_solid(object())

    def test_to_bluemira_solid_returns_original_when_not_imprinted(self, bm_solid):
        pytest.importorskip("OCC")
        imp = ImprintableSolid.from_bluemira_solid("box", bm_solid)
        # ``set_imprinted_solid`` was never called, so the cached BluemiraSolid
        # is returned verbatim.
        assert imp.to_bluemira_solid() is bm_solid

    def test_set_imprinted_then_to_bluemira_solid_roundtrips(self, bm_solid):
        pytest.importorskip("OCC")
        imp = ImprintableSolid.from_bluemira_solid("box", bm_solid)
        # Re-feed the OCC solid we already produced — flips ``_has_imprinted``
        # and forces the round-trip back to a BluemiraSolid.
        imp.set_imprinted_solid(imp.occ_solid)
        out = imp.to_bluemira_solid()
        assert out is not bm_solid
        assert out.label == "box"
        assert out.volume == pytest.approx(bm_solid.volume, rel=1e-6)

    def test_bind_and_finalise_replaces_imprinted_faces(self, bm_solid):
        pytest.importorskip("OCC")
        imp = ImprintableSolid.from_bluemira_solid("box", bm_solid)
        original_faces = imp.imprinted_faces
        # ``bind_imprinted_face`` accumulates into the shadow set without
        # touching the live set; ``finalise_binding`` swaps them.
        marker = next(iter(original_faces))
        imp.bind_imprinted_face(marker)
        assert imp.imprinted_faces is original_faces
        assert imp._shadow_imprinted_faces == {marker}
        imp.finalise_binding()
        assert imp.imprinted_faces == {marker}
        assert imp._shadow_imprinted_faces == set()

    def test_label_property(self, bm_solid):
        pytest.importorskip("OCC")
        imp = ImprintableSolid.from_bluemira_solid("my_label", bm_solid)
        assert imp.label == "my_label"

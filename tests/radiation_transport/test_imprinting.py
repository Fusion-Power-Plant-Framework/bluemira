# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from copy import deepcopy
from unittest import mock

import pytest

from bluemira.codes import _geometryapi as cadapi
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import extrude_shape, make_polygon
from bluemira.radiation_transport.neutronics.dagmc import _imprinting


@pytest.fixture
def require_cadquery_backend():
    """Skip OCP imprinting tests unless CadQuery backend is active."""
    pytest.importorskip("OCP")
    cq = pytest.importorskip("cadquery")
    if cadapi.apiSolid is not cq.Solid:
        pytest.skip("Solid imprinting tests require the CadQuery geometry backend.")
    return cq


@pytest.fixture
def bm_solid() -> BluemiraSolid:
    face = BluemiraFace(
        make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], closed=True)
    )
    return extrude_shape(face, [0, 0, 1])


class TestImprintSolids:
    @pytest.mark.parametrize("use_cgal", [True, False])
    @pytest.mark.parametrize(
        ("translate_x", "translate_y", "t_imprints", "a_faces", "bc_faces"),
        [(0.6, 0.6, 2, 8, 7), (0.3, 0.5, 3, 9, 8), (2.5, 2.5, 0, 6, 6)],
    )
    def test_imprint_solids(
        self,
        require_cadquery_backend,  # noqa: ARG002
        bm_solid,
        use_cgal: bool,  # noqa: FBT001
        translate_x: float,
        translate_y: float,
        t_imprints: int,
        a_faces: int,
        bc_faces: int,
    ):
        box_a = bm_solid
        box_b = deepcopy(box_a)
        box_b.translate([translate_x, translate_y, 1])
        box_c = deepcopy(box_a)
        box_c.translate([-translate_x, -translate_y, 1])

        pre_imps = [box_a, box_b, box_c]
        pre_imp_labels = ["box_a", "box_b", "box_c"]

        imp_result = _imprinting.imprint_solids(
            pre_imps, labels=pre_imp_labels, use_cgal=use_cgal
        )
        imps = imp_result.imprintables
        imp_solids = imp_result.solids

        assert imp_result.total_imprints == t_imprints
        assert len(imp_solids) == 3
        assert len(imp_solids[0].faces) == a_faces
        assert imps[0]._has_imprinted is (t_imprints > 0)
        assert len(imp_solids[1].faces) == bc_faces
        assert imps[1]._has_imprinted is (t_imprints > 0)
        assert len(imp_solids[2].faces) == bc_faces
        assert imps[2]._has_imprinted is (t_imprints > 0)

        # Rerun the tests mocking cgal as not available
        if use_cgal:
            with mock.patch(
                "bluemira.geometry.overlap_checking.cgal.cgal_available",
                return_value=False,
            ):
                imp_result = _imprinting.imprint_solids(
                    pre_imps, labels=pre_imp_labels, use_cgal=use_cgal
                )

            imps = imp_result.imprintables
            imp_solids = imp_result.solids

            assert imp_result.total_imprints == t_imprints
            assert len(imp_solids) == 3
            assert len(imp_solids[0].faces) == a_faces
            assert imps[0]._has_imprinted is (t_imprints > 0)
            assert len(imp_solids[1].faces) == bc_faces
            assert imps[1]._has_imprinted is (t_imprints > 0)
            assert len(imp_solids[2].faces) == bc_faces
            assert imps[2]._has_imprinted is (t_imprints > 0)


class TestImprintableSolid:
    """Unit tests for ``ImprintableSolid`` and its module-level helpers."""

    def test_from_bluemira_solid_rejects_wrong_type(self, require_cadquery_backend):  # noqa: ARG002
        with pytest.raises(TypeError, match="bm_solid must be a BluemiraSolid"):
            _imprinting.ImprintableSolid.from_bluemira_solid("bad", "not a solid")

    def test_bm_shape_to_ocp_solid_rejects_unknown_type(self, require_cadquery_backend):  # noqa: ARG002
        with pytest.raises(TypeError, match="requires a CadQuery shape"):
            _imprinting._bm_shape_to_ocp_solid(object())

    def test_to_bluemira_solid_returns_original_when_not_imprinted(
        self,
        require_cadquery_backend,  # noqa: ARG002
        bm_solid,
    ):
        imp = _imprinting.ImprintableSolid.from_bluemira_solid("box", bm_solid)
        # ``set_imprinted_solid`` was never called, so the cached BluemiraSolid
        # is returned verbatim.
        assert imp.to_bluemira_solid() is bm_solid

    def test_set_imprinted_then_to_bluemira_solid_roundtrips(
        self,
        require_cadquery_backend,  # noqa: ARG002
        bm_solid,
    ):
        imp = _imprinting.ImprintableSolid.from_bluemira_solid("box", bm_solid)
        # Re-feed the OCP solid we already produced — flips ``_has_imprinted``
        # and forces the round-trip back to a BluemiraSolid.
        imp.set_imprinted_solid(imp.ocp_solid)
        out = imp.to_bluemira_solid()
        assert out is not bm_solid
        assert out.label == "box"
        assert out.volume == pytest.approx(bm_solid.volume, rel=1e-6)

    def test_bind_and_finalise_replaces_imprinted_faces(
        self,
        require_cadquery_backend,  # noqa: ARG002
        bm_solid,
    ):
        imp = _imprinting.ImprintableSolid.from_bluemira_solid("box", bm_solid)
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

    def test_label_property(self, require_cadquery_backend, bm_solid):  # noqa: ARG002
        imp = _imprinting.ImprintableSolid.from_bluemira_solid("my_label", bm_solid)
        assert imp.label == "my_label"


def test_imprinting_is_not_supported_by_freecad(monkeypatch):
    """The public entry point gives a clear error for the FreeCAD backend."""
    pytest.importorskip("cadquery")
    monkeypatch.setattr(cadapi, "apiSolid", object())
    with pytest.raises(
        NotImplementedError, match="only supported with the CadQuery backend"
    ):
        _imprinting.imprint_solids([], [])

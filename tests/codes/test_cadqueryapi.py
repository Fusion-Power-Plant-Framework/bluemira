# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import json
import os
import struct
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BLUEMIRA_GEOMETRY_BACKEND") != "cadquery",
    reason="CadQuery-API tests; active backend is not cadquery",
)

cadapi = pytest.importorskip("bluemira.codes.cadapi._cadquery")

from tests.codes._shared.backend_api_tests import BackendApiTestsBase  # noqa: E402


class TestCadqueryapi(BackendApiTestsBase):
    cadapi = cadapi


class TestSaveCadMeshFormats:
    """Tests for the mesh-export formats added to the CadQuery backend.

    Covers ``CADFileType`` resolution, file-extension handling, and the
    binary-output guarantees we want downstream consumers to be able to rely
    on (binary STL, GLB magic header, glTF + sibling .bin).
    """

    @pytest.fixture
    def box(self):
        # Closed polygon → extrude to a unit-cube shell. extrude_shape returns
        # a Shell; that's enough for STL/glTF (both work off triangulated faces).
        outline = cadapi.make_polygon([
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 0),
        ])
        return cadapi.extrude_shape(outline, (0, 0, 1))

    # ---- CADFileType enum --------------------------------------------------

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("stl", "STL"),
            ("gltf", "GLTRANSMISSION"),
            ("glb", "GLTRANSMISSION"),
        ],
    )
    def test_cadfiletype_resolves_aliases(self, alias, expected):
        assert cadapi.CADFileType(alias).name == expected

    def test_cadfiletype_classifies_new_formats(self):
        ft = cadapi.CADFileType
        assert ft.STL in ft.unitless_formats()
        assert ft.GLTRANSMISSION in ft.unitless_formats()
        assert ft.STL in ft.not_importable_formats()
        assert ft.GLTRANSMISSION in ft.not_importable_formats()

    # ---- STL ---------------------------------------------------------------

    def test_save_stl_is_binary_by_default(self, box, tmp_path):
        path = tmp_path / "box.stl"
        cadapi.save_cad([box], str(path), cad_format="stl")

        data = path.read_bytes()
        # Binary STL has an 80-byte header followed by a uint32 triangle count.
        # ASCII STL starts with literal "solid". The OCCT binary writer happens
        # to put "STL " in its header, but the contract we care about is
        # "doesn't start with 'solid '" + "header parses as binary STL".
        assert not data.startswith(b"solid "), "should be binary STL, not ASCII"
        assert len(data) >= 84, "binary STL needs at least 80-byte header + uint32"
        ntri = struct.unpack_from("<I", data, 80)[0]
        assert ntri > 0, "no triangles written"
        # 80-byte header + 4-byte count + 50 bytes per facet
        assert len(data) == 84 + 50 * ntri

    def test_save_stl_appends_extension_when_missing(self, box, tmp_path):
        target = tmp_path / "noext"
        cadapi.save_cad([box], str(target), cad_format="stl")
        assert (tmp_path / "noext.stl").exists()

    def test_save_stl_accepts_existing_extension(self, box, tmp_path):
        path = tmp_path / "already.stl"
        cadapi.save_cad([box], str(path), cad_format="stl")
        assert path.exists()
        # No double-extension file ".stl.stl"
        assert not (tmp_path / "already.stl.stl").exists()

    def test_save_stl_combines_multiple_shapes(self, box, tmp_path):
        # Second shape: a smaller box translated, so the compound has 2 distinct
        # solids → expect roughly double the triangle count of a single box.
        small = cadapi.extrude_shape(
            cadapi.make_polygon([(2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0), (2, 0, 0)]),
            (0, 0, 0.5),
        )
        single = tmp_path / "single.stl"
        combined = tmp_path / "combined.stl"
        cadapi.save_cad([box], str(single), cad_format="stl")
        cadapi.save_cad([box, small], str(combined), cad_format="stl")
        n_single = struct.unpack_from("<I", single.read_bytes(), 80)[0]
        n_combined = struct.unpack_from("<I", combined.read_bytes(), 80)[0]
        assert n_combined > n_single

    # ---- glTF / GLB --------------------------------------------------------

    def test_save_gltf_is_valid_json_with_bin_sibling(self, box, tmp_path):
        path = tmp_path / "box.gltf"
        cadapi.save_cad([box], str(path), cad_format="gltf", labels=["box"])

        assert path.exists()
        sibling = tmp_path / "box.bin"
        assert sibling.exists(), "glTF (text) writer should emit a sibling .bin"

        doc = json.loads(path.read_text())
        # Spec-required top-level keys in a non-trivial glTF.
        for key in ("asset", "buffers", "bufferViews", "accessors", "meshes"):
            assert key in doc, f"glTF missing top-level '{key}'"
        # The buffer should reference our sibling .bin file.
        buffer_uris = [b.get("uri", "") for b in doc["buffers"]]
        assert any(uri.endswith(".bin") for uri in buffer_uris)

    def test_save_glb_has_correct_magic_header(self, box, tmp_path):
        path = tmp_path / "box.glb"
        cadapi.save_cad([box], str(path), cad_format="glb", labels=["box"])

        data = path.read_bytes()
        # GLB header per spec: magic 'glTF' (0x46546C67), version u32, length u32
        magic, version, length = struct.unpack_from("<4sII", data, 0)
        assert magic == b"glTF"
        assert version == 2
        assert length == len(data), "GLB length field must equal file size"

    def test_save_glb_appends_glb_extension(self, box, tmp_path):
        target = tmp_path / "noext"
        cadapi.save_cad([box], str(target), cad_format="glb")
        # Format string was "glb" → caller wanted .glb, not the enum-default .gltf
        assert (tmp_path / "noext.glb").exists()
        assert not (tmp_path / "noext.gltf").exists()

    def test_save_gltf_appends_gltf_extension(self, box, tmp_path):
        target = tmp_path / "noext"
        cadapi.save_cad([box], str(target), cad_format="gltf")
        assert (tmp_path / "noext.gltf").exists()
        assert not (tmp_path / "noext.glb").exists()

    def test_save_gltf_preserves_labels_as_node_names(self, box, tmp_path):
        path = tmp_path / "named.gltf"
        cadapi.save_cad([box], str(path), cad_format="gltf", labels=["my_box"])
        doc = json.loads(path.read_text())
        node_names = [n.get("name", "") for n in doc.get("nodes", [])]
        assert any("my_box" in name for name in node_names), (
            f"label not found in glTF node names: {node_names}"
        )

    # ---- Dispatcher edge cases --------------------------------------------

    def test_save_cad_rejects_unsupported_format(self, box, tmp_path):
        with pytest.raises(cadapi.FreeCADError, match="not supported"):
            cadapi.save_cad(
                [box], str(tmp_path / "x.iges"), cad_format=cadapi.CADFileType.IGES
            )

    def test_save_cad_accepts_pathlib_path(self, box, tmp_path):
        path = tmp_path / "from_path.stl"
        cadapi.save_cad([box], path, cad_format="stl")
        assert Path(path).exists()


class TestSerialiseTrimmedEdges:
    """Trimmed BSpline / Bezier round-trip via ``serialise_shape``.

    FreeCAD's ``make_bezier`` / ``make_bspline`` don't accept a parameter
    range — only the CadQuery backend can both build a trimmed edge and
    persist its FirstParameter/LastParameter. This guards that round-trip.
    """

    def test_trimmed_bezier_roundtrips(self):
        poles = [(0, 0, 0), (1, 2, 0), (2, -1, 0), (3, 0, 0)]
        original = cadapi.make_bezier(poles, first_parameter=0.2, last_parameter=0.7)
        serialised = cadapi.serialise_shape(original)
        assert "Wire" in serialised
        assert "BezierCurve" in serialised["Wire"][0]
        bez = serialised["Wire"][0]["BezierCurve"]
        assert bez["FirstParameter"] == pytest.approx(0.2)
        assert bez["LastParameter"] == pytest.approx(0.7)

        restored = cadapi.deserialise_shape(serialised)
        # Endpoints survive the round-trip — that's what trimmed-edge support
        # is for. (Length == length is a sanity check; trimmed bezier has
        # different length than the full curve.)
        assert restored.Length() == pytest.approx(original.Length(), rel=1e-9)


class TestWireFromWiresDisjoint:
    """``wire_from_wires`` warn-then-longest path on disjoint inputs.

    FreeCAD's ``wire_from_wires`` extends edges into a single ``Part.Wire``
    unconditionally; only the CadQuery backend has a connectivity check
    that warns and returns the longest disconnected piece.
    """

    def test_disjoint_inputs_warn_and_return_longest(self, caplog):
        short = cadapi.make_polygon([[0, 0, 0], [1, 0, 0]])
        long = cadapi.make_polygon([[10, 0, 0], [10, 0, 5]])
        result = cadapi.wire_from_wires([short, long])
        assert "did not all join" in caplog.text
        assert result.Length() == pytest.approx(long.Length())

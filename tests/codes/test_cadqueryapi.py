# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BLUEMIRA_GEOMETRY_BACKEND") != "cadquery",
    reason="CadQuery-API tests; active backend is not cadquery",
)

cadapi = pytest.importorskip("bluemira.codes.cadapi._cadquery")
cq = pytest.importorskip("cadquery")

from unittest.mock import MagicMock, patch  # noqa: E402

import cadquery as cq  # noqa: E402
import numpy as np  # noqa: E402

from bluemira.codes.error import FreeCADError  # noqa: E402
from bluemira.geometry.error import GeometryError  # noqa: E402
from bluemira.geometry.tools import convert  # noqa: E402
from tests.codes._shared.backend_api_tests import BackendApiTestsBase  # noqa: E402

E1, E2, E3, E4 = "e1", "e2", "e3", "e4"

E1, E2, E3, E4 = "e1", "e2", "e3", "e4"


def _closed_square(side: float = 1.0, origin=(0.0, 0.0, 0.0)):
    """A closed-polygon wire (last point == first). cadapi.make_polygon
    has no ``closed=`` kwarg — closure is encoded by repeating the
    first point at the end.
    """
    ox, oy, oz = origin
    return cadapi.make_polygon([
        [ox, oy, oz],
        [ox + side, oy, oz],
        [ox + side, oy + side, oz],
        [ox, oy + side, oz],
        [ox, oy, oz],
    ])


class TestCadqueryapi(BackendApiTestsBase):
    cadapi = cadapi

    def _make_box(self):
        return cadapi.extrude_shape(
            cadapi.make_face(
                cadapi.make_polygon([
                    (0, 0, 0),
                    (1, 0, 0),
                    (1, 1, 0),
                    (0, 1, 0),
                ]).close()
            ),
            (0, 0, 1),
        )

    def test_face_from_wires_tolerant_non_planar_path(self):
        """
        Test non-planar path.
        Create a rectangle on the surface of a cylinder and attempt to
        create a face from the wire.
        """
        radius = 1.0
        height = 2.0

        vertical_line = cadapi.make_polygon([(radius, 0, 0), (radius, 0, height)])
        cylinder_surface = (
            cadapi.revolve_shape(  # revolve vertical line to create cylinder
                vertical_line,
                base=(0, 0, 0),
                direction=(0, 0, 1),
                degree=360,
            )
        )

        rect_points = [
            (0, -2, 0),
            (radius, -2, 0),
            (radius, -2, height),
            (0, -2, height),
            (0, -2, 0),
        ]
        rect_wire = cadapi.make_polygon(rect_points)
        rect_face = cadapi.make_face(rect_wire)  # create a 2D rectangle in the XZ plane

        cutting_block = cadapi.extrude_shape(rect_face, (0, 4, 0))  # extrude in Y

        intersection_result = cylinder_surface.intersect(cutting_block)
        curved_faces = cadapi.faces(
            intersection_result
        )  # intersect to yield curved cylinder face

        if curved_faces:
            target_face = curved_faces[0]
            wrapped_wire = target_face.outerWire()

        face = cadapi._face_from_wires_tolerant(
            wrapped_wire, []
        )  # should succeed via non-planar path

        assert isinstance(face, cq.Face)  # verify face
        assert cadapi.area(face) > 0

        with pytest.raises(ValueError, match="not planar"):
            cq.Face.makeFromWires(wrapped_wire)  # expected error

    def test_face_from_wires_tolerant_strict_planar_path(self):
        """
        Test the strict planar path.
        Create a square in the XY plane and attempt to create a face from the wire.
        """
        points = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 0),
        ]

        wire = cadapi.make_polygon(points)  # planar square wire

        face = cadapi._face_from_wires_tolerant(
            wire, []
        )  # this should fail the first path but succeed with makeFromWires

        assert isinstance(face, cq.Face)  # verify face and area
        assert cadapi.area(face) == pytest.approx(1.0, abs=1e-6)

        direct_face = cq.Face.makeFromWires(
            wire
        )  # should work directly with makeFromWires
        assert cadapi.area(direct_face) == pytest.approx(1.0, abs=1e-6)

    def test_face_from_wires_tolerant_tolerant_planar_path(self):
        """
        Test the tolerant planar path.
        Create a planar square but add small noise to the points that exceeds
        the default confusion tolerance.
        Attempt to create a face from the wire, should fallback to SVD.
        """
        # corners of a square with out-of-plane noise in z
        p0 = (0.0, 0.0, 1.5e-5)
        p1 = (1.0, 0.0, -1.2e-5)
        p2 = (1.0, 1.0, 2.1e-5)
        p3 = (0.0, 1.0, -1.8e-5)

        # guarantee closed wire
        noisy_points = [p0, p1, p2, p3, p0]

        wire = cadapi.make_polygon(noisy_points)

        assert wire.IsClosed()

        with pytest.raises(ValueError, match="not planar"):
            cq.Face.makeFromWires(wire)

        face = cadapi._face_from_wires_tolerant(
            wire, []
        )  # should succeed via SVD fallback

        assert isinstance(face, cq.Face)
        assert face.isValid()
        assert cadapi.area(face) == pytest.approx(1.0, abs=1e-6)

    def test_face_from_wires_tolerant_with_holes(self):
        """Test planar path with inner hole wires."""

        outer_points = [  # 2x2 flat square
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-1.0, -1.0, 0.0),
        ]
        outer_wire = cadapi.make_polygon(outer_points)

        inner_points = [  # inner 1x1 square hole
            (-0.5, -0.5, 0.0),
            (0.5, -0.5, 0.0),
            (0.5, 0.5, 0.0),
            (-0.5, 0.5, 0.0),
            (-0.5, -0.5, 0.0),
        ]
        inner_wire = cadapi.make_polygon(inner_points)

        face = cadapi._face_from_wires_tolerant(outer_wire, [inner_wire])

        assert isinstance(face, cq.Face)
        assert face.isValid()

        assert len(face.innerWires()) == 1

        # area = 4.0 (outer) - 1.0 (inner)
        assert cadapi.area(face) == pytest.approx(3.0, abs=1e-6)

    def test_face_from_wires_tolerant_planar_with_noisy_hole(self):
        """Test planar path with a noisy inner hole that exceeds tolerance."""
        outer_points = [  # noisy 2x2 square
            (0.0, 0.0, 1.5e-5),
            (2.0, 0.0, -1.2e-5),
            (2.0, 2.0, 2.1e-5),
            (0.0, 2.0, -1.8e-5),
            (0.0, 0.0, 1.5e-5),
        ]
        outer_wire = cadapi.make_polygon(outer_points)

        inner_points = [  # noisy 1x1 square hole
            (0.5, 0.5, 1.1e-5),
            (1.5, 0.5, -1.3e-5),
            (1.5, 1.5, 2.0e-5),
            (0.5, 1.5, -1.6e-5),
            (0.5, 0.5, 1.1e-5),
        ]
        inner_wire = cadapi.make_polygon(inner_points)

        with pytest.raises(ValueError, match="not planar"):
            cq.Face.makeFromWires(outer_wire, [inner_wire])

        face = cadapi._face_from_wires_tolerant(outer_wire, [inner_wire])

        assert isinstance(face, cq.Face)
        assert face.isValid()
        assert len(face.innerWires()) == 1
        assert cadapi.area(face) == pytest.approx(3.0, abs=1e-6)

    def test_face_from_wires_tolerant_inner_same_winding(self):
        """Test that inner wires with the same winding as the outer wire
        are handled correctly.
        """
        outer = cadapi.make_polygon(  # out square
            [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 0)]
        )

        inner = cadapi.make_polygon(  # inner square hole with same anticlockwise winding
            [(0.5, 0.5, 0), (1.5, 0.5, 0), (1.5, 1.5, 0), (0.5, 1.5, 0), (0.5, 0.5, 0)]
        )

        face = cadapi._face_from_wires_tolerant(outer, [inner])

        assert face.isValid()
        assert len(face.innerWires()) == 1
        assert cadapi.area(face) == pytest.approx(3.0, abs=1e-6)

    def test_face_from_wires_tolerant_hole_outside_outer(self):
        """Test that an inner hole completely outside the outer wire is rejected."""
        outer = cadapi.make_polygon([
            (0, 0, 0),
            (2, 0, 0),
            (2, 2, 0),
            (0, 2, 0),
            (0, 0, 0),
        ])

        inner_outside = (
            cadapi.make_polygon(  # hole is located completely outside of the outer wire
                [(5, 5, 0), (6, 5, 0), (6, 6, 0), (5, 6, 0), (5, 5, 0)]
            )
        )

        with pytest.raises(GeometryError):
            _ = cadapi._face_from_wires_tolerant(outer, [inner_outside])

    def test_edge_pair_cos_angle_straight(self):
        """Perfectly collinear."""
        e1 = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)]).edges()
        e2 = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)]).edges()

        result = cadapi._edge_pair_cos_angle(e1, e2)
        assert result == pytest.approx(1.0)

    def test_edge_pair_cos_angle_orthogonal(self):
        """Orthogonal edges."""
        e1 = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)]).edges()
        e2 = cadapi.make_polygon([(0, 0, 0), (0, 1, 0)]).edges()

        result = cadapi._edge_pair_cos_angle(e1, e2)
        assert result == pytest.approx(0.0)

    def test_edge_pair_cos_angle_u_turn(self):
        """Opposite directions."""
        e1 = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)]).edges()
        e2 = cadapi.make_polygon([(0, 0, 0), (-1, 0, 0)]).edges()

        result = cadapi._edge_pair_cos_angle(e1, e2)
        assert result == pytest.approx(-1.0)

    def test_edge_junction_pairs(self):
        """Test pairwise list of edges"""
        wire = cadapi.make_polygon([
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
        ]).close()

        pairs = cadapi._edge_junction_pairs(wire)

        assert len(pairs) == 4

    @pytest.mark.parametrize(
        ("edges", "is_closed", "expected_pairs"),
        [
            ([E1, E2, E3, E4], False, [(E1, E2), (E2, E3), (E3, E4)]),
            ([E1, E2], False, [(E1, E2)]),
            ([E1], False, []),
            ([], False, []),
            ([E1, E2, E3], True, [(E1, E2), (E2, E3), (E3, E1)]),
            ([E1, E2], True, [(E1, E2), (E2, E1)]),
            ([E1], True, []),
            ([], True, []),
        ],
    )
    @patch("bluemira.codes._cadqueryapi._core.is_closed")
    @patch("bluemira.codes._cadqueryapi._core.ordered_edges")
    def test_edge_junction_pairs_mocked(
        self,
        mock_ordered_edges,
        mock_is_closed,
        edges,
        is_closed,
        expected_pairs,
    ):
        mock_ordered_edges.return_value = edges
        mock_is_closed.return_value = is_closed

        result = cadapi._edge_junction_pairs("dummy_wire")

        assert result == expected_pairs

    def test_planar_wire_returns_normal(self):
        """_get_planar_normal returns correct normal for valid planar wire."""
        result = cadapi._get_planar_normal(
            cadapi.make_polygon([
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
            ])
        )
        assert result == (0, 0, 1)

    def test_non_planar_wire_returns_none(self):
        """_get_planar_normal returns none for a non-planar wire."""
        result = cadapi._get_planar_normal(
            cadapi.make_polygon([
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 1),
                (0, 1, 0),
            ])
        )
        assert result is None

    def test_sewn_solid_successful_rebuild(self):
        """Sew disconnected faces into a valid solid."""
        box = self._make_box()
        faces = box.faces()
        assert not isinstance(faces, cq.Solid)
        solid = cadapi._sewn_solid(faces)
        assert isinstance(solid, cq.Solid)
        assert solid.isValid()
        assert len(solid.Faces()) == 6
        assert solid.Volume() == pytest.approx(1.0)

    def test_sewn_solid_missing_face(self):
        """Sew open shell fails to make closed solid."""
        box = self._make_box()
        faces = box.Faces()[:-1]
        open_shell = cadapi.make_compound(faces)
        open_solid = cadapi._sewn_solid(open_shell)
        assert open_solid is open_shell

    def test_sewn_solid_multiple_shells(self):
        """Sew disconnected regions gives >1 shell."""
        box1 = self._make_box()
        box2 = box = cadapi.translate_shape(self._make_box(), (5, 0, 0))

        all_faces = box1.Faces() + box2.Faces()
        split_compound = cadapi.make_compound(all_faces)

        multi_shell = cadapi._sewn_solid(split_compound)
        assert multi_shell is split_compound

    def test_sewn_solid_already_valid(self):
        """Valid solid should rebuild safely."""
        box = self._make_box()
        solid = cadapi._sewn_solid(box)
        assert isinstance(solid, cq.Solid)
        assert solid.isValid()
        assert solid.Volume() == pytest.approx(1.0)

    def test_wire_is_straight_single_straight(self):
        """True for single straight line."""
        assert cadapi._wire_is_straight(cadapi.make_polygon([(0, 0, 0), (1, 0, 0)]))

    def test_wire_is_straight_multi_straight(self):
        """False for multiple straight lines."""
        assert not cadapi._wire_is_straight(
            cadapi.make_polygon([
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
            ])
        )

    def test_wire_is_straight_curved(self):
        """False for curved line."""
        assert not cadapi._wire_is_straight(
            cadapi.make_circle_arc_3P(
                (0, 0, 0),
                (5, 5, 0),
                (10, 0, 0),
            )
        )

    def test_wire_is_planar_closed(self):
        """True for 2D flat shape."""
        assert cadapi._wire_is_planar(
            cadapi.make_polygon([
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
            ]).close()
        )

    def test_wire_is_planar_non_planar(self):
        """False for non-planar shape."""
        assert not cadapi._wire_is_planar(
            cadapi.make_polygon([
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 1),
                (0, 1, 0),
            ]).close()
        )

    def test_occ_face_area_square(self):
        """Simple area calculation on square."""
        square = cadapi.make_face(
            cadapi.make_polygon([
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
            ]).close()
        )
        assert cadapi._occ_face_area(square.wrapped) == pytest.approx(1.0)

    def test_occ_face_area_circle(self):
        """Simple area calculation on circle."""
        circle = cadapi.make_face(cadapi.make_circle())
        assert cadapi._occ_face_area(circle.wrapped) == pytest.approx(np.pi)

    def test_cq_area_prop_face(self):
        """Area property on standard face."""
        square = cadapi.make_face(
            cadapi.make_polygon([
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
            ]).close()
        )
        assert convert(square).area == pytest.approx(1.0)

    def test_cq_area_prop_face_with_holes(self):
        """Area property on face with holes."""
        square = cadapi.make_face(
            cadapi.make_polygon([
                (0, 0, 0),
                (2, 0, 0),
                (2, 2, 0),
                (0, 2, 0),
            ]).close()
        )
        circle = cadapi.make_face(cadapi.make_circle(radius=0.5, center=(1, 1, 0)))
        assert convert(cadapi.boolean_cut(square, circle)[0]).area == pytest.approx(
            4.0 - (np.pi * 0.25)
        )


@patch("bluemira.codes.error.FreeCADError", FreeCADError)
@patch("bluemira.codes._cadqueryapi._core._edge_pair_cos_angle")
@patch("bluemira.codes._cadqueryapi._core._edge_junction_pairs")
class TestCheckPathTangentContinuity:
    def test_smooth_path(self, mock_pairs, mock_cos_angle):
        """Smooth path should pass without error."""
        mock_pairs.return_value = [(E1, E2), (E2, E3)]
        mock_cos_angle.side_effect = [1.0, 1.0]  # covers all calls in for loop

        cadapi._check_path_tangent_continuity(MagicMock())

        assert mock_cos_angle.call_count == 2

    def test_kinked_path_raises_error(self, mock_pairs, mock_cos_angle):
        """Path with kink should raise FreeCADError."""
        mock_pairs.return_value = [(E1, E2)]
        mock_cos_angle.side_effect = [0.5]

        with pytest.raises(FreeCADError) as e:
            cadapi._check_path_tangent_continuity(MagicMock())

        assert "path is not tangent-continuous" in str(e.value)

    def test_degenerate_edge_skipped(self, mock_pairs, mock_cos_angle):
        """Continue without failing if _edge_pair_cos_angle returns None."""
        mock_pairs.return_value = [(E1, E2), (E2, E3)]
        mock_cos_angle.side_effect = [None, 1.0]

        cadapi._check_path_tangent_continuity(MagicMock())

        assert mock_cos_angle.call_count == 2

    def test_within_tolerance(self, mock_pairs, mock_cos_angle):
        """Values inside tolerance limit pass without error."""
        mock_pairs.return_value = [(E1, E2)]
        tol = 1e-6
        mock_cos_angle.side_effect = [1.0 - (1e-6 / 10)]

        cadapi._check_path_tangent_continuity(MagicMock(), tol=tol)

        assert mock_cos_angle.call_count == 1

    def test_outside_tolerance(self, mock_pairs, mock_cos_angle):
        """Values outside tolerance limit should raise FreeCADError."""
        mock_pairs.return_value = [(E1, E2)]
        tol = 1e-6
        mock_cos_angle.side_effect = [1.0 - (tol * 10)]

        with pytest.raises(FreeCADError):
            cadapi._check_path_tangent_continuity(MagicMock(), tol=tol)

    def test_raises_on_first_failure(self, mock_pairs, mock_cos_angle):
        """Raise FreeCADError on first kink found."""
        mock_pairs.return_value = [(E1, E2), (E2, E3), (E3, E4)]
        mock_cos_angle.side_effect = [1.0, 0.0, 1.0]

        with pytest.raises(FreeCADError):
            cadapi._check_path_tangent_continuity(MagicMock())

        assert mock_cos_angle.call_count == 2


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


class TestInternalHelpers:
    """Direct unit tests for ``_cadquery/core.py`` internal helpers.

    These functions don't have FreeCAD-side equivalents (FreeCAD's C++ does
    the same work behind ``Part.X`` constructors), so coverage from the
    backend-shared test suite plus the high-level ``geometry/tools.py`` API
    misses their edge/error branches.
    """

    # ---- _face_from_wires_tolerant ----------------------------------------

    def test_face_from_wires_inner_outside_bounds_raises(self):
        outer = _closed_square(1.0)
        # Inner extends well beyond outer's bounding box → topological error.
        bad_inner = _closed_square(10.0, origin=(-5, -5, 0))
        with pytest.raises(GeometryError, match="outside the bounds"):
            cadapi.apiFace([outer, bad_inner])

    def test_apiface_with_none_returns_empty_face(self):
        face = cadapi.apiFace(None)
        # The ``None`` branch returns ``cq.Face.__new__(cq.Face)`` — uninitialised
        # but still a cq.Face by metaclass-checked isinstance.
        assert isinstance(face, cadapi.apiFace)

    # ---- _pick_dominant_dangler -------------------------------------------

    def test_pick_dominant_dangler_picks_larger_no_tie(self):
        big = cadapi.extrude_shape(cadapi.apiFace(_closed_square(10.0)), (0, 0, 1))
        small = cadapi.extrude_shape(cadapi.apiFace(_closed_square(1.0)), (0, 0, 1))
        winner = cadapi._pick_dominant_dangler([small, big], source_idx=0)
        assert cadapi.volume(winner) == pytest.approx(cadapi.volume(big))

    def test_pick_dominant_dangler_ties_raise(self):
        # Build two distinct solids with identical volume — same-size unit
        # cubes at different origins. Can't reuse one solid here:
        # ``translate_shape`` mutates in place, so passing it twice would feed
        # the same Python object to the helper.
        a = cadapi.extrude_shape(
            cadapi.apiFace(_closed_square(1.0, origin=(0, 0, 0))), (0, 0, 1)
        )
        b = cadapi.extrude_shape(
            cadapi.apiFace(_closed_square(1.0, origin=(5, 0, 0))), (0, 0, 1)
        )
        with pytest.raises(GeometryError, match="equally-sized"):
            cadapi._pick_dominant_dangler([a, b], source_idx=0)

    # ---- _check_path_tangent_continuity -----------------------------------

    def test_check_path_tangent_continuity_kink_raises(self):
        # 90° kink at (1, 0, 0): incoming edge points +x, outgoing +y →
        # cos(angle) = 0, far below the 1 - 1e-6 tangent-continuity threshold.
        kinked = cadapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        with pytest.raises(FreeCADError, match="tangent-continuous"):
            cadapi._check_path_tangent_continuity(kinked)

    def test_check_path_tangent_continuity_straight_passes(self):
        # Single-segment polygon has no interior junctions → no checks → OK.
        straight = cadapi.make_polygon([[0, 0, 0], [1, 0, 0]])
        cadapi._check_path_tangent_continuity(straight)  # must not raise

    # ---- _force_close_wire ------------------------------------------------

    def test_force_close_wire_already_closed_is_noop(self):
        closed = _closed_square(1.0)
        assert closed.IsClosed()
        # Short-circuit branch: returns the same instance unchanged.
        assert cadapi._force_close_wire(closed) is closed

    def test_force_close_wire_bridges_open_wire(self):
        # Open polygon (last point != first) — the bridge edge gets appended.
        open_wire = cadapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        assert not open_wire.IsClosed()
        bridged = cadapi._force_close_wire(open_wire)
        assert bridged.IsClosed()

    # ---- _collect_subshapes -----------------------------------------------

    def test_collect_subshapes_recurses_into_compound(self):
        # Build two distinct unit cubes — ``translate_shape`` is in-place
        # in the cadquery backend, so building from separate faces is the
        # only way to get two genuinely different Python objects.
        s1 = cadapi.extrude_shape(
            cadapi.apiFace(_closed_square(1.0, origin=(0, 0, 0))), (0, 0, 1)
        )
        s2 = cadapi.extrude_shape(
            cadapi.apiFace(_closed_square(1.0, origin=(5, 0, 0))), (0, 0, 1)
        )
        compound = cadapi.make_compound([s1, s2])
        # cq.Solid (not cadapi.apiSolid as that IS cq.Solid via alias) —
        # _collect_subshapes uses the cq classes directly as dict keys.
        solids = cadapi._collect_subshapes(compound, cq.Solid)
        assert len(solids) == 2
        # Each unit cube has 6 faces, so two cubes → 12 faces via the
        # recursive explorer.
        faces = cadapi._collect_subshapes(compound, cq.Face)
        assert len(faces) == 12


class TestCurves:
    """Direct unit tests for ``_cadquery/curves.py`` constructors.

    Targets the largest coverage gap on the cadquery backend (curves.py
    14 % per Codecov): branches that the high-level API doesn't reach
    on its happy paths — full circle / full ellipse, ``make_circle_arc_3P``
    with ``axis``, the ``weights=None`` arms of ``make_bspline`` /
    ``make_bsplinesurface``, and the coincident-endpoint guard in
    ``make_bspline_g1_blend``.
    """

    # ---- make_circle / _freecad_ax2 ---------------------------------------

    def test_make_circle_full_circle_branch(self):
        # start_angle == end_angle short-circuits to a closed-circle edge
        # via BRepBuilderAPI_MakeEdge(circ), bypassing GC_MakeArcOfCircle.
        wire = cadapi.make_circle(radius=1.0, start_angle=0.0, end_angle=0.0)
        assert wire.IsClosed()
        assert wire.Length() == pytest.approx(2 * math.pi, rel=1e-6)

    def test_make_circle_with_x_direction(self):
        # x_direction kwarg pins the local X-axis (used by serialisation
        # round-trip); exercises the non-default branch in _freecad_ax2.
        wire = cadapi.make_circle(
            radius=1.0,
            start_angle=0.0,
            end_angle=180.0,
            axis=(0, 0, 1),
            x_direction=(0, 1, 0),
        )
        # Half-circumference still equals π regardless of the X-axis pick.
        assert wire.Length() == pytest.approx(math.pi, rel=1e-6)

    def test_make_circle_axis_parallel_to_x_falls_back(self):
        # _freecad_ax2 picks (0,1,0) as the reference when axis ∥ (1,0,0)
        # to avoid a degenerate cross-product.
        wire = cadapi.make_circle(
            radius=1.0, start_angle=0.0, end_angle=180.0, axis=(1, 0, 0)
        )
        assert wire.Length() == pytest.approx(math.pi, rel=1e-6)

    # ---- make_circle_arc_3P -----------------------------------------------

    def test_make_circle_arc_3P_with_axis(self):
        # Three points on the unit circle in z=0; axis=(0,0,1) takes the
        # axis-override path (lines 334-343) instead of the natural path.
        wire = cadapi.make_circle_arc_3P(
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], axis=(0, 0, 1)
        )
        # Half-circle on the unit circle — length should be π.
        assert wire.Length() == pytest.approx(math.pi, rel=1e-4)

    # ---- make_ellipse -----------------------------------------------------

    def test_make_ellipse_full_branch(self):
        # start_angle == end_angle (after %= 360) → full closed ellipse,
        # exercises the ``cq.Edge.makeEllipse(..., 5 args)`` short form.
        wire = cadapi.make_ellipse(
            major_radius=2.0, minor_radius=1.0, start_angle=0.0, end_angle=360.0
        )
        assert wire.IsClosed()

    # ---- make_bspline (direct, no/with weights) ---------------------------

    def test_make_bspline_no_weights(self):
        # Linear B-spline through 2 poles — degree 1, knots [0, 1] with
        # multiplicities [2, 2] (sum mults = npoles + degree + 1).
        poles = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        wire = cadapi.make_bspline(
            poles,
            mults=[2, 2],
            knots=[0.0, 1.0],
            periodic=False,
            degree=1,
            weights=None,
            check_rational=False,
        )
        assert wire.Length() == pytest.approx(1.0, rel=1e-6)

    def test_make_bspline_with_weights(self):
        poles = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        wire = cadapi.make_bspline(
            poles,
            mults=[2, 2],
            knots=[0.0, 1.0],
            periodic=False,
            degree=1,
            weights=[1.0, 1.0],
            check_rational=False,
        )
        assert wire.Length() == pytest.approx(1.0, rel=1e-6)

    # ---- make_bsplinesurface ----------------------------------------------

    def test_make_bsplinesurface_no_weights(self):
        # Bilinear patch over a 2x2 grid; degree 1 in both directions.
        poles = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        ])
        face = cadapi.make_bsplinesurface(
            poles,
            mults_u=[2, 2],
            mults_v=[2, 2],
            knot_vector_u=[0.0, 1.0],
            knot_vector_v=[0.0, 1.0],
            degree_u=1,
            degree_v=1,
            weights=None,
        )
        assert isinstance(face, cq.Face)
        # Surface area > 0 — a real face was built, not a null one.
        assert cadapi.area(face) > 0.0

    def test_make_bsplinesurface_with_weights(self):
        poles = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ])
        face = cadapi.make_bsplinesurface(
            poles,
            mults_u=[2, 2],
            mults_v=[2, 2],
            knot_vector_u=[0.0, 1.0],
            knot_vector_v=[0.0, 1.0],
            degree_u=1,
            degree_v=1,
            weights=np.ones((2, 2)),
        )
        assert isinstance(face, cq.Face)
        # Unit-square patch, all weights = 1 → area should be 1.
        assert cadapi.area(face) == pytest.approx(1.0, rel=1e-6)

    # ---- make_bspline_g1_blend --------------------------------------------

    def test_make_bspline_g1_blend_coincident_endpoints_raise(self):
        # Both edges meet at (1, 0, 0) — chord length is zero, the blend
        # has no direction, so the helper raises FreeCADError.
        edge1 = cq.Edge.makeLine(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0))
        edge2 = cq.Edge.makeLine(cq.Vector(1, 0, 0), cq.Vector(0, 1, 0))
        # edge1's last point and edge2's first point are both (1, 0, 0).
        with pytest.raises(FreeCADError, match="identical endpoints"):
            cadapi.make_bspline_g1_blend(edge1, edge2)

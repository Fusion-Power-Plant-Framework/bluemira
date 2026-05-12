# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BLUEMIRA_GEOMETRY_BACKEND") != "cadquery",
    reason="CadQuery-API tests; active backend is not cadquery",
)

cadapi = pytest.importorskip("bluemira.codes._cadqueryapi")

from unittest.mock import MagicMock, patch  # noqa: E402

import cadquery as cq  # noqa: E402
import numpy as np  # noqa: E402
from OCP.BRep import BRep_Builder  # noqa: E402
from OCP.TopoDS import TopoDS_Wire  # noqa: E402

from bluemira.codes.error import FreeCADError  # noqa: E402
from bluemira.geometry.error import GeometryError  # noqa: E402
from bluemira.geometry.tools import convert  # noqa: E402
from tests.codes._shared.backend_api_tests import BackendApiTestsBase  # noqa: E402

E1, E2, E3, E4 = "e1", "e2", "e3", "e4"
CORE = "bluemira.codes._cadqueryapi._core"


class TestCadqueryapi(BackendApiTestsBase):
    cadapi = cadapi

    def _make_box(self, s=1):
        return cadapi.extrude_shape(
            cadapi.make_face(
                cadapi.make_polygon([
                    (0, 0, 0),
                    (s, 0, 0),
                    (s, s, 0),
                    (0, s, 0),
                ]).close()
            ),
            (0, 0, s),
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
    @patch(f"{CORE}.is_closed")
    @patch(f"{CORE}.ordered_edges")
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

    def test_unify_same_domain_coplanar_faces(self):
        """Coplanar faces and collinear edges are unified."""
        box1 = self._make_box()
        box2 = cadapi.translate_shape(self._make_box(), (1, 0, 0))

        bad_shape = box1.fuse(box2)
        assert len(bad_shape.Faces()) > 6

        unified = cadapi._unify_same_domain(bad_shape)
        assert len(unified.Faces()) == 6
        assert isinstance(unified, cq.Shape)

    def test_unify_same_domain_clean_shape(self):
        """Passing clean shape is idempotent."""
        clean = self._make_box()
        unified = cadapi._unify_same_domain(clean)
        assert len(unified.Faces()) == len(clean.Faces())
        assert len(unified.Edges()) == len(clean.Edges())

    @patch(f"{CORE}.bluemira_warn")
    @patch(f"{CORE}.ShapeUpgrade_UnifySameDomain")
    def test_unify_same_domain_exception(self, mock_unify, mock_warn):
        """Exceptions are caught."""
        unifier = MagicMock()
        unifier.Build.side_effect = RuntimeError("Fake Error")
        mock_unify.return_value = unifier

        box = self._make_box()
        result = cadapi._unify_same_domain(box)

        mock_warn.assert_called_once()
        assert "UnifySameDomain failed" in mock_warn.call_args[0][0]
        assert result is box

    def test_assemble_wires_from_edges_empty(self):
        """Empty list of edges returns empty list."""
        assert cadapi._assemble_wires_from_edges([]) == []

    def test_assemble_wires_from_edges_connected(self):
        """Sequentially connected wires are merged into a single wire."""
        e1 = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)]).Edges()
        e2 = cadapi.make_polygon([(1, 0, 0), (1, 1, 0)]).Edges()
        e3 = cadapi.make_polygon([(1, 1, 0), (0, 1, 0)]).Edges()

        wires = cadapi._assemble_wires_from_edges(e1 + e2 + e3)

        assert len(wires) == 1
        assert len(wires[0].Edges()) == 3
        assert isinstance(wires[0], cq.Wire)

    def test_assemble_wires_from_edges_disconnected(self):
        """Disconnected edges return separate wires."""
        e1 = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)]).Edges()
        e2 = cadapi.make_polygon([(1, 1, 0), (2, 1, 0)]).Edges()

        wires = cadapi._assemble_wires_from_edges(e1 + e2)

        assert len(wires) == 2
        assert len(wires[0].Edges()) == 1
        assert len(wires[1].Edges()) == 1

    def test_split_wire_by_closed_tools_intersection(self):
        """Split wire using closed tool that perfectly crosses it."""
        wire = cadapi.make_polygon([(0, 0, 0), (10, 0, 0)])
        tool = cadapi.make_polygon([
            (3, -2, 0),
            (7, -2, 0),
            (7, 2, 0),
            (3, 2, 0),
        ]).close()

        result = cadapi._split_wire_by_closed_tools(wire, [tool])

        assert len(result) == 3
        assert sorted([w.Length() for w in result]) == [3.0, 3.0, 4.0]

    def test_split_wire_by_closed_tools_no_intersection(self):
        """Unchanged wire if the tool doesn't overlap."""
        wire = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)])
        tool = cadapi.make_polygon([
            (2, 0, 0),
            (4, 0, 0),
            (4, 2, 0),
            (2, 2, 0),
        ]).close()

        result = cadapi._split_wire_by_closed_tools(wire, [tool])

        assert len(result) == 1
        assert result[0].Length() == 1

    def test_split_wire_at_tool_crossings_single_cut(self):
        """Split wire with open tool."""
        wire = cadapi.make_polygon([(0, 0, 0), (5, 0, 0)])
        tool = cadapi.make_polygon([(2, -2, 0), (2, 2, 0)])

        result = cadapi._split_wire_at_tool_crossings(wire, [tool])

        assert len(result) == 2
        assert [w.Length() for w in result] == [3, 2]  # must be sorted

    def test_split_wire_at_tool_crossings_no_cut(self):
        """Split wire with open tool that doesn't intersect."""
        wire = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)])
        tool = cadapi.make_polygon([(2, -2, 0), (2, 2, 0)])

        result = cadapi._split_wire_at_tool_crossings(wire, [tool])

        assert len(result) == 1
        assert result[0].Length() == 1

    def test_collect_subshapes_direct_children(self):
        """Extract immediate children."""
        box = self._make_box()
        faces = cadapi._collect_subshapes(box, cq.Face)
        assert len(faces) == 6
        assert all(isinstance(f, cq.Face) for f in faces)

    def test_collect_subshapes_nested_compounds(self):
        """Compound inside a compounds."""
        box = self._make_box()
        inner_compound = cadapi.make_compound([box])
        outer_compound = cadapi.make_compound([inner_compound])

        solids = cadapi._collect_subshapes(outer_compound, cq.Solid)
        assert len(solids) == 1
        assert isinstance(solids[0], cq.Solid)

    def test_collect_subshapes_not_found(self):
        """Return an empty list if kind doesn't exist."""
        face = cadapi.make_face(
            cadapi.make_polygon([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]).close()
        )

        solids = cadapi._collect_subshapes(face, cq.Solid)

        assert isinstance(solids, list)
        assert len(solids) == 0

    def test_collect_subshapes_shell_promotion_valid(self):
        """Request solid from closed shell promotes shell to solid."""
        box = self._make_box()
        shell = box.Shells()[0]
        compound = cadapi.make_compound([shell])

        solids = cadapi._collect_subshapes(compound, cq.Solid)

        assert len(solids) == 1
        assert isinstance(solids[0], cq.Solid)
        assert solids[0].isValid()

    @patch(f"{CORE}.fix_shape")
    @patch.object(cq.Solid, "isValid", return_value=False)
    def test_collect_subshapes_shell_promotion_invalid_repair(self, mock_fix_shape):
        """fix_shape called to repair invalid promoted solid."""
        box = self._make_box()
        shell = box.Shells()[0]
        compound = cadapi.make_compound([shell])

        solids = cadapi._collect_subshapes(compound, cq.Solid)

        assert len(solids) == 1
        assert isinstance(solids[0], cq.Solid)
        mock_fix_shape.assert_called_once_with(solids[0])

    def test_repair_closed_wire_closed(self):
        """Repair already closed wire is idempotent."""
        wire = cadapi.make_circle()
        assert wire.IsClosed()

        with patch(f"{CORE}.ShapeFix_Wire") as mock_fixer:
            result = cadapi._repair_closed_wire(wire)
            assert result is wire
            mock_fixer.assert_not_called()

    def test_repair_closed_wire_unflagged(self):
        """Wire is closed but topods close is false."""
        e1 = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)]).Edges()[0]
        e2 = cadapi.make_polygon([(1, 0, 0), (0, 0, 0)]).Edges()[0]

        raw_wire = TopoDS_Wire()
        builder = BRep_Builder()
        builder.MakeWire(raw_wire)
        builder.Add(raw_wire, e1.wrapped)
        builder.Add(raw_wire, e2.wrapped)

        wire = cq.Wire(raw_wire)
        assert not wire.IsClosed()

        repaired = cadapi._repair_closed_wire(wire)
        assert repaired.IsClosed()
        assert isinstance(repaired, cq.Wire)

    def test_repair_closed_wire_open_wire(self):
        """Open wire remains open."""
        wire = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)])
        assert not wire.IsClosed()

        repaired = cadapi._repair_closed_wire(wire)
        assert not repaired.IsClosed()

    @patch(f"{CORE}.ShapeFix_Wire")
    def test_repair_closed_wire_null_fallback(self, mock_shape_fix):
        """Return original wire if ShapeFix_Wire fails."""
        wire = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)])

        mock_fixer = MagicMock()
        mock_null = MagicMock()
        mock_null.IsNull.return_value = True
        mock_fixer.Wire.return_value = mock_null
        mock_shape_fix.return_value = mock_fixer

        repaired = cadapi._repair_closed_wire(wire)

        assert repaired is wire

    def test_force_close_wire_already_closed(self):
        """Force close already closed wire returns."""
        wire = cadapi.make_circle()
        assert wire.IsClosed()

        result = cadapi._force_close_wire(wire)
        assert result is wire

    @patch(f"{CORE}._repair_closed_wire")
    def test_force_close_coincident_endpoints(self, mock_repair):
        """Open wire with coincident endpoints calls _repair_closed_wire."""
        wire = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)])
        assert not wire.IsClosed()

        mock_repair.return_value = "fake_repaired_wire"

        with (
            patch.object(wire, "startPoint", return_value=cq.Vector(0, 0, 0)),
            patch.object(wire, "endPoint", return_value=cq.Vector(0, 0, 0)),
        ):
            result = cadapi._force_close_wire(wire)

        mock_repair.assert_called_once_with(wire)
        assert result == "fake_repaired_wire"

    def test_force_close_open_wire(self):
        """Bridge open wire to close it."""
        e1 = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)]).Edges()[0]
        e2 = cadapi.make_polygon([(1, 0, 0), (1, 1, 0)]).Edges()[0]
        wire = cadapi.wire_from_edges([e1, e2])

        assert not wire.IsClosed()
        assert len(wire.Edges()) == 2

        closed = cadapi._force_close_wire(wire)
        assert len(closed.Edges()) == 3
        assert closed.IsClosed()
        assert isinstance(closed, cq.Wire)

    def test_piece_mass_solid(self):
        """Solid's mass evaluates to its volume."""
        box = self._make_box()
        mass = cadapi._piece_mass(box)
        assert mass == pytest.approx(1.0)

    def test_piece_mass_face(self):
        """Face's mass evaluates to its area."""
        face = cadapi.make_face(cadapi.make_circle())
        mass = cadapi._piece_mass(face)
        assert mass == pytest.approx(np.pi)

    def test_piece_mass_edge(self):
        """Edge's mass evaluates to its area."""
        e1 = cadapi.make_polygon([(0, 0, 0), (1, 0, 0)]).Edges()[0]
        mass = cadapi._piece_mass(e1)
        assert mass == pytest.approx(1.0)

    def test_pick_dominant_dangler_success(self):
        """Largest mass piece is selected."""
        small = self._make_box()
        large = self._make_box(s=2)

        dominant = cadapi._pick_dominant_dangler([small, large], source_idx=0)
        assert dominant is large

    def test_pick_dominant_dangler_tie(self):
        """Raise when equal largest pieces."""
        box1 = self._make_box()
        box2 = self._make_box()

        with pytest.raises(GeometryError, match="equally-sized dangling pieces"):
            cadapi._pick_dominant_dangler([box1, box2], source_idx=1)

    def test_piece_source_map(self):
        """Fragment maps correctly group identical shapes."""
        s1 = self._make_box()
        s2 = self._make_box()
        s3 = self._make_box()

        fragment_map = [[s1, s2], [s1, s3]]
        unique_map = cadapi._piece_source_map(fragment_map)
        assert len(unique_map) == 3

        s1_entry = next((i for i in unique_map if i[0] is s1), None)
        s2_entry = next((i for i in unique_map if i[0] is s2), None)
        s3_entry = next((i for i in unique_map if i[0] is s3), None)

        assert s1_entry is not None
        assert s1_entry[1] == {0, 1}
        assert s2_entry is not None
        assert s2_entry[1] == {0}
        assert s3_entry is not None
        assert s3_entry[1] == {1}

    def test_grow_keepers_by_overlap(self):
        """Add pieces if they touch a kept piece, from 2 sources to max overlap."""
        k1 = self._make_box()
        p_layer_2 = self._make_box(s=2)
        p_layer_3 = self._make_box(s=3)
        p_isolated_2 = cadapi.translate_shape(self._make_box(s=4), (10, 10, 10))

        keepers = [k1]
        unique_pieces = [
            (k1, {0}),
            (p_layer_2, {0, 1}),  # 2 sources, touches k1
            (p_isolated_2, {1, 2}),  # 2 sources, touches nothing
            (p_layer_3, {0, 1, 2}),  # 2 sources, touches p_layer_2
        ]

        cadapi._grow_keepers_by_overlap(keepers, unique_pieces)

        assert len(keepers) == 3
        assert k1 in keepers
        assert p_layer_2 in keepers
        assert p_layer_3 in keepers
        assert p_isolated_2 not in keepers

    def test_shapes_touch_true(self):
        """Touching shapes return true."""
        box1 = self._make_box()
        box2 = cadapi.translate_shape(self._make_box(), (1, 0, 0))
        assert cadapi._shapes_touch(box1, box2)

    def test_shapes_touch_false(self):
        """Separated shapes return false."""
        box1 = self._make_box()
        box2 = cadapi.translate_shape(self._make_box(), (2, 0, 0))
        assert not cadapi._shapes_touch(box1, box2)


@patch("bluemira.codes.error.FreeCADError", FreeCADError)
@patch(f"{CORE}._edge_pair_cos_angle")
@patch(f"{CORE}._edge_junction_pairs")
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

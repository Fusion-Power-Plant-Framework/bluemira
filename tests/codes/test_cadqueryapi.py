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

from bluemira.codes.error import FreeCADError  # noqa: E402
from bluemira.geometry.error import GeometryError  # noqa: E402
from bluemira.geometry.tools import convert  # noqa: E402
from tests.codes._shared.backend_api_tests import BackendApiTestsBase  # noqa: E402

E1, E2, E3, E4 = "e1", "e2", "e3", "e4"


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

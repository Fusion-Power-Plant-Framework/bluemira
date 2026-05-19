# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Isolated tests for the CadQuery API backend (_cadquery).

These tests exercise the ~15 functions needed by the fusion_1 reactor model
without involving FreeCAD, the geometry wrapper classes, or any GUI.
"""

import numpy as np
import pytest

from bluemira.codes.error import InvalidCADInputsError


def _skip_cadquery():
    try:
        import cadquery  # noqa: F401, PLC0415
    except ImportError:
        return True
    return False


pytestmark = pytest.mark.skipif(_skip_cadquery(), reason="cadquery not installed")

import bluemira.codes.cadapi._cadquery as cqapi  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SQUARE = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],  # closed
])

N_CIRCLE = 40
_angles = np.linspace(0, 2 * np.pi, N_CIRCLE, endpoint=False)
CIRCLE_PTS = np.column_stack([
    5.0 * np.cos(_angles),
    5.0 * np.sin(_angles),
    np.zeros(N_CIRCLE),
])


# ---------------------------------------------------------------------------
# make_polygon
# ---------------------------------------------------------------------------


class TestMakePolygon:
    def test_returns_wire(self):
        wire = cqapi.make_polygon(SQUARE)
        assert isinstance(wire, cqapi.apiWire)

    def test_is_closed(self):
        wire = cqapi.make_polygon(SQUARE)
        assert cqapi.is_closed(wire)

    def test_length_correct(self):
        wire = cqapi.make_polygon(SQUARE)
        assert pytest.approx(cqapi.length(wire), rel=1e-4) == 4

    def test_open_polygon(self):
        pts = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
        wire = cqapi.make_polygon(pts)
        assert not cqapi.is_closed(wire)


# ---------------------------------------------------------------------------
# interpolate_bspline
# ---------------------------------------------------------------------------


class TestInterpolateBspline:
    def test_returns_wire(self):
        wire = cqapi.interpolate_bspline(CIRCLE_PTS, closed=True)
        assert isinstance(wire, cqapi.apiWire)

    def test_closed_spline(self):
        wire = cqapi.interpolate_bspline(CIRCLE_PTS, closed=True)
        assert cqapi.is_closed(wire)

    def test_open_spline(self):
        pts = [[0, 0, 0], [1, 0.5, 0], [2, 0, 0], [3, 0.5, 0]]
        wire = cqapi.interpolate_bspline(pts)
        assert isinstance(wire, cqapi.apiWire)
        assert not cqapi.is_closed(wire)

    def test_length_approx_circumference(self):
        wire = cqapi.interpolate_bspline(CIRCLE_PTS, closed=True)
        expected = 2 * np.pi * 5.0
        assert pytest.approx(cqapi.length(wire), rel=1e-2) == expected

    def test_too_few_points_raises(self):
        with pytest.raises(InvalidCADInputsError):
            cqapi.interpolate_bspline([[0, 0, 0]])

    def test_equal_endpoints_forces_closed(self):
        pts = [*list(CIRCLE_PTS), CIRCLE_PTS[0].tolist()]
        wire = cqapi.interpolate_bspline(pts, closed=False)
        assert cqapi.is_closed(wire)


# ---------------------------------------------------------------------------
# make_face
# ---------------------------------------------------------------------------


class TestMakeFace:
    def test_returns_face(self):
        wire = cqapi.make_polygon(SQUARE)
        face = cqapi.make_face(wire)
        assert isinstance(face, cqapi.apiFace)

    def test_face_is_valid(self):
        wire = cqapi.make_polygon(SQUARE)
        face = cqapi.make_face(wire)
        assert cqapi.is_valid(face)

    def test_face_area(self):
        wire = cqapi.make_polygon(SQUARE)
        face = cqapi.make_face(wire)
        assert pytest.approx(cqapi.area(face), rel=1e-4) == 1


# ---------------------------------------------------------------------------
# revolve_shape
# ---------------------------------------------------------------------------


class TestRevolveShape:
    def _make_face(self):
        # Rectangle in the xz-plane offset from z-axis (for revolve)
        pts = [
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [3.0, 0.0, 0.0],
        ]
        wire = cqapi.make_polygon(pts)
        return cqapi.make_face(wire)

    def test_full_revolve_returns_solid(self):
        face = self._make_face()
        solid = cqapi.revolve_shape(
            face, base=(0, 0, 0), direction=(0, 0, 1), degree=360
        )
        assert isinstance(solid, cqapi.apiSolid)

    def test_full_revolve_is_valid(self):
        face = self._make_face()
        solid = cqapi.revolve_shape(
            face, base=(0, 0, 0), direction=(0, 0, 1), degree=360
        )
        assert cqapi.is_valid(solid)

    def test_partial_revolve(self):
        face = self._make_face()
        solid = cqapi.revolve_shape(
            face, base=(0, 0, 0), direction=(0, 0, 1), degree=180
        )
        assert cqapi.is_valid(solid)


# ---------------------------------------------------------------------------
# offset_wire
# ---------------------------------------------------------------------------


class TestOffsetWire:
    def test_returns_wire(self):
        wire = cqapi.make_polygon(SQUARE)
        offset = cqapi.offset_wire(wire, 0.1)
        assert isinstance(offset, cqapi.apiWire)

    def test_outward_offset_larger(self):
        wire = cqapi.make_polygon(SQUARE)
        offset = cqapi.offset_wire(wire, 0.5)
        assert cqapi.length(offset) > cqapi.length(wire)

    def test_inward_offset_smaller(self):
        wire = cqapi.make_polygon(SQUARE)
        offset = cqapi.offset_wire(wire, -0.1)
        assert cqapi.length(offset) < cqapi.length(wire)

    def test_zero_offset_returns_wire(self):
        wire = cqapi.make_polygon(SQUARE)
        result = cqapi.offset_wire(wire, 0.0)
        assert isinstance(result, cqapi.apiWire)

    def test_straight_wire_raises(self):

        line = cqapi.make_polygon([[0, 0, 0], [1, 0, 0]])
        with pytest.raises(InvalidCADInputsError):
            cqapi.offset_wire(line, 0.1)

    @pytest.mark.parametrize("join", ["arc", "intersect"])
    def test_join_styles(self, join):
        wire = cqapi.make_polygon(SQUARE)
        offset = cqapi.offset_wire(wire, 0.1, join=join)
        assert isinstance(offset, cqapi.apiWire)


# ---------------------------------------------------------------------------
# dist_to_shape
# ---------------------------------------------------------------------------


class TestDistToShape:
    def test_dist_between_parallel_wires(self):
        wire1 = cqapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
        wire2 = cqapi.make_polygon([[0, 1, 0], [1, 1, 0], [1, 1, 0]])
        dist, vectors = cqapi.dist_to_shape(wire1, wire2)
        assert pytest.approx(dist, abs=1e-4) == 1
        assert len(vectors) > 0
        assert len(vectors[0]) == 2  # (point_on_1, point_on_2)

    def test_coincident_shapes_zero_dist(self):
        wire = cqapi.make_polygon(SQUARE)
        dist, _ = cqapi.dist_to_shape(wire, wire)
        assert pytest.approx(dist, abs=1e-6) == 0


# ---------------------------------------------------------------------------
# Shape property queries
# ---------------------------------------------------------------------------


class TestShapeProperties:
    def test_length_of_wire(self):
        wire = cqapi.make_polygon(SQUARE)
        assert pytest.approx(cqapi.length(wire), rel=1e-4) == 4

    def test_area_of_face(self):
        wire = cqapi.make_polygon(SQUARE)
        face = cqapi.make_face(wire)
        assert pytest.approx(cqapi.area(face), rel=1e-4) == 1

    def test_volume_of_solid(self):
        pts = [[3, 0, 0], [4, 0, 0], [4, 0, 1], [3, 0, 1], [3, 0, 0]]
        face = cqapi.make_face(cqapi.make_polygon(pts))
        solid = cqapi.revolve_shape(face, degree=360)
        # Annular cylinder volume = pi*(4^2 - 3^2)*1
        expected = np.pi * (16 - 9) * 1.0
        assert pytest.approx(cqapi.volume(solid), rel=1e-2) == expected

    def test_is_closed_true(self):
        wire = cqapi.make_polygon(SQUARE)
        assert cqapi.is_closed(wire) is True

    def test_is_closed_false(self):
        wire = cqapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        assert cqapi.is_closed(wire) is False

    def test_is_valid(self):
        wire = cqapi.make_polygon(SQUARE)
        assert cqapi.is_valid(wire) is True


# ---------------------------------------------------------------------------
# Tessellation
# ---------------------------------------------------------------------------


class TestTessellate:
    def _make_solid(self):
        pts = [[3, 0, 0], [4, 0, 0], [4, 0, 1], [3, 0, 1], [3, 0, 0]]
        face = cqapi.make_face(cqapi.make_polygon(pts))
        return cqapi.revolve_shape(face, degree=360)

    def test_tessellate_returns_arrays(self):
        solid = self._make_solid()
        verts, tris = cqapi.tessellate(solid, tolerance=0.1)
        assert isinstance(verts, np.ndarray)
        assert isinstance(tris, np.ndarray)
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert tris.ndim == 2
        assert tris.shape[1] == 3

    def test_tessellate_invalid_tolerance(self):
        solid = self._make_solid()
        with pytest.raises(ValueError, match=r"Tolerance must be greater than 0\.0"):
            cqapi.tessellate(solid, tolerance=0.0)

    def test_collect_verts_faces(self):
        solid = self._make_solid()
        verts, faces = cqapi.collect_verts_faces(solid, tesselation=0.1)
        assert verts is not None
        assert faces is not None
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3

    def test_collect_wires(self):
        solid = self._make_solid()
        verts, edges = cqapi.collect_wires(solid, Deflection=0.1)
        assert verts.shape[1] == 3
        assert edges.shape[1] == 2


# ---------------------------------------------------------------------------
# Wire validation helpers
# ---------------------------------------------------------------------------


class TestWireValidation:
    def test_straight_wire_is_straight(self):
        line = cqapi.make_polygon([[0, 0, 0], [1, 0, 0]])
        assert cqapi._wire_is_straight(line) is True

    def test_polygon_is_not_straight(self):
        wire = cqapi.make_polygon(SQUARE)
        assert cqapi._wire_is_straight(wire) is False

    def test_planar_wire_is_planar(self):
        wire = cqapi.make_polygon(SQUARE)
        assert cqapi._wire_is_planar(wire) is True

    def test_spline_circle_is_planar(self):
        wire = cqapi.interpolate_bspline(CIRCLE_PTS, closed=True)
        assert cqapi._wire_is_planar(wire) is True

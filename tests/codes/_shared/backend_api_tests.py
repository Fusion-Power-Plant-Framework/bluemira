# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Shared test base for the geometry backends (``_freecad.api``, ``_cadquery``).

The concrete test classes in ``test_freecad.api.py`` / ``test_cadquery``
inherit from :class:`BackendApiTestsBase` and set ``cadapi`` to the backend
module under test. Any test method moved into this base runs against both
backends via its two subclasses, keeping the API contract in sync.

This module intentionally has no ``test_`` prefix so pytest will not collect it
directly.
"""

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.special import ellipe

from bluemira.base.constants import EPS
from bluemira.codes.error import FreeCADError
from bluemira.geometry.constants import D_TOLERANCE, EPS_FREECAD

if TYPE_CHECKING:
    from types import ModuleType


class BackendApiTestsBase:
    """Shared contract tests for a geometry backend module.

    Subclasses set ``cadapi`` to the backend module under test (e.g.
    ``bluemira.codes.cadapi._freecad.api``). Tests reference ``self.cadapi`` /
    ``cls.cadapi`` rather than importing a module directly, so the same
    method exercises whichever backend the subclass wired in.
    """

    cadapi: ModuleType  # set by each concrete subclass

    square_points = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    closed_square_points = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    def _offsetter(self, wire):
        return self.cadapi.offset_wire(wire, 0.05, join="intersect", open_wire=False)

    def test_multi_offset_wire(self):
        circ = self.cadapi.make_circle(10)
        wire1 = self._offsetter(circ)
        wire2 = self._offsetter(wire1)

        assert (
            self.cadapi.length(circ)
            < self.cadapi.length(wire1)
            < self.cadapi.length(wire2)
        )

    def test_fail_vector_to_numpy(self):
        with pytest.raises(TypeError):
            self.cadapi.vector_to_numpy(list(self.square_points))

    def test_single_vector_to_numpy(self):
        inp = np.array((1.0, 0.5, 2.0))
        vector = self.cadapi.apiVector(*inp)
        arr = self.cadapi.vector_to_numpy(vector)
        assert np.array_equal(arr, inp)

    def test_vector_to_numpy(self):
        vectors = list(starmap(self.cadapi.apiVector, self.square_points))
        arr = self.cadapi.vector_to_numpy(vectors)
        assert np.array_equal(arr, np.array(self.square_points))

    def test_make_polygon(self):
        # open wire
        open_wire = self.cadapi.make_polygon(self.square_points)
        assert len(self.cadapi.vertexes(open_wire)) == 4
        assert len(self.cadapi.edges(open_wire)) == 3
        assert np.array_equal(
            self.cadapi.vertexes(open_wire), np.array(self.square_points)
        )
        assert not self.cadapi.is_closed(open_wire)
        # closed wire
        closed_wire = self.cadapi.make_polygon(self.closed_square_points)
        assert len(self.cadapi.vertexes(closed_wire)) == 4
        assert len(self.cadapi.edges(closed_wire)) == 4
        assert np.array_equal(
            self.cadapi.vertexes(closed_wire), np.array(self.square_points)
        )
        assert self.cadapi.is_closed(closed_wire)

    def test_length(self):
        open_wire = self.cadapi.make_polygon(self.square_points)
        assert self.cadapi.length(open_wire) == pytest.approx(3.0, rel=0, abs=EPS)
        closed_wire = self.cadapi.make_polygon(self.closed_square_points)
        assert self.cadapi.length(closed_wire) == pytest.approx(4.0, rel=0, abs=EPS)

    def test_area(self):
        wire = self.cadapi.make_polygon(self.closed_square_points)
        assert self.cadapi.area(wire) == pytest.approx(0.0, rel=0, abs=EPS)
        face = self.cadapi.make_face(wire)
        assert self.cadapi.area(face) == pytest.approx(1.0, rel=0, abs=EPS)

    def test_center_of_mass(self):
        wire = self.cadapi.make_polygon(self.closed_square_points)
        com = self.cadapi.center_of_mass(wire)
        assert isinstance(com, np.ndarray)
        assert np.array_equal(com, np.array((0.5, 0.5, 0.0)))

    def test_split_circular_wire(self):
        full_circle = self.cadapi.make_circle(
            radius=1.0, center=(1, 0, 0), axis=(0, 1, 0)
        )
        arc_of_circ = self.cadapi.make_circle_arc_3P(
            [0, 0, 0], [1, 1, 0], [2, 0, 0], axis=(0, 1, 0)
        )
        _, semi_circle_upper = self.cadapi.split_wire(full_circle, [0, 0, 0], EPS * 10)
        assert np.allclose(
            self.cadapi.start_point(semi_circle_upper)
            - self.cadapi.start_point(arc_of_circ),
            0,
            atol=D_TOLERANCE,
        )
        assert self.cadapi.split_wire(arc_of_circ, [2, 0, 0], EPS * 10)[1] is None
        assert (
            list(self.cadapi.split_wire(full_circle, [2, 0, 0], EPS * 10)).count(None)
            == 1
        ), (
            "Splitting vertex on the start- AND end-point, "
            "so one of the wires must have zero length."
        )

        with pytest.raises(FreeCADError):
            self.cadapi.split_wire(full_circle, (3, 0, 0), EPS * 10)
        with pytest.raises(FreeCADError):
            self.cadapi.split_wire(arc_of_circ, (3, 0, 0), EPS * 10)

    def test_split_nonperiodic_wire(self):
        closed_wire = self.cadapi.make_polygon(self.closed_square_points)
        bezier = self.cadapi.make_bezier(self.square_points)
        bspline = self.cadapi.interpolate_bspline(self.square_points)
        self.cadapi.split_wire(closed_wire, self.closed_square_points[1], EPS * 10)
        self.cadapi.split_wire(bezier, self.square_points[0], EPS * 10)
        self.cadapi.split_wire(bspline, self.square_points[1], EPS * 10)

    def test_scale_shape(self):
        factor = 2.0
        wire = self.cadapi.make_polygon(self.closed_square_points)
        scaled_wire = self.cadapi.scale_shape(wire.copy(), factor)
        face = self.cadapi.make_face(scaled_wire)
        assert self.cadapi.area(face) == pytest.approx(1.0 * factor**2, rel=0, abs=EPS)
        assert (
            self.cadapi.length(face)
            == self.cadapi.length(scaled_wire)
            == pytest.approx(4.0 * factor, rel=0, abs=EPS)
        )
        face_from_wire = self.cadapi.make_face(wire)
        scaled_face = self.cadapi.scale_shape(face_from_wire.copy(), factor)
        assert self.cadapi.length(scaled_face) == self.cadapi.length(face)
        assert self.cadapi.area(scaled_face) == self.cadapi.area(face)

    def test_discretise(self):
        wire = self.cadapi.make_polygon(self.closed_square_points)
        ndiscr = 10
        points = self.cadapi.discretise(wire, ndiscr)
        assert len(points) == ndiscr
        length_w = self.cadapi.length(wire)
        dl = length_w / float(ndiscr - 1)
        points = self.cadapi.discretise(wire, dl=dl)
        assert len(points) == ndiscr

    def test_discretise_by_edges(self):
        wire = self.cadapi.make_polygon(self.closed_square_points)
        ndiscr = 10
        self.cadapi.discretise_by_edges(wire, ndiscr)

        dl = 0.4
        points1 = self.cadapi.discretise_by_edges(wire, dl=dl)

        dl = 0.4
        points2 = self.cadapi.discretise_by_edges(wire, ndiscr=100, dl=dl)
        assert np.allclose(points1 - points2, 0, atol=D_TOLERANCE)

    def test_start_point_given_polygon(self):
        wire = self.cadapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        start_point = self.cadapi.start_point(wire)
        assert isinstance(start_point, np.ndarray)
        np.testing.assert_equal(start_point, np.array([0, 0, 0]))

    def test_end_point_given_polygon(self):
        wire = self.cadapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        end_point = self.cadapi.end_point(wire)
        assert isinstance(end_point, np.ndarray)
        np.testing.assert_equal(end_point, np.array([1, 1, 0]))

    def test_vertex_to_numpy(self):
        vertexes = list(starmap(self.cadapi.make_vertex, self.square_points))
        arr = self.cadapi.vertex_to_numpy(vertexes)
        assert np.array_equal(arr, np.array(self.square_points))

    def test_change_placement_translation_and_rotation(self):
        """Applying a single placement to a fresh shape must move the
        centre-of-mass by that placement's rigid transform (``p' = R·p + t``).
        """
        # translation-only
        wire = self.cadapi.make_polygon(self.closed_square_points)
        assert np.allclose(
            self.cadapi.center_of_mass(wire), [0.5, 0.5, 0.0], atol=EPS_FREECAD
        )
        translate = self.cadapi.make_placement(
            base=(10.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), angle=0.0
        )
        self.cadapi.change_placement(wire, translate)
        assert np.allclose(
            self.cadapi.center_of_mass(wire), [10.5, 0.5, 0.0], atol=EPS_FREECAD
        )

        # rotation-only, applied to a fresh shape
        wire2 = self.cadapi.make_polygon(self.closed_square_points)
        rotate_z90 = self.cadapi.make_placement(
            base=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), angle=90.0
        )
        self.cadapi.change_placement(wire2, rotate_z90)
        assert np.allclose(
            self.cadapi.center_of_mass(wire2), [-0.5, 0.5, 0.0], atol=EPS_FREECAD
        )

    def test_move_placement_translates_base_only(self):
        """``move_placement`` adds ``vector`` to the placement's base and leaves
        the rotation untouched (FreeCAD ``Placement.move`` semantics).
        """
        p = self.cadapi.make_placement(
            base=(1.0, 2.0, 3.0), axis=(0.0, 0.0, 1.0), angle=45.0
        )
        original_angle = p.Rotation.Angle
        self.cadapi.move_placement(p, [10.0, -5.0, 1.5])
        assert np.allclose(
            [p.Base.x, p.Base.y, p.Base.z], [11.0, -3.0, 4.5], atol=EPS_FREECAD
        )
        assert np.isclose(p.Rotation.Angle, original_angle, atol=EPS_FREECAD)

    def test_move_placement_via_bluemira_placement(self):
        """``BluemiraPlacement.move`` (which routes through ``move_placement``)
        must shift a downstream-applied shape by the same vector.
        """
        from bluemira.geometry.placement import BluemiraPlacement  # noqa: PLC0415

        bp = BluemiraPlacement(base=(0.0, 0.0, 0.0), angle=0.0)
        bp.move([2.0, 3.0, 0.0])
        wire = self.cadapi.make_polygon(self.closed_square_points)
        self.cadapi.change_placement(wire, bp._shape)
        assert np.allclose(
            self.cadapi.center_of_mass(wire), [2.5, 3.5, 0.0], atol=EPS_FREECAD
        )

    @pytest.mark.parametrize(
        ("r_diag", "expected_com"),
        [
            # R = diag(-1,-1, 1): z-axis 180°
            ((-1.0, -1.0, 1.0), (-0.5, -0.5, 0.0)),
            # R = diag( 1,-1,-1): x-axis 180°
            ((1.0, -1.0, -1.0), (0.5, -0.5, 0.0)),
            # R = diag(-1, 1,-1): y-axis 180°
            ((-1.0, 1.0, -1.0), (-0.5, 0.5, 0.0)),
        ],
    )
    def test_make_placement_from_matrix_180deg(self, r_diag, expected_com):
        """``make_placement_from_matrix`` must handle the θ = π Rodrigues
        singularity without producing a NaN axis, and the resulting
        placement must transform geometry as expected.
        """
        matrix = np.eye(4)
        matrix[0, 0], matrix[1, 1], matrix[2, 2] = r_diag
        placement = self.cadapi.make_placement_from_matrix(matrix)
        wire = self.cadapi.make_polygon(self.closed_square_points)
        self.cadapi.change_placement(wire, placement)
        assert np.allclose(
            self.cadapi.center_of_mass(wire), expected_com, atol=EPS_FREECAD
        )

    def test_wire_parameter_at_endpoints_large_tolerance(self):
        """``wire_parameter_at`` must return 0.0 at the start, 1.0 at the
        end, and a sensible value in between — even when the caller passes
        a very large tolerance (``VERY_BIG`` ~ 1e10 is common in
        ``PathInterpolator.to_L``). The tolerance only gates wire-membership,
        it must NOT relax the per-edge selection logic.
        """
        # 2-edge polyline: (0,0,0) — (2,0,1) — (4,0,0). Arc-midpoint at (2,0,1).
        wire = self.cadapi.make_polygon([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [4.0, 0.0, 0.0],
        ])
        very_big = 1e10
        assert self.cadapi.wire_parameter_at(wire, [0.0, 0.0, 0.0], very_big) == (
            pytest.approx(0.0, abs=EPS_FREECAD)
        )
        assert self.cadapi.wire_parameter_at(wire, [2.0, 0.0, 1.0], very_big) == (
            pytest.approx(0.5, abs=EPS_FREECAD)
        )
        assert self.cadapi.wire_parameter_at(wire, [4.0, 0.0, 0.0], very_big) == (
            pytest.approx(1.0, abs=EPS_FREECAD)
        )

    def test_discretise_vs_discretise_by_edges(self):
        wire1 = self.cadapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        wire2 = self.cadapi.make_polygon([[0, 0, 0], [0, 1, 0], [1, 1, 0]])
        wire2 = self.cadapi.reverse_shape(wire2)
        wire = self.cadapi.wire_from_wires([wire1, wire2])

        # ndiscr is chosen so that discretise and discretise_by_edges give the
        # same points (so that a direct comparison is possible).
        points1 = self.cadapi.discretise(wire, ndiscr=5)
        points2 = self.cadapi.discretise_by_edges(wire, ndiscr=4)

        assert np.allclose(points1 - points2, 0, atol=D_TOLERANCE)

    def test_catcherror(self):
        @self.cadapi.catch_caderr(ValueError)
        def func():
            raise FreeCADError("Error")

        with pytest.raises(ValueError):  # noqa: PT011
            func()

    def test_circle_ellipse_arc(self):
        # from make_circle
        arc = self.cadapi.make_circle(start_angle=0, end_angle=180)
        assert self.cadapi.length(arc) == pytest.approx(np.pi)
        assert np.allclose(self.cadapi.start_point(arc), [1.0, 0.0, 0.0])
        assert np.allclose(self.cadapi.end_point(arc), [-1.0, 0.0, 0.0])
        # same arc but using start>end (should normalise)
        arc2 = self.cadapi.make_circle(start_angle=360, end_angle=180)
        assert np.allclose(
            self.cadapi.discretise(arc, 10), self.cadapi.discretise(arc2, 10)
        )

        # from make_circle_arc_3P
        arc3 = self.cadapi.make_circle_arc_3P(
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]
        )
        assert np.allclose(
            self.cadapi.discretise(arc, 10), self.cadapi.discretise(arc3, 10)
        )
        with pytest.raises(FreeCADError):
            self.cadapi.make_circle_arc_3P(
                [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]
            )

        # from make_ellipse
        arc4 = self.cadapi.make_ellipse(start_angle=0, end_angle=90)
        ellipse_major_radius = 2
        ellipse_eccentricity = 0.75
        arc_length = ellipse_major_radius * ellipe(ellipse_eccentricity)
        assert np.isclose(arc_length, self.cadapi.length(arc4), 6)

    def _assert_arc_endpoints_match(self, original, reconstructed, *, reverse: bool):
        """Compare endpoints as an unordered set — invariant under reversal,
        which matters because ``cq.Edge.startPoint`` ignores orientation
        while FreeCAD's equivalent honours it.
        """
        del reverse  # endpoint set is invariant under wire-orientation flips
        orig_pts = sorted(
            map(
                tuple,
                [
                    self.cadapi.start_point(original),
                    self.cadapi.end_point(original),
                ],
            )
        )
        recon_pts = sorted(
            map(
                tuple,
                [
                    self.cadapi.start_point(reconstructed),
                    self.cadapi.end_point(reconstructed),
                ],
            )
        )
        np.testing.assert_allclose(orig_pts, recon_pts, rtol=0, atol=EPS_FREECAD)

    @pytest.mark.parametrize("reverse", [True, False])
    def test_serialise_circle(self, *, reverse: bool):
        """Round-trip make_circle_arc_3P through serialise/deserialise."""
        arc = self.cadapi.make_circle_arc_3P(
            [1, 0, 0], [np.sqrt(2), 0, np.sqrt(2)], [0, 0, 1]
        )
        if reverse:
            arc = self.cadapi.reverse_shape(arc)
        reconstructed = self.cadapi.deserialise_shape(self.cadapi.serialise_shape(arc))
        self._assert_arc_endpoints_match(arc, reconstructed, reverse=reverse)

    @pytest.mark.parametrize("reverse", [True, False])
    def test_serialise_ellipse(self, *, reverse: bool):
        """Round-trip make_ellipse through serialise/deserialise."""
        ellipse_arc = self.cadapi.make_ellipse(start_angle=0, end_angle=190)
        if reverse:
            ellipse_arc = self.cadapi.reverse_shape(ellipse_arc)
        reconstructed = self.cadapi.deserialise_shape(
            self.cadapi.serialise_shape(ellipse_arc)
        )
        self._assert_arc_endpoints_match(ellipse_arc, reconstructed, reverse=reverse)

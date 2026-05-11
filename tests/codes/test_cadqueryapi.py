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

import numpy as np
import cadquery as cq

from bluemira.geometry.error import GeometryError
from tests.codes._shared.backend_api_tests import BackendApiTestsBase  # noqa: E402


class TestCadqueryapi(BackendApiTestsBase):
    cadapi = cadapi

    def test_face_from_wires_tolerant_non_planar_path(self):
        """
        Test non-planar path.
        Create a rectangle on the surface of a cylinder and attempt to create a face from the wire.
        """
        radius = 1.0
        height = 2.0

        vertical_line = cadapi.make_polygon([(radius, 0, 0), (radius, 0, height)])
        cylinder_surface = cadapi.revolve_shape(  # revolve vertical line to create cylinder
            vertical_line,
            base=(0, 0, 0),
            direction=(0, 0, 1),
            degree=360,
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
        curved_faces = cadapi.faces(intersection_result)  # intersect to yield curved cylinder face

        if curved_faces:
            target_face = curved_faces[0]
            wrapped_wire = target_face.outerWire()

        face = cadapi._face_from_wires_tolerant(wrapped_wire, [])  # should succeed via non-planar path

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

        face = cadapi._face_from_wires_tolerant(wire, [])  # this should fail the first path but succeed with makeFromWires

        assert isinstance(face, cq.Face)  # verify face and area
        assert cadapi.area(face) == pytest.approx(1.0, abs=1e-6)

        direct_face = cq.Face.makeFromWires(wire)  # should work directly with makeFromWires
        assert cadapi.area(direct_face) == pytest.approx(1.0, abs=1e-6)

    def test_face_from_wires_tolerant_tolerant_planar_path(self):
        """
        Test the tolerant planar path.
        Create a planar square but add small noise to the points that exceeds the default confusion tolerance.
        Attempt to create a face from the wire, should fallback to SVD.
        """
        # corners of a square with out-of-plane noise in z
        p0 = (0.0, 0.0,  1.5e-5)
        p1 = (1.0, 0.0, -1.2e-5)
        p2 = (1.0, 1.0,  2.1e-5)
        p3 = (0.0, 1.0, -1.8e-5)

        # guarantee closed wire
        noisy_points = [p0, p1, p2, p3, p0]

        wire = cadapi.make_polygon(noisy_points)

        assert wire.IsClosed()

        with pytest.raises(ValueError, match="not planar"):
            cq.Face.makeFromWires(wire)

        face = cadapi._face_from_wires_tolerant(wire, [])  # should succeed via SVD fallback

        assert isinstance(face, cq.Face)
        assert face.isValid()
        assert cadapi.area(face) == pytest.approx(1.0, abs=1e-6)

    def test_face_from_wires_tolerant_with_holes(self):
        """Test planar path with inner hole wires."""

        outer_points = [  # 2x2 flat square
            (-1.0, -1.0, 0.0), 
            ( 1.0, -1.0, 0.0), 
            ( 1.0,  1.0, 0.0), 
            (-1.0,  1.0, 0.0), 
            (-1.0, -1.0, 0.0)
        ]
        outer_wire = cadapi.make_polygon(outer_points)

        inner_points = [  # inner 1x1 square hole
            (-0.5, -0.5, 0.0), 
            ( 0.5, -0.5, 0.0), 
            ( 0.5,  0.5, 0.0), 
            (-0.5,  0.5, 0.0), 
            (-0.5, -0.5, 0.0)
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
            (0.0, 0.0,  1.5e-5), 
            (2.0, 0.0, -1.2e-5), 
            (2.0, 2.0,  2.1e-5), 
            (0.0, 2.0, -1.8e-5), 
            (0.0, 0.0,  1.5e-5),
        ]
        outer_wire = cadapi.make_polygon(outer_points)

        inner_points = [  # noisy 1x1 square hole
            (0.5, 0.5,  1.1e-5), 
            (1.5, 0.5, -1.3e-5), 
            (1.5, 1.5,  2.0e-5), 
            (0.5, 1.5, -1.6e-5), 
            (0.5, 0.5,  1.1e-5),
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
        """Test that inner wires with the same winding as the outer wire are handled correctly."""
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
        outer = cadapi.make_polygon(
            [(0,0,0), (2,0,0), (2,2,0), (0,2,0), (0,0,0)]
        )

        inner_outside = cadapi.make_polygon(  # hole is located completely outside of the outer wire
            [(5,5,0), (6,5,0), (6,6,0), (5,6,0), (5,5,0)]
        )
 
        with pytest.raises(GeometryError):
            face = cadapi._face_from_wires_tolerant(outer, [inner_outside])
            assert not face.isValid()

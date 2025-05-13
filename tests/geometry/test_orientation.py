# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest

import bluemira.codes.cadapi._freecad.api as cadapi
import bluemira.geometry.tools as geo_tools
from bluemira.geometry.error import MixedOrientationWireError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire


class TestOrientation:
    def test_wire(self):
        wire = cadapi.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]],
        )

        bm_wire = BluemiraWire(wire)
        assert bm_wire.boundary[0].Orientation == wire.Orientation
        assert bm_wire.shape.Orientation == wire.Orientation

        wire.reverse()

        bm_wire = BluemiraWire(wire)
        assert bm_wire.boundary[0].Orientation == wire.Orientation
        assert bm_wire.shape.Orientation == wire.Orientation

    def test_face(self):
        wire = cadapi.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]],
        )
        face = cadapi.apiFace(wire)
        bm_face = BluemiraFace._create(face)

        assert bm_face.shape.Orientation == face.Orientation

        face.reverse()
        bm_face = BluemiraFace._create(face)

        assert bm_face.shape.Orientation == face.Orientation

    def test_face_with_hole(self):
        wire = cadapi.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]],
        )
        circle = cadapi.make_circle(radius=10, axis=[0, 1, 0])
        face = cadapi.apiFace([circle, wire])
        bm_face = BluemiraFace._create(face)

        assert bm_face.shape.Orientation == face.Orientation
        assert bm_face.area == face.Area

        face.reverse()
        bm_face = BluemiraFace._create(face)

        assert bm_face.shape.Orientation == face.Orientation
        assert bm_face.area == face.Area

    def test_bad_wire(self):
        wire_1 = cadapi.make_polygon([[0, 0, 0], [1, 0, 0]])
        wire_2 = cadapi.make_polygon([[1, 0, 0], [2, 0, 0]])
        wire_2.reverse()
        with pytest.raises(MixedOrientationWireError):
            bm_wire = BluemiraWire([wire_1, wire_2])


@pytest.mark.parametrize("extrusion", [1, -1])
class TestExtrudeOrientation:
    """
    Helper class for testing orientations are respected after an extrude operation.
    """

    VERTS = (
        [0.5, 0, -0.5],
        [1.5, 0, -0.5],
        [1.5, 0, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0, -0.5],
    )

    @pytest.fixture(autouse=True)
    def _setup(self, extrusion):
        profile = cadapi.make_polygon(self.VERTS)
        face = cadapi.apiFace(profile)
        self.solid_fc = cadapi.extrude_shape(face, (0, extrusion, 0))

        profile = geo_tools.make_polygon(self.VERTS)
        face = BluemiraFace(profile)
        self.solid_bm = geo_tools.extrude_shape(face, (0, extrusion, 0))

    @pytest.mark.parametrize(
        "shapes_name", ["wires", "faces", "edges", "shells", "solids"]
    )
    def test_shapes(self, extrusion, shapes_name: str):  # noqa: ARG002
        shapes_name = shapes_name.capitalize()
        shapes_fc = getattr(self.solid_fc, shapes_name)
        shapes_bm = getattr(self.solid_bm.shape, shapes_name)
        assert all(
            fc.Orientation == bm.Orientation
            for fc, bm in zip(shapes_fc, shapes_bm, strict=False)
        )

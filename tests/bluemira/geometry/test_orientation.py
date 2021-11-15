# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

import bluemira.geometry._freecadapi as cadapi
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace


class TestOrientation:
    def test_wire(self):

        wire = cadapi.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True
        )

        bm_wire = BluemiraWire(wire)
        assert bm_wire._boundary[0].Orientation == wire.Orientation
        assert bm_wire._shape.Orientation == wire.Orientation

        wire.reverse()

        bm_wire = BluemiraWire(wire)
        assert bm_wire._boundary[0].Orientation == wire.Orientation
        assert bm_wire._shape.Orientation == wire.Orientation

    def test_face(self):
        wire = cadapi.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True
        )
        face = cadapi.apiFace(wire)
        bm_face = BluemiraFace._create(face)

        assert bm_face._shape.Orientation == face.Orientation

        face.reverse()
        bm_face = BluemiraFace._create(face)

        assert bm_face._shape.Orientation == face.Orientation

    def test_face_with_hole(self):
        wire = cadapi.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True
        )
        circle = cadapi.make_circle(radius=10, axis=[0, 1, 0])
        face = cadapi.apiFace([circle, wire])
        bm_face = BluemiraFace._create(face)

        assert bm_face._shape.Orientation == face.Orientation
        assert bm_face.area == face.Area

        face.reverse()
        bm_face = BluemiraFace._create(face)

        assert bm_face._shape.Orientation == face.Orientation
        assert bm_face.area == face.Area

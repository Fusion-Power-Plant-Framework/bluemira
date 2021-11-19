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

import numpy as np

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon, offset_wire
from bluemira.geometry.parameterisations import (
    PrincetonD,
    TripleArc,
    SextupleArc,
    TaperedPictureFrame,
)


class TestBluemiraFace:
    @classmethod
    def setup_class(cls):
        polygon = make_polygon(
            [[4, -1, 0], [5, -1, 0], [5, 1, 0], [4, 1, 0]],
            closed=True,
        )
        princeton = PrincetonD().create_shape(n_points=80)
        triple = TripleArc().create_shape()
        sextuple = SextupleArc().create_shape()
        tapered = TaperedPictureFrame().create_shape()
        cls.shapes = [polygon, princeton, triple, sextuple, tapered]

    def test_single_complicated(self):
        for shape in self.shapes:
            face = BluemiraFace(shape)
            assert face.is_valid()
            assert not face.is_null()
            assert face.area > 0.0

    def test_two_complicated(self):
        for shape in self.shapes:
            wire_outer = offset_wire(shape, 0.5, join="arc")
            face = BluemiraFace([wire_outer, shape])
            assert not face.is_null()
            assert face.is_valid()
            assert np.isclose(
                face.area, BluemiraFace(wire_outer).area - BluemiraFace(shape).area
            )

    def test_two_complicated2(self):
        for shape in self.shapes:
            wire_inner = offset_wire(shape, -0.5, join="arc")
            face = BluemiraFace([shape, wire_inner])
            assert not face.is_null()
            assert face.is_valid()
            assert np.isclose(
                face.area, -BluemiraFace(wire_inner).area + BluemiraFace(shape).area
            )

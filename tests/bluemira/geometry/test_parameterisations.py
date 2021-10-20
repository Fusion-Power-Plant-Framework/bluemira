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

import pytest
import numpy as np

from bluemira.utilities.error import OptVariablesError
from bluemira.utilities.opt_variables import OptVariables, BoundedVariable
from bluemira.geometry.error import GeometryParameterisationError
from bluemira.geometry.parameterisations import (
    GeometryParameterisation,
    PrincetonD,
    TripleArc,
    SextupleArc,
    PictureFrame,
    PolySpline,
    TaperedPictureFrame,
)
from bluemira.geometry.tools import make_polygon
from bluemira.geometry._deprecated_tools import get_perimeter
from bluemira.geometry.wire import BluemiraWire


class TestGeometryParameterisation:
    def test_subclass(self):
        class TestPara(GeometryParameterisation):
            def __init__(self):
                variables = OptVariables(
                    [
                        BoundedVariable("a", 0, -1, 1),
                        BoundedVariable("b", 2, 0, 4),
                        BoundedVariable("c", 4, 2, 6, fixed=True),
                    ],
                    frozen=True,
                )
                super().__init__(variables)

            def create_shape(self, **kwargs):
                return BluemiraWire(
                    make_polygon(
                        [
                            [self.variables["a"], 0, 0],
                            [self.variables["b"], 0, 0],
                            [self.variables["c"], 0, 1],
                            [self.variables["a"], 0, 1],
                        ]
                    )
                )

        t = TestPara()
        assert t.name == "TestPara"


class TestPrincetonD:
    @pytest.mark.parametrize("x1", [1, 4, 5])
    @pytest.mark.parametrize("x2", [6, 10, 200])
    @pytest.mark.parametrize("dz", [0, 100])
    def test_princeton_d(self, x1, x2, dz):
        x, z = PrincetonD._princeton_d(x1, x2, dz, 500)
        assert len(x) == 500
        assert len(z) == 500
        assert np.isclose(np.min(x), x1)
        assert np.isclose(np.max(x), x2, rtol=1e-3)
        # check symmetry
        assert np.isclose(np.mean(z), dz)
        assert np.allclose(x[:250], x[250:][::-1])

    def test_error(self):
        with pytest.raises(GeometryParameterisationError):
            PrincetonD._princeton_d(10, 3, 0)

    def test_parameterisation(self):
        p = PrincetonD()
        p.adjust_variable("x1", 4, lower_bound=3, upper_bound=5)
        p.adjust_variable("x2", 16, lower_bound=10, upper_bound=20)
        p.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)

        wire = p.create_shape()
        array = p.create_array(n_points=200)

        assert np.isclose(wire.length, get_perimeter(*array), rtol=1e-3)

    def test_bad_behaviour(self):
        p = PrincetonD()
        with pytest.raises(OptVariablesError):
            p.variables.add_variable(BoundedVariable("new", 0, 0, 0))

        with pytest.raises(OptVariablesError):
            p.variables.remove_variable("x1")


class TestPictureFrame:
    def test_length(self):
        p = PictureFrame(
            {
                "x1": {"value": 4},
                "x2": {"value": 16},
                "z1": {"value": 8},
                "z2": {"value": -8},
                "ri": {"value": 1, "upper_bound": 1},
                "ro": {"value": 1},
            }
        )
        wire = p.create_shape()
        length = 2 * (np.pi + 10 + 14)
        assert np.isclose(wire.length, length)

    def test_no_corners(self):
        p = PictureFrame()
        p.adjust_variable("x1", value=4)
        p.adjust_variable("x2", value=16)
        p.adjust_variable("z1", value=8)
        p.adjust_variable("z2", value=-8)
        p.adjust_variable("ri", value=0, lower_bound=0)
        p.adjust_variable("ro", value=0, lower_bound=0)
        wire = p.create_shape()
        assert len(wire._boundary) == 4
        length = 2 * (12 + 16)
        assert np.isclose(wire.length, length)


class TestTripleArc:
    def test_circle(self):
        p = TripleArc()
        p.adjust_variable("x1", value=4)
        p.adjust_variable("z1", value=0)
        p.adjust_variable("sl", value=0, lower_bound=0)
        p.adjust_variable("f1", value=3)
        p.adjust_variable("f2", value=3)
        p.adjust_variable("a1", value=45)
        p.adjust_variable("a2", value=45)
        wire = p.create_shape()
        assert len(wire._boundary) == 6
        length = 2 * np.pi * 3
        assert np.isclose(wire.length, length)


class TestPolySpline:
    def test_segments(self):
        p = PolySpline()
        p.adjust_variable("flat", value=0)
        wire = p.create_shape()
        assert len(wire._boundary) == 5

        p.adjust_variable("flat", value=1)

        wire = p.create_shape()
        assert len(wire._boundary) == 6


class TestTaperedPictureFrame:
    def test_segments(self):
        p = TaperedPictureFrame()
        wire = p.create_shape()
        assert len(wire._boundary) == 4
        p.adjust_variable("r", value=0)
        wire = p.create_shape()
        assert len(wire._boundary) == 2


class TestSextupleArc:
    def test_segments(self):
        p = SextupleArc()
        wire = p.create_shape()
        assert len(wire._boundary) == 7

    def test_circle(self):
        p = SextupleArc(
            {
                "x1": {"value": 4},
                "z1": {"value": 0},
                "r1": {"value": 4},
                "r2": {"value": 4},
                "r3": {"value": 4},
                "r4": {"value": 4},
                "r5": {"value": 4},
                "a1": {"value": 60, "upper_bound": 60},
                "a2": {"value": 60, "upper_bound": 60},
                "a3": {"value": 60, "upper_bound": 60},
                "a4": {"value": 60, "upper_bound": 60},
                "a5": {"value": 60, "upper_bound": 60},
            }
        )
        wire = p.create_shape()

        assert np.isclose(wire.length, 2 * np.pi * 4)

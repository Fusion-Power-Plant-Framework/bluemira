# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from bluemira.codes._freecadapi import _wire_edges_tangent
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryParameterisationError
from bluemira.geometry.parameterisations import (
    GeometryParameterisation,
    PFrameSection,
    PictureFrame,
    PictureFrameTools,
    PolySpline,
    PrincetonD,
    PrincetonDDiscrete,
    SextupleArc,
    TripleArc,
    _calculate_discrete_constant_tension_shape,
    _princeton_d,
)
from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.magnetostatics.circuits import ArbitraryPlanarRectangularXSCircuit
from bluemira.utilities.opt_variables import OptVariable, OptVariablesFrame, ov


@pytest.mark.parametrize(
    "param_class",
    [PictureFrame, PolySpline, PrincetonD, SextupleArc, TripleArc],
)
def test_read_write(param_class: type[GeometryParameterisation]):
    tempdir = tempfile.mkdtemp()
    try:
        the_path = Path(tempdir, f"{param_class.__name__}.json")
        param = param_class()
        param.to_json(the_path)
        new_param = param_class.from_json(the_path)
        for attr in GeometryParameterisation.__slots__:
            if attr == "_variables":
                assert new_param.variables.as_dict() == param.variables.as_dict()
            else:
                assert getattr(new_param, attr) == getattr(param, attr)
    finally:
        shutil.rmtree(tempdir)


@dataclass
class TGeometryParameterisationOptVariables(OptVariablesFrame):
    a: OptVariable = ov("a", 0, -1, 1)
    b: OptVariable = ov("b", 2, 0, 4)
    c: OptVariable = ov("c", 4, 2, 6, fixed=True)


class TestGeometryParameterisation:
    def test_subclass(self):
        class TestPara(GeometryParameterisation[TGeometryParameterisationOptVariables]):
            def __init__(self):
                variables = TGeometryParameterisationOptVariables()
                super().__init__(variables)

            def create_shape(self, **kwargs):  # noqa: ARG002
                return BluemiraWire(
                    make_polygon([
                        [self.variables.a, 0, 0],
                        [self.variables.b, 0, 0],
                        [self.variables.c, 0, 1],
                        [self.variables.a, 0, 1],
                    ])
                )

        t = TestPara()
        assert t.name == "TestPara"


class TestPrincetonD:
    @pytest.mark.parametrize("x1", [1, 4, 5])
    @pytest.mark.parametrize("x2", [6, 10, 200])
    @pytest.mark.parametrize("dz", [0, 100])
    def test_princeton_d(self, x1, x2, dz):
        x, z = _princeton_d(x1, x2, dz, 500)
        assert len(x) == 500
        assert len(z) == 500
        assert np.isclose(np.min(x), x1)
        assert np.isclose(np.max(x), x2, rtol=1e-3)
        # check symmetry
        assert np.isclose(np.mean(z), dz)
        assert np.allclose(x[:250], x[250:][::-1])

    def test_error(self):
        with pytest.raises(GeometryParameterisationError):
            _princeton_d(10, 3, 0)

    def test_instantiation_fixed(self):
        p = PrincetonD({
            "x1": {"value": 5, "fixed": True},
            "x2": {"value": 14, "fixed": False},
        })
        assert p.variables["x1"].fixed
        assert not p.variables["x2"].fixed


class DummyToroidalFieldSolver:
    def field(x, _, z):  # noqa: N805
        return np.array([np.zeros_like(x), 1.0 / x, np.zeros_like(z)])


class TestPrincetonDDiscrete:
    @pytest.mark.parametrize("x1", [4, 5])
    @pytest.mark.parametrize("x2", [10, 12])
    @pytest.mark.parametrize("n_tf", [12, 18])
    def test_princeton_d_discrete(self, x1, x2, n_tf):
        x, z = _calculate_discrete_constant_tension_shape(
            x1,
            x2,
            n_tf,
            0.25,
            0.1,
            40,
            solver=ArbitraryPlanarRectangularXSCircuit,
            tolerance=1e-4,
        )

        assert np.isclose(np.min(x), x1)
        assert np.isclose(np.max(x), x2)
        # check symmetry
        assert np.isclose(np.sum(x[: len(x) // 2 + 1]), np.sum(x[len(x) // 2 :]))
        assert np.isclose(np.sum(z), 0.0)

    def test_princeton_d_discrete_bs(self):
        x, z = _calculate_discrete_constant_tension_shape(
            4, 16, 16, 0.25, 0.1, 100, solver=BiotSavartFilament, tolerance=1e-4
        )

        assert np.isclose(np.min(x), 4.0)
        assert np.isclose(np.max(x), 16.0)
        # check symmetry
        assert np.isclose(np.sum(x[: len(x) // 2 + 1]), np.sum(x[len(x) // 2 :]))
        assert np.isclose(np.sum(z), 0.0)

    def test_verify_princeton_d_discrete(self):
        """
        This to verify that we recover the semi-analytical Princeton-D form
        with this numerical prodecure
        """
        x, z = _calculate_discrete_constant_tension_shape(
            4.0, 16, 1, 0.0, 0.0, 300, DummyToroidalFieldSolver, tolerance=1e-3
        )
        c1 = Coordinates({"x": x, "z": z})
        xd, zd = _princeton_d(4.0, 16.0, 0.0, 200)
        c2 = Coordinates({"x": xd, "z": zd})
        assert not c1.closed
        assert np.isclose(c1.length, c2.length, rtol=1e-2)
        c1.close()
        c2.close()
        assert np.isclose(c1.length, c2.length, rtol=1e-4)

    def test_princeton_d_discrete_parameterisation_init_error(self):
        with pytest.raises(GeometryParameterisationError):
            PrincetonDDiscrete({
                "x1": {"value": 5, "fixed": True},
                "x2": {"value": 14, "fixed": False},
                "dz": {"value": 0.1},
            })

    def test_princeton_d_discrete_parameterisation_call_error(self):
        with pytest.raises(GeometryParameterisationError):
            PrincetonDDiscrete({
                "x1": {"value": 14, "fixed": True},
                "x2": {"value": 5, "fixed": False},
                "dz": {"value": 0.1},
            })

    def test_princeton_d_discrete_parameterisation_init_error_2(self):
        with pytest.raises(GeometryParameterisationError):
            PrincetonDDiscrete(
                {
                    "x1": {"value": 5, "fixed": True},
                    "x2": {"value": 14, "fixed": False},
                    "dz": {"value": 0.1},
                },
                n_TF=16,
            )

    def test_princeton_d_disctrete_shape(self):
        parameterisation = PrincetonDDiscrete(
            {
                "x1": {"value": 5, "fixed": True},
                "x2": {"value": 14, "fixed": False},
                "dz": {"value": 0.1},
            },
            n_TF=16,
            tf_wp_depth=0.7,
            tf_wp_width=0.4,
            n_points=30,
            tolerance=0.01,
        )
        shape = parameterisation.create_shape()
        assert shape.is_closed()
        com = shape.center_of_mass
        bb = shape.bounding_box
        assert np.isclose(bb.x_min, 5.0)
        assert np.isclose(bb.x_max, 14.0, rtol=1e-3)
        assert np.isclose(com[1], 0.0)
        assert np.isclose(com[2], 0.1)


class TestPictureFrame:
    def test_length(self):
        p = PictureFrame({
            "x1": {"value": 4},
            "x2": {"value": 16},
            "z1": {"value": 8},
            "z2": {"value": -8},
            "ri": {"value": 1, "upper_bound": 1},
            "ro": {"value": 1},
        })
        wire = p.create_shape()
        length = 2 * (np.pi + 10 + 14)
        assert np.isclose(wire.length, length)

    def test_no_corners(self):
        p = PictureFrame()
        p.adjust_variable("x1", value=4, lower_bound=4, upper_bound=5)
        p.adjust_variable("x2", value=16, lower_bound=14, upper_bound=18)
        p.adjust_variable("z1", value=8, lower_bound=5, upper_bound=15)
        p.adjust_variable("z2", value=-8, lower_bound=-15, upper_bound=-5)
        p.adjust_variable("ri", value=0, lower_bound=0, upper_bound=2)
        p.adjust_variable("ro", value=0, lower_bound=0, upper_bound=5)
        wire = p.create_shape()
        assert len(wire._boundary) == 4
        length = 2 * (12 + 16)
        assert np.isclose(wire.length, length)

    def test_ordering(self):
        p = PictureFrame({
            "x1": {"value": 4},
            "x2": {"value": 16},
            "z1": {"value": 8},
            "z2": {"value": -8},
            "ri": {"value": 1, "upper_bound": 1},
            "ro": {"value": 1},
        })
        wire = p.create_shape()
        assert _wire_edges_tangent(wire.shape)


class TestComplexPictureFrame:
    @pytest.mark.parametrize(
        ("upper", "lower", "inner", "result"),
        [
            ("CURVED", "CURVED", "TAPERED_INNER", 56.331),
            ("CURVED", "FLAT", "TAPERED_INNER", 54.714),
            ("FLAT", "CURVED", "TAPERED_INNER", 54.714),
            (
                PFrameSection.CURVED,
                PFrameSection.CURVED,
                PFrameSection.TAPERED_INNER,
                56.331,
            ),
            (
                PFrameSection.CURVED,
                PFrameSection.FLAT,
                PFrameSection.TAPERED_INNER,
                54.714,
            ),
            (
                PFrameSection.FLAT,
                PFrameSection.CURVED,
                PFrameSection.TAPERED_INNER,
                54.714,
            ),
            ("CURVED", "CURVED", None, 57.6308),
            ("CURVED", "FLAT", None, 56.014),
            ("FLAT", "CURVED", None, 56.014),
            (PFrameSection.CURVED, PFrameSection.CURVED, None, 57.6308),
            (PFrameSection.CURVED, PFrameSection.FLAT, None, 56.014),
            (PFrameSection.FLAT, PFrameSection.CURVED, None, 56.014),
        ],
    )
    def test_length(self, upper, lower, inner, result):
        p = PictureFrame(upper=upper, lower=lower, inner=inner)
        wire = p.create_shape()
        assert np.isclose(wire.length, result, rtol=1e-4, atol=1e-5)
        if p.upper == PFrameSection.CURVED and p.lower == PFrameSection.CURVED:
            assert p.variables.ro.fixed
        if p.upper == PFrameSection.FLAT and p.lower == PFrameSection.FLAT:
            assert p.variables.x3.fixed
            assert p.variables.z1_peak.fixed
            assert p.variables.z2_peak.fixed
        if p.inner != PFrameSection.TAPERED_INNER:
            assert p.variables.x4.fixed
            assert p.variables.z3.fixed

    @pytest.mark.parametrize(
        ("upper", "lower", "inner"),
        [
            ("FLAT", "FLAT", "TAPERED_INNER"),
            ("CURVED", "CURVED", "TAPERED_INNER"),
            ("CURVED", "FLAT", "TAPERED_INNER"),
            ("FLAT", "CURVED", "TAPERED_INNER"),
            (PFrameSection.FLAT, PFrameSection.FLAT, PFrameSection.TAPERED_INNER),
            (PFrameSection.CURVED, PFrameSection.CURVED, PFrameSection.TAPERED_INNER),
            (PFrameSection.CURVED, PFrameSection.FLAT, PFrameSection.TAPERED_INNER),
            (PFrameSection.FLAT, PFrameSection.CURVED, PFrameSection.TAPERED_INNER),
            ("FLAT", "FLAT", None),
            ("CURVED", "CURVED", None),
            ("CURVED", "FLAT", None),
            ("FLAT", "CURVED", None),
            (PFrameSection.FLAT, PFrameSection.FLAT, None),
            (PFrameSection.CURVED, PFrameSection.CURVED, None),
            (PFrameSection.CURVED, PFrameSection.FLAT, None),
            (PFrameSection.FLAT, PFrameSection.CURVED, None),
        ],
    )
    def test_ordering(self, upper, lower, inner):
        p = PictureFrame(upper=upper, lower=lower, inner=inner)
        wire = p.create_shape()
        assert _wire_edges_tangent(wire.shape)

    def test_tapered_segments(self):
        p = PictureFrame(inner="TAPERED_INNER")
        wire = p.create_shape()
        assert len(wire._boundary) == 4
        p.adjust_variable("ri", value=0, lower_bound=0, upper_bound=2)
        p.adjust_variable("ro", value=0, lower_bound=0, upper_bound=5)
        wire = p.create_shape()
        assert len(wire._boundary) == 4

    @pytest.mark.parametrize(
        ("x_in", "x_mid", "z_bot", "z_taper", "z_top", "r_min"),
        [
            # z_in is not z2<-z_in<0<z_in<z1
            (0.5, 2.0, 4.0, 4.0, 3.0, 0.5),
            (0.5, 2.0, -4.0, 4.0, 3.0, 0.5),
            (0.5, 2.0, -2.0, 3.0, -3.0, 0.5),
            (0.5, 2.0, -4.0, 3.0, -3.0, 0.5),
            # ridiculously large radius minimum radius,
            # so taper cannot be as deep as required.
            (0.3, 2.0, -3.0, 0.1, 3.0, 100),
            # Taper required is so deep that it makes it turns into an Omega shape
            # rather than the simple dome shape.
            (0.3, 2.0, -3.0, 0.1, 3.0, 0.5),
        ],
    )
    def test_inner_taper_xz(self, x_in, x_mid, z_bot, z_taper, z_top, r_min):
        """
        Check that the tapered inner leg of the PictureFrame
        throws an error when it's non-sensical.
        """
        _make_taper = PictureFrameTools._make_tapered_inner_leg
        with pytest.raises(GeometryParameterisationError):
            _make_taper(x_in, x_mid, z_bot, z_taper, z_top, r_min)

    @pytest.mark.parametrize(
        "vals",
        [
            {"inner": "CURVED"},
            {"upper": "TAPERED_INNER"},
            {"lower": "TAPERED_INNER"},
            {"inner": PFrameSection.CURVED},
            {"upper": PFrameSection.TAPERED_INNER},
            {"lower": PFrameSection.TAPERED_INNER},
        ],
    )
    def test_bad_combinations_raise_ValueError(self, vals):
        with pytest.raises(ValueError):  # noqa: PT011
            PictureFrame(**vals).create_shape()

    @pytest.mark.parametrize(
        "vals", [{"inner": "hiiii"}, {"upper": "tpi"}, {"lower": "hello"}]
    )
    def test_unknown_keys_raise_KeyError(self, vals):
        with pytest.raises(KeyError):
            PictureFrame(**vals)


class TestTripleArc:
    def test_circle(self):
        p = TripleArc()
        p.adjust_variable("x1", value=4)
        p.adjust_variable("dz", value=0)
        p.adjust_variable("sl", value=0, lower_bound=0)
        p.adjust_variable("f1", value=3)
        p.adjust_variable("f2", value=3)
        p.adjust_variable("a1", value=45)
        p.adjust_variable("a2", value=45)
        wire = p.create_shape()
        assert len(wire._boundary) == 6
        length = 2 * np.pi * 3
        assert np.isclose(wire.length, length)

    def test_too_big_circle(self):
        # curve too long, defined over too much angle.
        p = TripleArc()
        for i in range(1, 3):
            p.adjust_variable(f"a{i}", getattr(p.variables, f"a{i}").upper_bound)
        with pytest.raises(GeometryParameterisationError):
            p.create_shape()

        # curve too tight, crosses the symmetry plane too early due to strict conditions.
        p.variables.a1.adjust(120)
        p.variables.a2.adjust(40)
        p.variables.f1.adjust(2)
        p.variables.f2.adjust(12)
        p.variables.sl.adjust(5)
        with pytest.raises(GeometryParameterisationError):
            p.create_shape()

    def test_plot(self):
        p = TripleArc()
        p.variables.dz.adjust(1.0)
        p.plot(labels=True)


class TestPolySpline:
    def test_segments(self):
        p = PolySpline()
        p.adjust_variable("flat", value=0)
        wire = p.create_shape()
        assert len(wire._boundary) == 5

        p.adjust_variable("flat", value=1)

        wire = p.create_shape()
        assert len(wire._boundary) == 6


class TestSextupleArc:
    def test_segments(self):
        p = SextupleArc()
        wire = p.create_shape()
        assert len(wire._boundary) == 7

    def test_circle(self):
        p = SextupleArc({
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
        })
        wire = p.create_shape()

        assert np.isclose(wire.length, 2 * np.pi * 4)

    def test_too_wide_cirlce(self):
        """
        Top half of this circle is too wide, so the bottom half of the circle has to
        curl past the start point (crossing over itself).
        """
        p = SextupleArc()
        p.variables.a1.adjust(50)
        p.variables.a2.adjust(80)
        p.variables.a3.adjust(100)
        p.variables.a4.adjust(80)
        p.variables.a5.adjust(20)
        p.variables.r1.adjust(12)
        p.variables.r2.adjust(12)
        p.variables.r3.adjust(12)
        p.variables.r4.adjust(10)
        p.variables.r5.adjust(10)
        with pytest.raises(GeometryParameterisationError):
            p.create_shape()

    def test_too_narrow_circle(self):
        """
        Top half of this circle is too narrow, so the bottom half crosses into the
        territory where x< starting x-coordinate.
        """
        p = SextupleArc()
        p.variables.a1.adjust(50)
        p.variables.a2.adjust(80)
        p.variables.a3.adjust(100)
        p.variables.a4.adjust(80)
        p.variables.a5.adjust(20)
        p.variables.r1.adjust(4)
        p.variables.r2.adjust(4)
        p.variables.r3.adjust(4)
        p.variables.r4.adjust(10)
        p.variables.r5.adjust(10)
        with pytest.raises(GeometryParameterisationError):
            p.create_shape()

    def test_too_big_circle(self):
        p = SextupleArc()
        for i in range(1, 6):
            p.adjust_variable(f"a{i}", getattr(p.variables, f"a{i}").upper_bound)
        with pytest.raises(GeometryParameterisationError):
            p.create_shape()

    def test_plot(self):
        p = SextupleArc()
        p.plot(labels=True)

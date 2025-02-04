# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import json
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.codes.error import InvalidCADInputsError
from bluemira.geometry.error import GeometryError
from bluemira.geometry.parameterisations import (
    PictureFrame,
    PolySpline,
    PrincetonD,
    TripleArc,
)
from bluemira.geometry.tools import (
    deserialise_shape,
    distance_to,
    make_polygon,
    offset_wire,
)


class TestOffset:
    @classmethod
    def setup_class(cls):
        cls.p_wire = PrincetonD().create_shape(label="princeton")
        cls.pf_wire = PictureFrame().create_shape(label="pict_frame")
        cls.t_wire = TripleArc().create_shape(label="triple")
        cls.tpf_wire = PictureFrame(inner="TAPERED_INNER").create_shape(label="tpf")
        cls.ps_wire = PolySpline().create_shape(label="poly")
        cls.rect_wire = make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="sqaure"
        )
        cls.tri_wire = make_polygon(
            [[0, 0, 0], [1, 0, 0], [0.5, 0, 0.5]], closed=True, label="triangle"
        )

    @property
    def all_wires(self):
        return [
            self.p_wire,
            self.pf_wire,
            self.t_wire,
            self.tpf_wire,
            self.ps_wire,
            self.rect_wire,
            self.tri_wire,
        ]

    @pytest.mark.parametrize("join", ["intersect", "arc"])
    def test_simple(self, join):
        for wire in self.all_wires:
            new_wire = offset_wire(wire, 0.0, label="new", join=join)
            assert np.isclose(wire.length, new_wire.length)
            assert new_wire.label == "new"

        for wire in self.all_wires:
            new_wire = offset_wire(wire, 0.0, label="new")
            assert np.isclose(wire.length, new_wire.length)
            assert new_wire.label == "new"

    def test_bad_princeton(self):
        p = PrincetonD({
            "x1": {"value": 4},
            "x2": {"value": 14},
            "dz": {"value": 0},
        })
        wire = p.create_shape()
        offset = offset_wire(wire, -0.5, join="intersect")
        assert offset.is_valid()

    @pytest.mark.parametrize("join", ["intersect", "arc"])
    def test_orientation(self, join):
        for wire in self.all_wires:
            new_wire = offset_wire(wire, 1.0, join=join)
            assert new_wire.length > wire.length
            # Check that discretisation doesn't break
            new_wire.discretise(ndiscr=1000, byedges=True)

        for wire in self.all_wires:
            new_wire = offset_wire(wire, -0.15, join=join)
            assert new_wire.length < wire.length
            # Check that discretisation doesn't break
            new_wire.discretise(ndiscr=1000, byedges=True)

    @pytest.mark.parametrize("wire", [PrincetonD().create_shape(with_tangency=True)])
    @pytest.mark.parametrize(
        "join",
        [
            pytest.param("intersect", marks=pytest.mark.xfail(reason="tangency #3586")),
            "arc",
        ],
    )
    def test_princetonD_tangent_offset(self, join, wire):
        new_wire = offset_wire(wire, 1.0, join=join)
        assert new_wire.length > wire.length
        # Check that discretisation doesn't break
        new_wire.discretise(ndiscr=1000, byedges=True)

        new_wire = offset_wire(wire, -0.15, join=join)
        assert new_wire.length < wire.length
        # Check that discretisation doesn't break
        new_wire.discretise(ndiscr=1000, byedges=True)

    def test_1_offset(self):
        o_rect = offset_wire(self.rect_wire, 0.25, join="intersect")
        assert self.rect_wire.length == pytest.approx(4.0, rel=0, abs=EPS)
        assert o_rect.length == pytest.approx(6.0, rel=0, abs=EPS)

    def test_errors(self):
        with pytest.raises(KeyError):
            offset_wire(self.rect_wire, 1.0, join="bad")

    def test_straight_line(self):
        straight = make_polygon([[0, 0, 0], [0, 0, 1]], label="straight_line")

        with pytest.raises(InvalidCADInputsError):
            offset_wire(straight, 1.0)

    def test_non_planar(self):
        non_planar = make_polygon([[0, 0, 0], [1, 0, 0], [2, 0, 1], [3, 1, 1]])
        with pytest.raises(InvalidCADInputsError):
            offset_wire(non_planar, 1.0)

    def test_offset_destroyed_shape_error(self):
        with pytest.raises(GeometryError):
            # This will offset the triangle such that it no longer exists
            offset_wire(self.tri_wire, -1.0)


class TestFallBackOffset:
    @classmethod
    def setup_class(cls):
        fp = get_bluemira_path("geometry/test_data", subfolder="tests")

        with open(Path(fp, "offset_wire2022-04-08_10-19-27.json")) as file:
            data = json.load(file)

        cls.wire = deserialise_shape(data)

    @pytest.mark.parametrize("fallback_method", ["square"])
    @pytest.mark.parametrize("join", ["arc", "intersect"])
    @pytest.mark.parametrize("delta", [(0.75), (-0.75)])
    def test_primitive_offsetting_catch(self, delta, join, fallback_method):
        """
        This is a test for offset operations on wires that have failed primitive
        offsetting.
        """
        result = offset_wire(
            self.wire, delta, join, fallback_method=fallback_method, open_wire=False
        )

        np.testing.assert_allclose(
            distance_to(self.wire, result)[0], abs(delta), rtol=1e-2
        )

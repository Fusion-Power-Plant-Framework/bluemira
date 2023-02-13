# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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

import os
from unittest import mock

import numpy as np
import pytest

from bluemira.geometry.error import GeometryError
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import make_polygon
from eudemo.ivc.panelling import WrappedString


class TestWrappedString:

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    @staticmethod
    def make_princeton_d():
        return PrincetonD(
            {"x1": {"value": 4}, "x2": {"value": 14}, "dz": {"value": 0}}
        ).create_shape()

    def test_created_shape_points_match_snapshot(self):
        """
        The data for this test was generated using the
        BLUEPRINT.geometry.stringgeom.String class from BLUEPRINT.

        WrappedString is intended to replicate the functionality of
        BLUEPRINT's String, so we check we get consistent results here.

        The code used to generate the test data:

        .. code-block:: python

            from BLUEPRINT.geometry.stringgeom import String
            from bluemira.geometry.parameterisations import PrincetonD

            shape = PrincetonD().create_shape()
            points = shape.discretize()
            s = String(points, angle=20, dx_min=0.5, dx_max=2.5)

            np.save("panelling_ref_data.npy", s.new_points)

        Using bluemira 437a1c10, and BLUEPRINT e3fb8d1c.
        """
        boundary = self.make_princeton_d()
        string = WrappedString(
            boundary,
            var_dict={
                "max_angle": {"value": 20},
                "min_segment_len": {"value": 0.5},
                "max_segment_len": {"value": 2.5},
            },
            n_boundary_points=100,
        )
        string_shape = string.create_shape()

        new_points = string_shape.discretize(ndiscr=100, byedges=True)

        ref_data = np.load(os.path.join(self.DATA_DIR, "panelling_ref_data.npy"))
        ref_wire = make_polygon(ref_data, closed=True)
        ref_points = ref_wire.discretize(ndiscr=100, byedges=True)
        np.testing.assert_almost_equal(new_points.xyz, ref_points)

    @pytest.mark.parametrize("max_len", [1.0001, 1.5, 4])
    def test_shape_ineq_constraint_lt_0_given_max_segment_len_gt_min(self, max_len):
        shape = WrappedString(
            self.make_princeton_d(),
            var_dict={
                "min_segment_len": {"value": 1},
                "max_segment_len": {"value": max_len},
            },
        )

        constraint = np.zeros(1)
        x = shape.variables.get_normalised_values()
        grad = np.zeros((1, 3))
        shape.shape_ineq_constraints(constraint, x, grad)

        assert constraint.size == 1
        assert constraint[0] < 0
        np.testing.assert_allclose(grad, [[0, 1, -1]])

    @pytest.mark.parametrize("max_len", [0.9999, 0.5, 0])
    def test_shape_ineq_constraint_gt_0_given_max_segment_len_lt_min(self, max_len):
        shape = WrappedString(
            self.make_princeton_d(),
            var_dict={
                "min_segment_len": {"value": 1},
                "max_segment_len": {"value": max_len},
            },
        )

        constraint = np.zeros(1)
        x = shape.variables.get_normalised_values()
        grad = np.zeros((1, 3))
        shape.shape_ineq_constraints(constraint, x, grad)

        assert constraint.size == 1
        assert constraint[0] > 0
        np.testing.assert_allclose(grad, [[0, 1, -1]])

    @pytest.mark.parametrize("max_len", [0.9999, 0.5, 0])
    def test_GeometryError_on_create_shape_given_min_gt_max_segment_len(self, max_len):
        shape = WrappedString(
            self.make_princeton_d(),
            var_dict={
                "min_segment_len": {"value": 1},
                "max_segment_len": {"value": max_len},
            },
        )

        with pytest.raises(GeometryError):
            shape.create_shape()

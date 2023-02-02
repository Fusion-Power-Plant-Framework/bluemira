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
import numpy as np
import pytest

from bluemira.geometry.parameterisations import (
    GeometryParameterisation,
    PictureFrame,
    PrincetonD,
    SextupleArc,
    TripleArc,
)
from bluemira.geometry.tools import make_circle, make_polygon, signed_distance
from bluemira.optimisation import optimise_geometry


class TestGeometry:
    def test_simple_optimisation_with_keep_out_zone(self):
        def length(geom: GeometryParameterisation):
            return geom.create_shape().length

        # Create a PictureFrame with un-rounded edges (a rectangle) and
        # a circular keep-out zone within it.
        # We expect the rectangle to contract such that the distance
        # between the parallel edges is equal to the diameter of the
        # keep-out zone.
        koz_radius = 4.5
        koz_center = [10, 0, 0]
        keep_out_zone = make_circle(radius=koz_radius, center=koz_center, axis=[0, 1, 0])
        parameterisation = PictureFrame(
            {
                # Make sure bounds are set within the keep-out zone so
                # we know it's doing some work
                "x1": {"value": 4.5, "upper_bound": 6, "lower_bound": 3},
                "x2": {"value": 16, "upper_bound": 17.5, "lower_bound": 14.0},
                "z1": {"value": 8, "upper_bound": 15, "lower_bound": 2.5},
                "z2": {"value": -6, "upper_bound": -2.5, "lower_bound": -15},
                "ri": {"value": 0, "fixed": True},
                "ro": {"value": 0, "fixed": True},
            }
        )

        opt_result = optimise_geometry(
            parameterisation,
            length,
            keep_out_zones=[keep_out_zone],
            algorithm="SLSQP",
            opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6},
        )

        optimised_shape = opt_result.geom.create_shape()
        np.testing.assert_array_almost_equal(
            list(optimised_shape.center_of_mass), koz_center, decimal=2
        )
        bounds = optimised_shape.bounding_box
        assert bounds.x_max - bounds.x_min == pytest.approx(2 * koz_radius, rel=0.01)
        assert bounds.z_max - bounds.z_min == pytest.approx(2 * koz_radius, rel=0.01)

    def test_princeton_d(self):
        parameterisation = PrincetonD(
            {
                "x1": {"value": 5.5, "upper_bound": 10, "lower_bound": 5},
                "x2": {"value": 16, "upper_bound": 19.5, "lower_bound": 12.5},
                "dz": {"value": 1, "upper_bound": 1.5, "lower_bound": -1.5},
            }
        )
        # TODO(hsaunders1904): think about whether we want to keep the
        # original parameterisation constant, or whether we change it
        # in-place. Add a test for the behaviour
        original_length = parameterisation.create_shape().length
        koz_radius = 4.5
        koz_center = [12.5, 0, 0]
        keep_out_zone = make_circle(radius=koz_radius, center=koz_center, axis=[0, 1, 0])

        opt_result = optimise_geometry(
            parameterisation,
            lambda geom: geom.create_shape().length,
            keep_out_zones=[keep_out_zone],
            opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6},
        )

        opt_shape = opt_result.geom.create_shape()
        # The x-extrema of the PrincetonD should be right on the edge of
        # the keep-out zone
        assert opt_result.geom.variables["x1"].value == pytest.approx(
            koz_center[0] - koz_radius, rel=1e-3
        )
        assert opt_result.geom.variables["x2"].value == pytest.approx(
            koz_center[0] + koz_radius, rel=1e-3
        )
        assert opt_shape.length < original_length
        signed_dist = signed_distance(opt_shape, keep_out_zone)
        # The PrincetonD should fully enclose the keep-out zone
        np.testing.assert_array_less(signed_dist, np.zeros_like(signed_dist))

    def test_maximise_length_with_keep_in_zone_with_TripleArc(self):
        # Make a rectangular keep-in zone and check the TripleArc
        # respects the bounds. Note that in this example we expect the
        # height of the TripleArc to match the height of the keep-in
        # zone, and the width of the TripleArc should be less than the
        # width of the keep-in zone.
        arc = TripleArc(
            {
                "x1": {"value": 3.5, "lower_bound": 2.75},
                "dz": {"value": 0},
            }
        )
        kiz = make_polygon(
            np.array([[3, 13, 13, 3], [0, 0, 0, 0], [-5, -5, 6, 6]]), closed=True
        )

        result = optimise_geometry(
            arc,
            f_objective=lambda geom: -geom.create_shape().length,
            keep_in_zones=(kiz,),
            algorithm="SLSQP",
            opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6},
        )

        opt_shape = result.geom.create_shape()
        assert opt_shape.center_of_mass[2] == pytest.approx(0.5, rel=0.01)
        bbox = opt_shape.bounding_box
        assert bbox.z_min == pytest.approx(-5, rel=0.01)
        assert bbox.z_max == pytest.approx(6, rel=0.01)
        assert bbox.x_min >= 3
        assert bbox.x_max * 0.99905 <= 13

    def test_maximise_angles_with_TripleArc(self):
        # Maximise the sum of the angles in a TripleArc. The shape's
        # constraint should guarantee that the sum is never greater than
        # 180 degrees
        angle_vars = ["a1", "a2"]

        def sum_angles(geom: TripleArc) -> float:
            angles = [geom.variables[a].normalised_value for a in angle_vars]
            return np.sum(angles)

        def d_sum_angles(geom: TripleArc) -> np.ndarray:
            grad = np.zeros(len(geom.variables.get_normalised_values()))
            grad[geom._get_x_norm_index("a1")] = 1
            grad[geom._get_x_norm_index("a2")] = 1
            return grad

        arc = TripleArc()

        result = optimise_geometry(
            arc,
            f_objective=lambda geom: -sum_angles(geom),
            df_objective=lambda geom: -d_sum_angles(geom),
            opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6},
        )

        angles = [result.geom.variables[a].value for a in angle_vars]
        # The shape's inequality constraint should mean the sum is
        # strictly less than 180
        assert sum(angles) < 180
        # The maximisation should mean the angles approximately sum to 180
        assert sum(angles) == pytest.approx(180, rel=1e-3)

    def test_maximise_angles_with_SextupleArc(self):
        # Run an optimisation to maximise the size of the angles in a
        # SextupleArc. The shape constraint should ensure the angles do
        # not sum to more than 360 degrees.
        arc = SextupleArc()
        angle_vars = ["a1", "a2", "a3", "a4", "a5"]

        def sum_angles(geom: SextupleArc) -> float:
            angles = [geom.variables[a].normalised_value for a in angle_vars]
            return np.sum(angles)

        result = optimise_geometry(
            arc,
            f_objective=lambda geom: -sum_angles(geom),
            opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6},
            algorithm="SLSQP",
        )

        angles = [result.geom.variables[a].value for a in angle_vars]
        # The shape's inequality constraint should mean the sum is
        # strictly less than 360
        assert sum(angles) < 360
        # The maximisation should mean the angles approximately sum to 360
        assert sum(angles) == pytest.approx(360, rel=1e-2)

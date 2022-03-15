# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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

from typing import Callable

import numpy as np
import pytest
from scipy.interpolate import interp1d

from bluemira.geometry.parameterisations import PictureFrame
from bluemira.geometry.tools import find_clockwise_angle_2d
from bluemira.geometry.varied_offset import variable_offset_curve, varied_offset_function
from bluemira.geometry.wire import BluemiraWire


class TestVariableOffsetCurve:

    _fixtures = [
        {
            "minor_distance": np.sqrt(2),
            "minor_angle": np.pi / 4,
            "major_distance": np.sqrt(8),
            "major_angle": 3 * np.pi / 2,
            "origin": np.array([1, 1]),
            "num_points": 20,
        },
        {
            "minor_distance": np.sqrt(125),
            "minor_angle": np.radians(20),
            "major_distance": 18,
            "major_angle": np.radians(320),
            "origin": np.array([-11, 0]),
            "num_points": 25,
        },
    ]

    @pytest.mark.parametrize("params", _fixtures)
    def test_first_point_is_at_minor_angle_from_origin(self, params):
        coords = variable_offset_curve(**params)

        angle = np.radians(
            find_clockwise_angle_2d(np.array([-1, 0]), coords[:, 0] - params["origin"])
        )
        assert angle == pytest.approx(params["minor_angle"])

    @pytest.mark.parametrize("params", _fixtures)
    def test_distance_from_first_coord_is_minor_distance(self, params):
        coords = variable_offset_curve(**params)

        dist = np.linalg.norm(coords[:, 0] - params["origin"])
        assert dist == pytest.approx(params["minor_distance"])

    @pytest.mark.parametrize("params", _fixtures)
    def test_final_point_is_at_major_angle_from_origin(self, params):
        coords = variable_offset_curve(**params)

        angle = np.radians(
            find_clockwise_angle_2d(np.array([-1, 0]), coords[:, -1] - params["origin"])
        )
        assert angle == pytest.approx(params["major_angle"])

    @pytest.mark.parametrize("params", _fixtures)
    def test_distance_from_final_coord_is_major_distance(self, params):
        coords = variable_offset_curve(**params)

        dist = np.linalg.norm(coords[:, -1] - params["origin"])
        assert dist == pytest.approx(params["major_distance"])

    @pytest.mark.parametrize("params", _fixtures)
    def test_distance_to_origin_increases_linearly_between_angles(self, params):
        coords = variable_offset_curve(**params)

        distances = np.linalg.norm(coords - params["origin"].reshape((2, 1)), axis=0)
        gradient = np.gradient(distances)
        # Gradient is always positive
        assert np.all(gradient > 0)
        # Gradient is constant
        assert np.all(np.isclose(np.gradient(gradient), 0))

    @pytest.mark.parametrize("params", _fixtures)
    def test_curve_has_num_points(self, params):
        coords = variable_offset_curve(**params)

        assert coords.shape[1] == params["num_points"]


class TestVariedOffsetFunction:
    @classmethod
    def setup_class(cls):
        cls.picture_frame = PictureFrame(
            {
                "ro": {"value": 6},
                "ri": {"value": 3},
            }
        ).create_shape()

    def setup_method(self):
        self.params = {
            "minor_offset": 1,
            "major_offset": 4,
            "offset_angle_deg": 45,  # degrees
        }

    def test_offset_wire_is_closed(self):
        offset_wire = varied_offset_function(self.picture_frame, **self.params)

        assert offset_wire.is_closed

    @pytest.mark.xfail(
        reason="The plot looks correct, but I think the interpolation to find the point "
        "at the angle is not quite right. Probably to do with the fact the curve is not "
        "smooth enough at the ob_radius"
    )
    def test_the_offset_at_the_ob_radius_is_major_offset(self):
        offset_wire = varied_offset_function(self.picture_frame, **self.params)

        offset_interp = self._interpolation_func_closed_wire(offset_wire)
        shape_interp = self._interpolation_func_closed_wire(self.picture_frame)
        offset_size = np.linalg.norm(offset_interp(np.pi) - shape_interp(np.pi))
        assert offset_size == pytest.approx(self.params["major_offset"])

    def test_offset_from_shape_never_lt_minor_offset(self):
        offset_wire = varied_offset_function(self.picture_frame, **self.params)

        offset_interp = self._interpolation_func_closed_wire(offset_wire)
        shape_interp = self._interpolation_func_closed_wire(self.picture_frame)
        ang_space = np.linspace(0, 2 * np.pi, 50)
        offset_size = np.linalg.norm(
            offset_interp(ang_space) - shape_interp(ang_space), axis=0
        )
        assert np.all(offset_size >= self.params["minor_offset"])

    def test_offset_from_shape_never_gt_major_offset(self):
        offset_wire = varied_offset_function(self.picture_frame, **self.params)

        offset_interp = self._interpolation_func_closed_wire(offset_wire)
        shape_interp = self._interpolation_func_closed_wire(self.picture_frame)
        ang_space = np.linspace(0, 2 * np.pi, 50)
        offset_size = np.linalg.norm(
            offset_interp(ang_space) - shape_interp(ang_space), axis=0
        )
        assert np.all(offset_size <= self.params["major_offset"])

    def test_shape_is_closed(self):
        offset_wire = varied_offset_function(self.picture_frame, **self.params)

        assert offset_wire.is_closed

    @staticmethod
    def _interpolation_func_closed_wire(wire: BluemiraWire) -> Callable:
        coords = wire.discretize(50).xz
        centroid = wire.center_of_mass[[0, 2]]
        angles = np.radians(
            find_clockwise_angle_2d(np.array([-1, 0]), coords - centroid.reshape((2, 1)))
        )
        sort_idx = np.argsort(angles)
        angle_space = angles[sort_idx]
        sorted_coords = coords[:, sort_idx]
        return interp1d(angle_space, sorted_coords, fill_value="extrapolate")

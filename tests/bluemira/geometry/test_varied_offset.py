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
from bluemira.geometry.varied_offset import varied_offset_function
from bluemira.geometry.wire import BluemiraWire


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
            "offset_angle": 45,  # degrees
        }

    def test_offset_wire_is_closed(self):
        offset_wire = varied_offset_function(self.picture_frame, **self.params)

        assert offset_wire.is_closed

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

    def test_offset_never_decreases_between_offset_angle_and_ob_axis(self):
        offset_wire = varied_offset_function(self.picture_frame, **self.params)

        offset_interp = self._interpolation_func_closed_wire(offset_wire)
        shape_interp = self._interpolation_func_closed_wire(self.picture_frame)
        offset_angle_rad = np.radians(self.params["offset_angle"])
        ang_space = np.linspace(offset_angle_rad, np.pi, 50)
        offset_size = np.linalg.norm(
            offset_interp(ang_space) - shape_interp(ang_space), axis=0
        )
        offset_size_gradient = np.gradient(offset_size)
        assert np.all(offset_size_gradient >= 0)

    def test_offset_eq_to_minor_offset_within_offset_angle(self):
        offset_wire = varied_offset_function(self.picture_frame, **self.params)

        offset_interp = self._interpolation_func_closed_wire(offset_wire)
        shape_interp = self._interpolation_func_closed_wire(self.picture_frame)
        offset_angle_rad = np.radians(self.params["offset_angle"])
        ang_space = np.concatenate(
            (
                np.linspace(0, offset_angle_rad, 25),
                np.linspace(2 * np.pi - offset_angle_rad, 2 * np.pi, 25),
            )
        )
        offset_size = np.linalg.norm(
            offset_interp(ang_space) - shape_interp(ang_space), axis=0
        )
        np.testing.assert_allclose(offset_size, self.params["minor_offset"])

    @staticmethod
    def _interpolation_func_closed_wire(wire: BluemiraWire) -> Callable:
        coords = wire.discretize(50).xz
        centroid = wire.center_of_mass[[0, 2]]
        angles = np.radians(
            find_clockwise_angle_2d(np.array([-1, 0]), coords - centroid.reshape((2, 1)))
        )
        if angles[0] == angles[-1]:
            angles = angles[:-1]
        sort_idx = np.argsort(angles)
        angle_space = angles[sort_idx]
        sorted_coords = coords[:, sort_idx]
        return interp1d(angle_space, sorted_coords, fill_value="extrapolate")

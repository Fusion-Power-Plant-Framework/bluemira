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
from bluemira.geometry.varied_offset import varied_offset
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
        offset_wire = varied_offset(self.picture_frame, **self.params)

        assert offset_wire.is_closed

    def test_the_offset_at_the_ob_radius_is_major_offset(self):
        offset_wire = varied_offset(self.picture_frame, **self.params)

        offset_size = self._get_offset_sizes(offset_wire, self.picture_frame, np.pi)
        # Trade-off here between finer discretization when we
        # interpolate and a tighter tolerance. A bit of lee-way here for
        # the sake of fewer points to interpolate
        assert offset_size == pytest.approx(self.params["major_offset"], rel=1e-2)

    def test_offset_from_shape_never_lt_minor_offset(self):
        offset_wire = varied_offset(self.picture_frame, **self.params)

        ang_space = np.linspace(0, 2 * np.pi, 50)
        offset_size = self._get_offset_sizes(offset_wire, self.picture_frame, ang_space)
        assert np.all(offset_size >= self.params["minor_offset"])

    def test_offset_from_shape_never_gt_major_offset(self):
        offset_wire = varied_offset(self.picture_frame, **self.params)

        ang_space = np.linspace(np.radians(self.params["offset_angle"]), np.pi, 50)
        offset_size = self._get_offset_sizes(offset_wire, self.picture_frame, ang_space)
        assert np.all(offset_size <= self.params["major_offset"])

    def test_offset_never_decreases_between_offset_angle_and_ob_axis(self):
        offset_wire = varied_offset(self.picture_frame, **self.params)

        # Don't go all the way to π, as the interpolation used to
        # calculate the offset isn't perfect and we may go past π in
        # real terms
        ang_space = np.linspace(
            np.radians(self.params["offset_angle"]), np.pi * 0.99, 50
        )
        offset_size = self._get_offset_sizes(offset_wire, self.picture_frame, ang_space)
        offset_size_gradient = np.diff(offset_size)
        assert np.all(offset_size_gradient >= 0)

    @pytest.mark.xfail(
        "The test is comparing to the wrong offset, we want the 'normal' offset, not "
        " the 'radial' offset."
    )
    def test_offset_eq_to_minor_offset_within_offset_angle(self):
        offset_wire = varied_offset(self.picture_frame, **self.params)

        offset_angle_rad = np.radians(self.params["offset_angle"])
        ang_space = np.concatenate(
            (
                np.linspace(0, offset_angle_rad, 25),
                np.linspace(2 * np.pi - offset_angle_rad, 2 * np.pi, 25),
            )
        )
        offset_size = self._get_offset_sizes(offset_wire, self.picture_frame, ang_space)
        np.testing.assert_allclose(offset_size, self.params["minor_offset"])

    @staticmethod
    def _interpolation_func_closed_wire(wire: BluemiraWire) -> Callable:
        coords = wire.discretize(100).xz
        centroid = wire.center_of_mass[[0, 2]]
        angles = np.radians(
            find_clockwise_angle_2d(np.array([-1, 0]), coords - centroid.reshape((2, 1)))
        )
        if angles[0] == angles[-1]:
            # If the end angle == the start angle, we've looped back
            # around from 0 to 2π. We only want the range [0, 2π), so
            # remove the last. Not doing this causes the interpolation
            # function to return NaNs at 0.
            angles = angles[:-1]
        sort_idx = np.argsort(angles)
        angle_space = angles[sort_idx]
        sorted_coords = coords[:, sort_idx]
        return interp1d(angle_space, sorted_coords, fill_value="extrapolate")

    @staticmethod
    def _get_offset_sizes(
        shape_1: BluemiraWire, shape_2: BluemiraWire, angles: np.ndarray
    ) -> np.ndarray:
        interp_1 = TestVariedOffsetFunction._interpolation_func_closed_wire(shape_1)
        interp_2 = TestVariedOffsetFunction._interpolation_func_closed_wire(shape_2)
        return np.linalg.norm(interp_1(angles) - interp_2(angles), axis=0)

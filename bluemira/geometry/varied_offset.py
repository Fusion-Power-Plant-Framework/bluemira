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
import numpy as np
from scipy.interpolate import interp1d

from bluemira.geometry.tools import find_clockwise_angle_2d, make_polygon
from bluemira.geometry.wire import BluemiraWire


def varied_offset_function(
    shape: BluemiraWire,
    minor_offset: float,
    major_offset: float,
    offset_angle: float,
    num_points: int = 200,
) -> BluemiraWire:
    shape_coords = shape.discretize(num_points).xz
    offset_angle = np.radians(offset_angle)  # % 2*np.pi ?
    center_of_mass = shape.center_of_mass[[0, 2]].reshape((2, 1))
    ib_axis = np.array([-1, 0])

    angles = np.radians(find_clockwise_angle_2d(ib_axis, shape_coords - center_of_mass))
    # Sorting angles makes smoothing via interpolation easier later on
    sorted_angles, sorted_coords = _sort_coords_by_angle(angles, shape_coords)

    offsets = _calculate_offset_magnitudes(
        sorted_angles, offset_angle, minor_offset, major_offset, num_points
    )
    normals = _calculate_normals_2d(sorted_coords)
    new_shape_coords = sorted_coords + normals * offsets
    return _2d_coords_to_wire(new_shape_coords)


def _sort_coords_by_angle(angles, coords):
    angle_sort_idx = np.argsort(angles)
    return angles[angle_sort_idx], coords[:, angle_sort_idx]


def _find_constant_offset_indices(angles, offset_angle):
    return np.logical_or(angles < offset_angle, angles > (2 * np.pi - offset_angle))


def _calculate_offset_magnitudes(
    angles, offset_angle, minor_offset, major_offset, num_points
):
    constant_offset_idxs = _find_constant_offset_indices(angles, offset_angle)
    variable_offset_idxs = np.logical_not(constant_offset_idxs)

    offsets = np.empty_like(angles)
    offsets[constant_offset_idxs] = minor_offset
    offsets[variable_offset_idxs] = _calculate_variable_offset_magnitudes(
        angles[variable_offset_idxs], offset_angle, minor_offset, major_offset
    )

    # Smooth out the shape using an interpolation over 0 to 2π
    angular_space = np.linspace(0, 2 * np.pi, num_points)
    return _interpolate_over_angles(angles, offsets, angular_space)


def _calculate_variable_offset_magnitudes(
    angles, offset_angle, minor_offset, major_offset
):
    angles[angles > np.pi] = 2 * np.pi - angles[angles > np.pi]
    return (angles - offset_angle) / (np.pi - offset_angle) * (
        major_offset - minor_offset
    ) + minor_offset


def _interpolate_over_angles(angles, values, angular_space):
    interp_func = interp1d(angles, values, fill_value="extrapolate", kind="linear")
    return interp_func(angular_space)


def _calculate_normals_2d(shape_coords: np.ndarray) -> np.ndarray:
    gradients = np.gradient(shape_coords, axis=1)
    normals = np.empty_like(gradients)
    normals[0] = -gradients[1]
    normals[1] = gradients[0]
    return normals / np.linalg.norm(normals, axis=0)


def _2d_coords_to_wire(coords_2d):
    coords_3d = np.zeros((3, coords_2d.shape[1]))
    coords_3d[(0, 2), :] = coords_2d
    return make_polygon(coords_3d, closed=True)

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
    angle_sort_idx = np.argsort(angles)
    sorted_angles = angles[angle_sort_idx]
    sorted_coords = shape_coords[:, angle_sort_idx]

    # Find outward-pointing normal vectors to points
    normals = -_calculate_normals_2d(sorted_coords)

    constant_offset_idx = np.logical_or(
        sorted_angles < offset_angle, sorted_angles > (2 * np.pi - offset_angle)
    )
    variable_offset_idx = np.logical_not(constant_offset_idx)

    # Offset values should be a function of the angle
    var_offset_angles = sorted_angles[variable_offset_idx]
    var_offset_angles[var_offset_angles > np.pi] = (
        2 * np.pi - var_offset_angles[var_offset_angles > np.pi]
    )
    var_offset_offsets = (var_offset_angles - offset_angle) / (np.pi - offset_angle) * (
        major_offset - minor_offset
    ) + minor_offset

    offsets = np.empty_like(angles)
    offsets[constant_offset_idx] = minor_offset
    offsets[variable_offset_idx] = var_offset_offsets

    interp_func = interp1d(
        sorted_angles, offsets, fill_value="extrapolate", kind="slinear"
    )
    var_ang_space = np.linspace(0, 2 * np.pi, num_points)
    offsets = interp_func(var_ang_space)

    # Add the offsets to the original shape
    new_shape_coords = sorted_coords + normals * offsets
    new_shape_coords_3d = np.zeros((3, new_shape_coords.shape[1]))
    new_shape_coords_3d[(0, 2), :] = new_shape_coords
    new_shape = make_polygon(new_shape_coords_3d, closed=True)
    return new_shape


def _calculate_normals_2d(shape_coords: np.ndarray) -> np.ndarray:
    gradients = np.gradient(shape_coords, axis=1)
    normals = np.empty_like(gradients)
    normals[0] = gradients[1]
    normals[1] = -gradients[0]
    return normals / np.linalg.norm(normals, axis=0)

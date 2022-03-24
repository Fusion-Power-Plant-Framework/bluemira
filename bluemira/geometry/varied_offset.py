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
"""
Module containing functions to generate variable offset curves
"""

import numpy as np

from bluemira.geometry.tools import find_clockwise_angle_2d, make_bspline
from bluemira.geometry.wire import BluemiraWire


def varied_offset(
    wire: BluemiraWire,
    min_offset: float,
    max_offset: float,
    min_offset_angle: float,
    max_offset_angle: float,
    num_points: int = 200,
) -> BluemiraWire:
    """
    Create a new wire that offsets the given wire using a variable
    offset in the xz plane.

    All angles are measured from the negative x-direction (9 o'clock),
    centred at the center of mass of the wire.
    The offset will be 'min_offset' between the negative x-direction
    and 'min_offset_angle'. Between 'max_offset_angle' and
    the positive x-direction the offset will be 'max_offset'. Between
    those angles, the offset will linearly transition between the min
    and max.

    Parameters
    ----------
    wire: BluemiraWire
        The wire to create the offset from. This should be convex in
        order to get a sensible, non-intersecting, offset.
    min_offset: float
        The size of the minimum offset.
    max_offset: float
        The size of the maximum offset.
    min_offset_angle: float
        The angle at which the variable offset should begin, in degrees.
    max_offset_angle: float
        The angle at which the variable offset should end, in degrees.
    num_points: int
        The number of points to use in the discretization of the input
        wire.

    Returns
    -------
    offset_wire: BluemiraWire
        New wire at a variable offset to the input.
    """
    wire_coords = wire.discretize(num_points).xz
    min_offset_angle = np.radians(min_offset_angle)
    max_offset_angle = np.radians(max_offset_angle)
    center_of_mass = wire.center_of_mass[[0, 2]].reshape((2, 1))

    ib_axis = np.array([-1, 0])
    angles = np.radians(find_clockwise_angle_2d(ib_axis, wire_coords - center_of_mass))
    # Sort angles so coordinates are always clockwise and normals point outward
    angles, wire_coords = _sort_coords_by_angle(angles, wire_coords)

    offsets = _calculate_offset_magnitudes(
        angles, min_offset_angle, max_offset_angle, min_offset, max_offset
    )
    normals = _calculate_normals_2d(wire_coords)
    new_shape_coords = wire_coords + normals * offsets
    return _2d_coords_to_wire(new_shape_coords)


def _sort_coords_by_angle(angles: np.ndarray, coords: np.ndarray):
    """Sort the given angles and use that to re-order the coords."""
    angle_sort_idx = np.argsort(angles)
    return angles[angle_sort_idx], coords[:, angle_sort_idx]


def _calculate_offset_magnitudes(
    angles,
    min_offset_angle,
    max_offset_angle,
    min_offset,
    max_offset,
):
    """Calculate the magnitude of the offset at each angle."""
    offsets = np.empty_like(angles)
    # All angles less than min_offset_angle set to min offset
    constant_minor_offset_idxs = np.logical_or(
        angles < min_offset_angle, angles > (2 * np.pi - min_offset_angle)
    )
    offsets[constant_minor_offset_idxs] = min_offset

    # All angles greater than max_offset_angle set to max offset
    constant_major_offset_idxs = np.logical_and(
        angles > max_offset_angle, angles < 2 * np.pi - max_offset_angle
    )
    offsets[constant_major_offset_idxs] = max_offset

    variable_offset_idxs = np.logical_not(
        np.logical_or(constant_minor_offset_idxs, constant_major_offset_idxs)
    )
    offsets[variable_offset_idxs] = _calculate_variable_offset_magnitudes(
        angles[variable_offset_idxs],
        min_offset_angle,
        max_offset_angle,
        min_offset,
        max_offset,
    )
    return offsets


def _calculate_variable_offset_magnitudes(
    angles, start_angle, end_angle, min_offset, max_offset
):
    """
    Calculate the variable offset magnitude for each of the given angles.

    The offset increases linearly between start_angle and end_angle,
    between min_offset and max_offset.
    """
    angles[angles > np.pi] = 2 * np.pi - angles[angles > np.pi]
    angle_fraction = (angles - start_angle) / (end_angle - start_angle)
    return min_offset + angle_fraction * (max_offset - min_offset)


def _calculate_normals_2d(wire_coords: np.ndarray) -> np.ndarray:
    """
    Calculate the unit normals to the tangents at each of the given
    coordinates.

    Note that this applies an anti-clockwise rotation to the tangents,
    so to get an outward facing normal to a polygon, the coordinates
    should be ordered in the clockwise direction.
    """
    gradients = np.gradient(wire_coords, axis=1)
    normals = np.array([-gradients[1], gradients[0]])
    return normals / np.linalg.norm(normals, axis=0)


def _2d_coords_to_wire(coords_2d):
    """
    Build a wire from a 2D array of coordinates using a bspline.
    """
    coords_3d = np.zeros((3, coords_2d.shape[1]))
    coords_3d[(0, 2), :] = coords_2d
    return make_bspline(coords_3d, closed=True)

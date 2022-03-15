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
from scipy.spatial import ConvexHull

from bluemira.geometry.tools import find_clockwise_angle_2d, make_polygon, offset_wire
from bluemira.geometry.wire import BluemiraWire


def varied_offset_function(
    shape: BluemiraWire,
    minor_offset: float,
    major_offset: float,
    offset_angle_deg: float,
) -> BluemiraWire:
    """
    Make a wire that is a variable offset from the given shape.

    The variable offset will begin at the given offset angle, and reach
    its maximum on x-axis. The radius grows linearly from
    ``minor_offset`` to ``major_offset``, about the input shape's
    center-of-mass.

    The offset from ``shape`` will never be less than ``minor_offset``.
    This combined with the above means that, depending on the shape and
    for smaller angles, the offset from ``shape`` may not begin
    increasing until after the input angle, even though the radius of
    the new shape is increasing about the center-of-mass.

    Parameters
    ----------
    shape: BluemiraWire
        The wire to create the offset from.
    minor_offset: float
        The minimum offset from the shape.
    major_offset: float
        The maximum offset from the shape.
    offset_angle_deg: float
        The angle from the negative x-axis where the offset must be
        ``minor_offset``.

    Returns
    -------
    offset_shape: BluemiraWire
        The wire with a variable offset from the given shape.
    """
    ib_axis = np.array([-1, 0])
    centroid = shape.center_of_mass[[0, 2]]
    num_points = 100
    offset_angle = np.radians(offset_angle_deg)

    shape_coords = shape.discretize(num_points).xz
    # Constant offset to ensure new shape is never less than minor_offset
    constant_offset_coords = offset_wire(shape, minor_offset).discretize(num_points).xz

    offset_begin_coord = _point_at_angle(
        shape_coords=shape_coords,
        angular_space=np.radians(
            find_clockwise_angle_2d(
                ib_axis, constant_offset_coords - centroid.reshape((2, 1))
            )
        ),
        angle=offset_angle,
    )
    offset_start_dist = np.linalg.norm(offset_begin_coord - centroid)

    ob_radius_coord = _point_at_angle(
        shape_coords=shape_coords,
        angular_space=np.radians(
            find_clockwise_angle_2d(ib_axis, shape_coords - centroid.reshape((2, 1)))
        ),
        angle=np.pi,
    )
    ob_radius = ob_radius_coord[0] - centroid[0]
    variable_offset_coords = _make_variable_offset_loop(
        minor_offset=offset_start_dist,
        major_offset=major_offset + ob_radius,
        minor_angle=offset_angle,
        major_angle=np.pi,
        origin=centroid,
        num_points=2 * num_points,
    )

    hull_coords = np.hstack((constant_offset_coords, variable_offset_coords))
    hull = ConvexHull(hull_coords.T)
    coords_3d = np.zeros((3, len(hull.vertices)))
    coords_3d[[0, 2]] = hull_coords[:, hull.vertices]

    return make_polygon(coords_3d, closed=True)


def variable_offset_curve(
    minor_distance: float,
    major_distance: float,
    minor_angle: float,
    major_angle: float,
    origin: np.ndarray,
    num_points: int,
):
    """
    Draw a curve (clockwise) that is ``minor_distance`` distance from
    ``origin`` at angle ``minor_angle`` and ``major_distance`` distance
    at ``major_angle``. The distance from ``origin`` increases linearly
    between ``minor_angle`` and ``major_angle``.

    Parameters
    ----------
    minor_distance: float
        The smallest distance between the origin and the new curve.
    major_distance: float
        The greatest distance between the origin and the new curve.
    minor_angle: float
        The angle, in the range [0, 2π) from the negative x direction at
        which to create the minor distance point.
    major_angle: float
        The angle, in the range [0, 2π) from the negative x direction at
        which to create the major distance point.
    origin: np.ndarray[float, (2, 1)]
        The point to draw the curve around.
    num_points: int
        The number of coordinates to generate.
    """
    offset_from_origin = np.linspace(minor_distance, major_distance, num_points)
    angles = np.linspace(minor_angle, major_angle, num_points)
    offset_vec = offset_from_origin.T * np.array([-np.cos(angles), np.sin(angles)])
    return np.array(origin).reshape((2, 1)) + offset_vec


def _make_variable_offset_loop(
    minor_offset, major_offset, minor_angle, major_angle, origin, num_points
):
    """
    Make a variable offset loop by joining two variable offset curves.
    """
    variable_offset_coords = np.empty((2, 2 * num_points))
    variable_offset_coords[:, :num_points] = variable_offset_curve(
        minor_offset,
        major_offset,
        minor_angle,
        major_angle,
        origin,
        num_points,
    )
    variable_offset_coords[:, num_points:] = variable_offset_curve(
        major_offset,
        minor_offset,
        major_angle,
        2 * np.pi - minor_angle,
        origin,
        num_points,
    )
    return variable_offset_coords


def _point_at_angle(shape_coords, angular_space, angle):
    """
    Find the point at a given angle on a shape using interpolation.
    """
    interp_func = interp1d(angular_space, shape_coords, fill_value="extrapolate")
    return interp_func(angle)

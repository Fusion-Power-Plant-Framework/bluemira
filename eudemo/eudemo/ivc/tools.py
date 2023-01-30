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
"""
IVC tools
"""
import numpy as np

from bluemira.geometry.tools import boolean_cut, make_polygon
from bluemira.geometry.wire import BluemiraWire


def cut_wall_below_x_point(shape: BluemiraWire, x_point_z: float) -> BluemiraWire:
    """
    Remove the parts of the wire below the given value in the z-axis.
    """
    # Create a box that surrounds the wall below the given z
    # coordinate, then perform a boolean cut to remove that portion
    # of the wall's shape.
    bounding_box = shape.bounding_box
    cut_box_points = np.array(
        [
            [bounding_box.x_min, 0, bounding_box.z_min],
            [bounding_box.x_min, 0, x_point_z],
            [bounding_box.x_max, 0, x_point_z],
            [bounding_box.x_max, 0, bounding_box.z_min],
            [bounding_box.x_min, 0, bounding_box.z_min],
        ]
    )
    cut_zone = make_polygon(cut_box_points, label="_shape_cut_exclusion", closed=True)
    # For a single-null, we expect three 'pieces' from the cut: the
    # upper wall shape and the two separatrix legs
    pieces = boolean_cut(shape, [cut_zone])

    wall_piece = pieces[np.argmax([p.center_of_mass[2] for p in pieces])]
    if wall_piece.center_of_mass[2] < x_point_z:
        raise ValueError(
            "Could not cut wall shape below x-point. "
            "No parts of the wall found above x-point."
        )
    return wall_piece

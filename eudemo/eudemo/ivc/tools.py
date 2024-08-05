# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
IVC tools
"""

import numpy as np

from bluemira.geometry.tools import boolean_cut, make_polygon
from bluemira.geometry.wire import BluemiraWire


def cut_wall_below_x_point(shape: BluemiraWire, x_point_z: float) -> BluemiraWire:
    """
    Remove the parts of the wire below the given value in the z-axis.

    Raises
    ------
    ValueError
        No parts of shape found about the x points
    """
    # Create a box that surrounds the wall below the given z
    # coordinate, then perform a boolean cut to remove that portion
    # of the wall's shape.
    bounding_box = shape.bounding_box
    cut_box_points = np.array([
        [bounding_box.x_min, 0, bounding_box.z_min],
        [bounding_box.x_min, 0, x_point_z],
        [bounding_box.x_max, 0, x_point_z],
        [bounding_box.x_max, 0, bounding_box.z_min],
        [bounding_box.x_min, 0, bounding_box.z_min],
    ])
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

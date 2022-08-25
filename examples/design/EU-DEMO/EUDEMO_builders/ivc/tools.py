# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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

"""
from typing import Tuple

import numpy as np

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, make_polygon
from bluemira.geometry.wire import BluemiraWire


def _cut_vessel_shape(
    in_vessel_face: BluemiraFace, rm_clearance_face: BluemiraFace
) -> Tuple[BluemiraFace, BluemiraFace]:
    """
    Cut a remote maintainance clearance into the given vessel shape.
    """
    pieces = boolean_cut(in_vessel_face, [rm_clearance_face])
    blanket_face = pieces[np.argmax([p.center_of_mass[2] for p in pieces])]
    divertor_face = pieces[np.argmin([p.center_of_mass[2] for p in pieces])]
    return blanket_face, divertor_face


def _make_clearance_face(
    x_min: float, x_max: float, z: float, thickness: float
) -> BluemiraFace:
    """
    Makes a rectangular face in xz with the given thickness in z.

    The face is intended to be used to cut a remote maintainance
    clearance between blankets and divertor.
    """
    x_coords = [x_min, x_min, x_max, x_max]
    y_coords = [0, 0, 0, 0]
    z_coords = [
        z + thickness / 2,
        z - thickness / 2,
        z - thickness / 2,
        z + thickness / 2,
    ]
    return BluemiraFace(make_polygon([x_coords, y_coords, z_coords], closed=True))


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
    cut_zone = make_polygon(cut_box_points, label="_shape_cut_exclusion")
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

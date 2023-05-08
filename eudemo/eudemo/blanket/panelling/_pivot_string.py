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
from itertools import count
from typing import Tuple

import numpy as np


def make_pivoted_string(
    boundary_points: np.ndarray,
    max_angle: float = 10,
    dx_min: float = 0,
    dx_max: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a set of pivot points along the given boundary.

    Given a set of boundary points, some maximum angle, and minimum and
    maximum segment length, this function derives a set of pivot points
    along the boundary, that define a 'string'. You might picture a
    'string' as a thread wrapped around some nails (pivot points) on a
    board.

    Parameters
    ----------
    points:
        The coordinates (in 3D) of the pivot points. Must have shape
        (N, 3) where N is the number of boundary points.
    max_angle:
        The maximum angle between neighbouring pivot points.
    dx_min:
        The minimum distance between pivot points.
    dx_max:
        The maximum distance between pivot points.

    Returns
    -------
    new_points:
        The pivot points' coordinates. Has shape (M, 3), where M is the
        number of pivot points.
    index:
        The indices of the pivot points into the input points.
    """
    if dx_min > dx_max:
        raise ValueError(
            f"'dx_min' cannot be greater than 'dx_max': '{dx_min} > {dx_max}'"
        )
    tangent_vec = boundary_points[1:] - boundary_points[:-1]
    tangent_vec_norm = np.linalg.norm(tangent_vec, axis=1)
    # Protect against dividing by zero
    tangent_vec_norm[tangent_vec_norm == 0] = 1e-32
    average_step_length = np.median(tangent_vec_norm)
    tangent_vec /= tangent_vec_norm.reshape(-1, 1) * np.ones(
        (1, np.shape(tangent_vec)[1])
    )

    new_points = np.zeros_like(boundary_points)
    index = np.zeros(boundary_points.shape[0], dtype=int)

    new_points[0] = boundary_points[0]
    to, po = tangent_vec[0], boundary_points[0]

    k = count(1)
    for i, (p, t) in enumerate(zip(boundary_points[1:], tangent_vec)):
        c = np.cross(to, t)
        c_mag = np.linalg.norm(c)
        dx = np.linalg.norm(p - po)  # segment length
        if (
            c_mag > np.sin(np.deg2rad(max_angle)) and dx > dx_min
        ) or dx + average_step_length > dx_max:
            j = next(k)
            new_points[j] = boundary_points[i]  # pivot point
            index[j] = i + 1  # pivot index
            to, po = t, p  # update
    if dx > dx_min:
        j = next(k)
    new_points[j] = p  # replace/append last point
    index[j] = i + 1  # replace/append last point index
    new_points = new_points[: j + 1]  # trim
    index = index[: j + 1]  # trim
    return new_points, index

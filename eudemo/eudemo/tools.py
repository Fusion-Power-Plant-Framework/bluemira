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
A collection of tools used in the EU-DEMO design.
"""

from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import slice_shape


# TODO(hsaunders1904): move this into blanket module
def get_inner_cut_point(breeding_blanket_xz, r_inner_cut):
    """
    Get the inner cut point of the breeding blanket geometry.
    """
    cut_plane = BluemiraPlane.from_3_points(
        [r_inner_cut, 0, 0], [r_inner_cut, 0, 1], [r_inner_cut, 1, 1]
    )
    # Get the first intersection with the vertical inner cut plane
    intersections = slice_shape(breeding_blanket_xz.boundary[0], cut_plane)
    intersections = intersections[intersections[:, -1] > 0.0]
    intersection = sorted(intersections, key=lambda x: x[-1])[0]
    return intersection

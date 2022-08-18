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
Sectioned component helpers
"""
from typing import List, Tuple

from bluemira.base.components import PhysicalComponent
from bluemira.builders.tools import (
    circular_pattern_component,
    find_xy_plane_radii,
    get_n_sectors,
    make_circular_xy_ring,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import revolve_shape


def build_sectioned_xy(
    face: BluemiraFace, plot_colour: Tuple[float]
) -> List[PhysicalComponent]:
    """
    Build the x-y components of sectioned component

    Parameters
    ----------
    face: BluemiraFace
        xz face to build xy component
    """
    xy_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 1, 0])

    r_ib_out, r_ob_out = find_xy_plane_radii(face.boundary[0], xy_plane)
    r_ib_in, r_ob_in = find_xy_plane_radii(face.boundary[1], xy_plane)

    sections = []
    for name, r_in, r_out in [
        ["inboard", r_ib_in, r_ib_out],
        ["outboard", r_ob_in, r_ob_out],
    ]:
        board = make_circular_xy_ring(r_in, r_out)
        section = PhysicalComponent(name, board)
        section.plot_options.face_options["color"] = plot_colour
        sections.append(section)

    return sections


def build_sectioned_xyz(
    name: str,
    n_TF: int,
    plot_colour: Tuple[float],
    face: BluemiraFace,
    degree: float = 360,
    working: bool = False,
) -> List[PhysicalComponent]:
    """
    Build the x-y-z components of sectioned component

    Parameters
    ----------
    face: BluemiraFace
        xz face to build xyz component
    """
    sector_degree, n_sectors = get_n_sectors(n_TF, degree)

    shape = revolve_shape(
        face,
        base=(0, 0, 0),
        direction=(0, 0, 1),
        degree=sector_degree if working else max(359, degree),
    )
    body = PhysicalComponent(name, shape)
    body.display_cad_options.color = plot_colour

    # this is currently broken in some situations
    # because of #1319 and related Topological naming issues
    return (
        [body]
        if working
        else circular_pattern_component(
            body, n_sectors, degree=sector_degree * n_sectors
        )
    )

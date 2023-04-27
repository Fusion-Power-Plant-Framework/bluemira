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
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, make_polygon
from bluemira.geometry.wire import BluemiraWire
from eudemo.blanket.designer import BlanketSegments


def make_simple_blanket() -> BlanketSegments:
    """
    Make a semi-circular (ish) blanket.

    The inboard and outboard are symmetrical quarter-circles with radius
    3. Where the centre of the inboard quarter-circle is [5, -1.5] and the
    centre of the outboard is [6, -1.5] (in the xz-plane).
    """
    # Inboard
    ib_arc_inner = make_circle(
        radius=3, center=[5, 0, -1.5], start_angle=180, end_angle=270, axis=(0, 1, 0)
    )
    ib_arc_outer = make_circle(
        radius=4, center=[5, 0, -1.5], start_angle=180, end_angle=270, axis=(0, 1, 0)
    )
    lower_join = make_polygon({"x": [1, 2], "z": [-1.5, -1.5]})
    upper_join = make_polygon({"x": [5, 5], "z": [1.5, 2.5]})
    inboard = BluemiraFace(
        BluemiraWire([ib_arc_inner, lower_join, ib_arc_outer, upper_join])
    )
    # Outboard
    ob_arc_inner = make_circle(
        radius=3, center=[6, 0, -1.5], start_angle=270, end_angle=360, axis=(0, 1, 0)
    )
    ob_arc_outer = make_circle(
        radius=4, center=[6, 0, -1.5], start_angle=270, end_angle=360, axis=(0, 1, 0)
    )
    lower_join = make_polygon({"x": [9, 10], "z": [-1.5, -1.5]})
    upper_join = make_polygon({"x": [6, 6], "z": [1.5, 2.5]})
    outboard = BluemiraFace(
        BluemiraWire([ob_arc_inner, upper_join, ob_arc_outer, lower_join])
    )
    return BlanketSegments(
        inboard=inboard,
        outboard=outboard,
        inboard_boundary=ib_arc_inner,
        outboard_boundary=ob_arc_inner,
    )

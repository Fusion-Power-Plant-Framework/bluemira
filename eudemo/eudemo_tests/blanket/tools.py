# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
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

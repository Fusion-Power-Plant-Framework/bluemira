# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from pprint import pprint

import matplotlib.pyplot as plt

import bluemira.codes._freecadapi as cadapi
from bluemira.base.file import get_bluemira_path
from bluemira.display import plot_2d
from bluemira.geometry.tools import serialise_shape
from bluemira.geometry.wire import BluemiraWire
from bluemira.radiation_transport.neutronics.wires import (
    CircleInfo,
    StraightLineInfo,
    WireInfo,
    WireInfoList,
)

if TYPE_CHECKING:
    import Part
Part = cadapi.Part

FAILING_GEOMETRY = Path(
    get_bluemira_path("geometry/test_data", subfolder="tests"), "vv_outer.json"
)
with open(FAILING_GEOMETRY) as j:
    data_dict = json.load(j)
# This should fail
# wire = deserialise_shape(data_dict)
# This is becacause the following line fails with Part.OCCError
# Part.Wire(deserialised_list[::])

# Very fortunately the universe align with the following coincidences:
# 1. The failing data set consist only of straight lines and arcs.
# 2. I have created something in the radiation_transport.neutronics module that manages
#    exactly the same two types of wires.
# Therefore I can use neutronics.wires.CircleInfo and .StraightLineInfo to diagnose
# the problem.

v = data_dict["BluemiraWire"]["boundary"][0]["Wire"]
deserialised_list = [cadapi.deserialise_shape(edge) for edge in v]


def plot_wire_objects(wire_object: list[Part.Wire] | Part.Wire, color=None, **kwargs):
    reconstructed_wire = BluemiraWire(wire_object)
    if color:
        return plot_2d(
            reconstructed_wire, wire_options={"color": color, "linewidth": 2.5}, **kwargs
        )
    return plot_2d(reconstructed_wire, wire_options={"linewidth": 2.5}, **kwargs)


def as_bluemira_wire_info_list(wire_info_dict_list: list[dict]) -> WireInfoList:
    """Convert a freecad wire (already serialized as a list of dict) into
    :class:`~bluemira.radiation_transport.neutronics.wire.WireInfo`.
    """
    wire_info_list = []
    for wire_info_dict in wire_info_dict_list:
        for wire_type, wire_content in wire_info_dict.items():
            if wire_type == "LineSegment":
                wire_info_list.append(as_straight_line_info(wire_content))
            elif wire_type == "ArcOfCircle":
                wire_info_list.append(as_circle_info(wire_content))
            else:
                raise NotImplementedError(f"{wire_type} is not available yet!")
    return WireInfoList(wire_info_list)


def as_circle_info(circle_info_dict: dict) -> WireInfo:
    """Convert freecad arc of circle dict into CircleInfo."""
    circ = WireInfo(
        key_points=CircleInfo(
            start_point=circle_info_dict["StartPoint"],
            end_point=circle_info_dict["EndPoint"],
            center=circle_info_dict["Center"],
            radius=circle_info_dict["Radius"],
        ),
        tangents=[[None] * 3, [None] * 3],
    )
    circ.axis = circle_info_dict["Axis"]
    circ.start_angle = circle_info_dict["StartAngle"]
    circ.end_angle = circle_info_dict["EndAngle"]
    circ.flip_through_0 = circ.start_angle > circ.end_angle
    if circ.axis[0] == 0 and circ.axis[2] == 0:
        # circle exist on xz plane
        # should be decreasing
        circ.direction = "counter-clockwise" if circ.axis[1] == 1 else "clockwise"
    circ.more_than_180 = "unknown"
    return circ


def as_straight_line_info(straight_line_info_dict: dict) -> WireInfo:
    """Convert freecad straight line dict into WireInfo."""
    return WireInfo(
        key_points=StraightLineInfo(
            start_point=straight_line_info_dict["StartPoint"],
            end_point=straight_line_info_dict["EndPoint"],
        ),
        tangents=[[None] * 3, [None] * 3],
    )


whole_wire = as_bluemira_wire_info_list(v)
bmwire = whole_wire.restore_to_wire()
for edge in whole_wire:
    if isinstance(edge.key_points, CircleInfo):
        print(edge.key_points.start_point, edge.key_points.end_point)
        print(str(edge.start_angle).rjust(15), str(edge.end_angle).rjust(15))
        print(
            edge.direction.rjust(17),
            "CROSSES zero" if edge.flip_through_0 else "does NOT cross zero",
        )
        print()

plt.plot(*bmwire.discretise(5000)[::2])
plt.show()
pprint(serialise_shape(bmwire))

# ax = plot_wire_objects(deserialised_list[:1], color="C0", show=False)
# ax = plot_wire_objects(deserialised_list[1:2], color="C1", ax=ax, show=False)
# ax = plot_wire_objects(deserialised_list[2:3], color="C2", ax=ax, show=False)
# ax = plot_wire_objects(deserialised_list[3:4], color="C3", ax=ax, show=False)
# ax = plot_wire_objects(deserialised_list[4:5], color="C5", ax=ax, show=False)
# ax = plot_wire_objects(deserialised_list[4:6], color="C6", ax=ax, show=False)
# ax = plot_wire_objects(deserialised_list[6:7], color="C7", ax=ax, show=False)
# ax = plot_wire_objects(deserialised_list[7:9], color="C8", ax=ax, show=False)
# ax = plot_wire_objects(deserialised_list[9:10], color="C9", ax=ax, show=False)
# plt.show()

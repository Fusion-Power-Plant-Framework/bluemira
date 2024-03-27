# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
# ruff: noqa: D105
"""
Info about straight line wires and circles. Made to be simpler to modify than a whole
BluemiraWire.
"""

from __future__ import annotations

from collections import abc
from dataclasses import dataclass
from typing import Iterable, List, NamedTuple, Optional, Sequence, Union

import numpy as np
from numpy import typing as npt

from bluemira.geometry.error import GeometryError
from bluemira.geometry.tools import make_circle_arc_3P, make_polygon
from bluemira.geometry.wire import BluemiraWire


class StraightLineInfo(NamedTuple):
    """Key information about a straight line"""

    start_point: Iterable[float]  # 3D coordinates
    end_point: Iterable[float]  # 3D coordinates


class CircleInfo(NamedTuple):
    """Key information about a circle"""

    start_point: Iterable[float]  # 3D coordinates
    end_point: Iterable[float]  # 3D coordinates
    center: Iterable[float]  # 3D coordinates
    radius: float  # scalar


@dataclass
class WireInfo:
    """
    A tuple to store:
    1. the key points about this wire (and what kind of wire this is)
    2. The tangent to that wire at the start and end
    3. A copy of the wire itself
    """

    key_points: Union[StraightLineInfo, CircleInfo]  # 2 points of xyz/ CircleInfo
    tangents: Optional[Sequence[Iterable[float]]]  # 2 normalized directional vectors xyz
    wire: Optional[BluemiraWire] = None

    @classmethod
    def from_2P(  # noqa: N802
        cls, start_point: npt.NDArray[float], end_point: npt.NDArray[float]
    ) -> WireInfo:
        """
        Create the WireInfo for a straight line (i.e. one where the key_points is of
        instance StraightLineInfo) using only two points.
        """
        direction = np.array(end_point) - np.array(start_point)
        normed_dir = direction / np.linalg.norm(direction)
        return cls(StraightLineInfo(start_point, end_point), [normed_dir, normed_dir])


class WireInfoList(abc.Sequence):
    """A class to store info about a series of wires"""

    def __init__(self, info_list: List[WireInfo]):
        self.info_list = list(info_list)
        for prev_wire, curr_wire in zip(self[:-1], self[1:]):
            if not np.array_equal(prev_wire.key_points[1], curr_wire.key_points[0]):
                raise GeometryError("Next wire must start where the previous wire stops")

    def __len__(self) -> int:
        return self.info_list.__len__()

    def __getitem__(self, index_or_slice) -> Union[List[WireInfo], WireInfo]:
        return self.info_list.__getitem__(index_or_slice)

    def __add__(self, other_info_list) -> WireInfoList:
        return WireInfoList([*self.info_list.copy(), *other_info_list.info_list.copy()])

    def __repr__(self) -> str:
        return super().__repr__().replace(" at ", f" of {len(self)} WireInfo at ")

    def pop(self, index):
        """Pop one element"""
        return self.info_list.pop(index)

    @property
    def start_point(self):
        """The start_point for the entire series of wires"""
        return self.info_list[0].key_points[0]

    @start_point.setter
    def start_point(self, new_start_point: npt.NDArray[float]):
        """
        Set the start_point to somewhere new. Note this doesn't change the tangents.
        """
        key_points = self.info_list[0].key_points
        # have to break it open because it's an immutable NamedTuple.
        new_kp = key_points.__class__(new_start_point, *key_points[1:])
        self.info_list[0].key_points = new_kp

    @property
    def end_point(self):
        """The end_point for the entire series of wires"""
        return self.info_list[-1].key_points[1]

    @end_point.setter
    def end_point(self, new_end_point):
        """Set the end_point to somewhere new. Note this doesn't change the tangents."""
        key_points = self.info_list[-1].key_points
        new_kp = key_points.__class__(key_points[0], new_end_point, *key_points[2:])
        self.info_list[0].key_points = new_kp

    def restore_to_wire(self) -> BluemiraWire:
        """Re-create a bluemira wire from a series of WireInfo."""
        wire_list = []
        for info in self:
            start_end = np.array(info.key_points[:2])
            if info.wire:
                # quick way to get the wire back without doing any computation is by
                # directly copying.
                wire_list.append(info.wire)
                continue
            if isinstance(info.key_points, StraightLineInfo):
                wire_list.append(make_polygon(start_end.T, closed=False))
            else:
                # given two points on the circumference, only makes the SHORTER of the
                # two possible arcs of the circle.
                chord_intersect = start_end.mean(axis=0)
                direction = chord_intersect - info.key_points.center
                normed_dir = direction / np.linalg.norm(direction)
                middle = info.key_points.center + info.key_points.radius * normed_dir
                wire_list.append(make_circle_arc_3P(start_end[0], middle, start_end[1]))
        return BluemiraWire(wire_list)

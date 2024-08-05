# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Load objects
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.structural.constants import LoadKind, LoadType, SubLoadType
from bluemira.structural.error import StructuralError

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class Load:
    """Load container"""

    kind: str | LoadKind
    subtype: str | LoadType
    Q: float | None = None
    w: float | None = None
    x: float | None = None
    node_id: int | None = None
    element_id: int | None = None

    def __post_init__(self):
        """Enforce enums

        Raises
        ------
        ValueError
            Required arguments not provided
        """
        self.kind = LoadKind(self.kind)
        self.subtype = LoadType(self.subtype)

        if self.kind is LoadKind.DISTRIBUTED_LOAD and None in {self.element_id, self.w}:
            raise ValueError("A distributed_load requires element_id and w set")
        if self.kind is LoadKind.ELEMENT_LOAD and None in {
            self.element_id,
            self.Q,
            self.x,
        }:
            raise ValueError("An element_load requires element_id, Q and x set")
        if self.kind is LoadKind.NODE_LOAD and None in {self.node_id, self.Q}:
            raise ValueError("A node_load requires node_id and Q set")


def _check_load_type(
    load_type: str | LoadType, sub_type: str | SubLoadType = "all"
) -> LoadType:
    load_type = LoadType(load_type)
    inp_sub_type = SubLoadType(sub_type)

    if inp_sub_type is SubLoadType.FORCE and load_type > LoadType.Fz:
        raise StructuralError(
            f"Cannot set a force load with a moment load type: {load_type}"
        )

    if inp_sub_type is SubLoadType.MOMENT and load_type < LoadType.Mx:
        raise StructuralError(
            f"Cannot set a moment load with a force load type: {load_type}"
        )
    return load_type


def node_load(load: float, load_type: str | LoadType) -> np.ndarray:
    """
    Calculates the reaction force vector due to a point load Q, applied at a
    Node.

    Parameters
    ----------
    load:
        The value of the point load (which may be a force or a moment)
    load_type:
        The type of load to apply (from 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
            'Fi': force in the 'i' direction
            'Mi': moment about the 'i' direction

    Returns
    -------
    The fixed-ends reaction force vector (6)
    """
    reactions = np.zeros(6)
    load_type = _check_load_type(load_type)
    reactions[load_type.value] = load
    return reactions


def point_load(
    load: float, x: float, length: float, load_type: str | LoadType
) -> np.ndarray:
    """
    Calculates the reaction force vector due to a point load Q, applied at x
    along the element of length L, going from node_1 to node_2.

    Parameters
    ----------
    load:
        The value of the point load (which may be a force or a moment)
    x:
        The parameterised distance along the element from node_1 to node_2,
        from 0 to 1.
    length:
        The element length
    load_type:
        The type of load to apply (from['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
            'Fi': force in the 'i' direction
            'Mi': moment about the 'i' direction

    Returns
    -------
    The fixed-ends reaction force vector (12)
    """
    load_type = _check_load_type(load_type)

    reactions = np.zeros(12)
    x *= length
    a = length - x
    if load_type is LoadType.Fx:
        reactions[0] = -load * a / length
        reactions[6] = -load * x / length
    elif load_type is LoadType.Fy:
        reactions[1] = load * a**2 * (length + 2 * x) / length**3
        reactions[5] = load * x * a**2 / length**2
        reactions[7] = load * x**2 * (length + 2 * a) / length**3
        reactions[11] = -load * x**2 * a / length**2
    elif load_type is LoadType.Fz:
        reactions[2] = load * a**2 * (length + 2 * x) / length**3
        reactions[4] = -load * x * a**2 / length**2
        reactions[8] = load * x**2 * (length + 2 * a) / length**3
        reactions[10] = load * x**2 * a / length**2
    elif load_type is LoadType.Mx:
        reactions[3] = -load * a / length
        reactions[9] = -load * x / length
    elif load_type is LoadType.My:
        reactions[2] = 6 * load * x * a / length**3
        reactions[4] = load * a * (2 * x - a) / length**2
        reactions[8] = -load * x * a / length**3
        reactions[10] = load * x * (2 * a - x) / length**2
    elif load_type is LoadType.Mz:
        reactions[1] = -load * x * a / length**3
        reactions[5] = load * a * (2 * x - a) / length**2
        reactions[7] = load * x * a / length**3
        reactions[11] = load * x * (2 * a - x) / length**2

    return reactions


def distributed_load(w: float, length: float, load_type: str | LoadType) -> np.ndarray:
    """
    Calculates the reactor force vector due to a distributed load w, applied
    over the length of the element.

    Parameters
    ----------
    w:
        The value of the distributed load
    length:
        Length of the element along which the load is applied
    load_type:
        The type of load to apply (from ['Fx', 'Fy', 'Fz']):
            'Fi': force in the 'i' direction
    """
    load_type = _check_load_type(load_type, sub_type=SubLoadType.FORCE)

    reactions = np.zeros(12)
    if load_type is LoadType.Fx:
        reactions[0] = w * length / 2
        reactions[6] = -w * length / 2
    elif load_type is LoadType.Fy:
        reactions[1] = w * length / 2
        reactions[5] = w * length**2 / 12
        reactions[7] = w * length / 2
        reactions[11] = -w * length**2 / 12
    elif load_type is LoadType.Fz:
        reactions[2] = w * length / 2
        reactions[4] = -w * length**2 / 12
        reactions[8] = w * length / 2
        reactions[10] = w * length**2 / 12

    return reactions


class LoadCase:
    """
    A simple container object for a collection of loads
    """

    def __init__(self, *data: Load):
        self._data = [*data]

    def __iter__(self) -> Iterator[Load]:
        """Iterate over loads"""
        yield from self._data

    def add_node_load(self, node_id: int, load: float, load_type: str):
        """
        Adds a node load to the LoadCase

        Parameters
        ----------
        node_id:
            The id_number of the Node to apply the load at
        load:
            The value of the load
        load_type:
            The type and axis of the load from ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        """
        _check_load_type(load_type)
        self._data.append(
            Load(kind="Node Load", subtype=load_type, node_id=node_id, Q=load)
        )

    def add_element_load(self, element_id: int, load: float, x: float, load_type: str):
        """
        Adds an element point load to the LoadCase

        Parameters
        ----------
        element_id:
            The id_number of the Element to apply the load at
        load:
            The value of the load
        x:
            The parameterised and normalised distance along the element x-axis
        load_type:
            The type and axis of the load (from ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
        """
        _check_load_type(load_type)
        x_clip = np.clip(x, 0, 1)
        if x_clip != x:
            bluemira_warn("x should be between 0 and 1.")
        self._data.append(
            Load(
                kind="Element Load",
                subtype=load_type,
                element_id=element_id,
                Q=load,
                x=x,
            )
        )

    def add_distributed_load(self, element_id: int, w: float, load_type: str):
        """
        Adds a distributed load to the LoadCase

        Parameters
        ----------
        element_id:
            The id_number of the Element to apply the load at
        w:
            The value of the distributed load
        load_type:
            The type and axis of the load (from ['Fx', 'Fy', 'Fz'])
        """
        _check_load_type(load_type, sub_type=SubLoadType.FORCE)
        self._data.append(
            Load(kind="Distributed Load", subtype=load_type, element_id=element_id, w=w)
        )

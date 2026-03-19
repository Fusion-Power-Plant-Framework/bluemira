# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tools to convert geometry wires into sets of square coils
"""

import numpy as np
from matproplib.material import Material

from bluemira.equilibria.coils import Coil, CoilGroup
from bluemira.geometry.wire import BluemiraWire

__all__ = [
    "_length_step",
    "get_coil_points",
    "make_coils_from_wire",
]


def _length_step(p1, p2, delta) -> float:
    """
    Calculates the tangent angle for two points and uses this to determine
    the wire length to use as the difference for a square with thickness
    delta.

    Returns
    -------
    float:
        Length value for a wire discretisation step.
    """
    theta = np.arctan2(p2[0] - p1[0], p2[2] - p1[2])
    return 0.5 * delta * ((np.sqrt(2) + 1) - (np.sqrt(2) - 1) * np.cos(4 * theta))


def get_coil_points(wire: BluemiraWire, thickness: float) -> np.ndarray:
    """
    Discretises input wire in such a way that squares centred on
    those points will not overlap whilst minimising gaps.

    Achieves by calculating tangent angle at given point and using
    this to determine how far along the wire to put the next point.

    Paramters
    ---------
    wire:
        The wire that the coilset will be centred on.
    thickness:
        The thickness of the coils, will also impact the number of coils.

    Returns
    -------
    np.ndarray:
        An array containing the discretised points of the input wire in 3D.
    """
    ip = wire.start_point().T[0]
    n_max = wire.length / thickness
    p = ip
    dl = thickness
    current_length = 0
    points = [ip]
    for _ in range(int(n_max)):
        p2 = wire.value_at(distance=current_length + dl).T
        g_val = _length_step(p, p2, thickness)
        point = wire.value_at(distance=current_length + g_val).T
        p = point
        if current_length + g_val < wire.length - thickness:
            points = np.append(points, [p], axis=0)
            current_length += g_val
        else:
            continue
    return points


def make_coils_from_wire(
    wire: BluemiraWire,
    thickness: float,
    material: Material | None = None,
    simple: bool = True,  # noqa: FBT001, FBT002
) -> CoilGroup:
    """
    Function to create a coilset from a wire, where the coils making up the coilset
    will have a dx and dz equal to half the given thickness, with coils separated by
    the full thickness value. Additionally the coils will be make of the provided
    material.

    The created coils will follow the wire by treating it as a centreline with coils
    centred on the line.

    Parameters
    ----------
    wire:
        The wire that the coilset will be centred on.
    thickness:
        The thickness of the coils, will also impact the number of coils.
    material:
        The matproplib material.
    simple:
        Method of discretising the input wire.

    Returns
    -------
    CoilGroup:
        A group of coils following the input wire.
    """
    if simple:
        coil_points = wire.discretise(dl=np.sqrt(2) * thickness).T
    else:
        coil_points = get_coil_points(wire, thickness)
    coils = [
        Coil(
            x=point[0],
            z=point[2],
            dx=0.5 * thickness,
            dz=0.5 * thickness,
            n_turns=1,
            discretisation=np.nan,
        )
        for point in coil_points
    ]
    cg = CoilGroup(*coils)
    if material:
        j_max = material.j_max  # how to get?
        b_max = material.b_max  # how to get?
        cg.assign_material(ctype=None, j_max=j_max, b_max=b_max)
    return cg

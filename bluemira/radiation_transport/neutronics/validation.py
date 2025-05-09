# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Validate the geometry of the csg model created."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluemira.radiation_transport.neutronics.error import CSGGeometryValidationError

if TYPE_CHECKING:
    from bluemira.radiation_transport.neutronics.cell import (
        Cell,
        CellStack,
        CellWall,
        Vertices,
    )


def check_vertices_ordering(vertices: Vertices) -> None:
    """Check that the vertices are correctly ordered.

    Raises
    ------
    CSGGeometryValidationError
        If the vertices are incorrectly ordered, it suggests that we may have an
        incorrectly ordered wire.
    """
    in_vec = -vertices.ccw_in + vertices.cw_in
    ex_vec = -vertices.ccw_ex + vertices.cw_ex
    if (in_vec @ ex_vec) <= 0:
        raise CSGGeometryValidationError(
            "The interior wire should point in the same direction as the exterior wire!"
        )


def check_stacks_are_neighbours(stack_ccw: CellStack, stack_cw: CellStack) -> None:
    """Check that two stacks are neighbours, by ensuring that the ccw stack's cw side==
    cw stack's ccw side. (ccw = counter-clockwise, cw = clockwise.)

    Parameters
    ----------
    stack_ccw:
        :class:`~CellStack`
    stack_cw:
        :class:`~CellStack`

    Raises
    ------
    CSGGeometryValidationError
        raised if the two stacks don't have 2 matching vertices where we expect them.
    """
    if stack_ccw.vertices.cw_in != stack_cw.vertices.ccw_in:
        raise CSGGeometryValidationError(
            f"{stack_ccw} and {stack_cw} has mismatched interior (plasma-facing) vertex!"
        )
    if stack_ccw.vertices.cw_ex != stack_cw.vertices.ccw_ex:
        raise CSGGeometryValidationError(
            f"{stack_ccw} and {stack_cw} has mismatched exterior vertex!"
        )


def check_cells_are_neighbours_by_vertices(in_cell: Cell, ex_cell: Cell) -> None:
    """Check that two cells in the same stack shares the appropriate corners.

    Parameters
    ----------
    in_cell:
        Cell on the interior side of the RadialInterface
    ex_cell:
        Cell on the exterior side of the RadialInterface

    Raises
    ------
    CSGGeometryValidationError
        Raised if the two stacks don't have 2 matching vertices where we expect them.
    """
    if in_cell.vertices.ccw_ex != ex_cell.vertices.ccw_in:
        raise CSGGeometryValidationError(
            f"{in_cell} and {ex_cell} has mismatched ccw vertex!"
        )
    if in_cell.vertices.cw_ex != ex_cell.vertices.cw_in:
        raise CSGGeometryValidationError(
            f"{in_cell} and {ex_cell} has mismatched cw vertex!"
        )


def check_cells_are_neighbours_by_face(in_cell: Cell, ex_cell: Cell) -> None:
    """
    Check that two cells in the same stack shares the same :class:`~RadialInterface`.

    Parameters
    ----------
    in_cell:
        Cell on the interior side of the RadialInterface
    ex_cell:
        Cell on the exterior side of the RadialInterface

    Raises
    ------
    CSGGeometryValidationError
        Raised if the two stacks don't share a :class:`~RadialInterface` where we expect
        them.
    """
    if in_cell.ex_face != ex_cell.in_face:
        raise CSGGeometryValidationError(
            f"{in_cell} and {ex_cell} does not share an interface!"
        )


def check_cell_wall_alignment(
    cell: Cell, cell_wall_ccw: CellWall, cell_wall_cw: CellWall
):
    """Check if a cell's corners are all lying on its cell walls.

    Raises
    ------
    CSGGeometryValidationError
        Thrown if cells' corners do not start and end in the ccw and cw cell walls.
    """
    # ccw wall
    if not (
        cell_wall_ccw.includes_point(cell.vertices.ccw_in)
        and cell_wall_ccw.includes_point(cell.vertices.ccw_ex)
    ):
        raise CSGGeometryValidationError(
            f"{cell}'s ccw edge is misaligned from {cell_wall_ccw}."
        )
    # cw wall
    if not (
        cell_wall_cw.includes_point(cell.vertices.cw_in)
        and cell_wall_cw.includes_point(cell.vertices.cw_ex)
    ):
        raise CSGGeometryValidationError(
            f"{cell}'s cw edge is misaligned from {cell_wall_cw}."
        )

# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Defining (and changing) the radial (side) walls of PreCell in PreCellArrays."""

from __future__ import annotations

from enum import IntEnum
from math import fsum
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt
from scipy.optimize import newton

from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.error import GeometryError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from bluemira.neutronics.make_pre_cell import PreCellArray


class Vert(IntEnum):
    """Vertices index for cells"""

    ext_end = 0
    int_start = 1
    int_end = 2
    ext_start = 3


def polygon_revolve_signed_volume(polygon: npt.ArrayLike) -> float:
    """
    Revolve a polygon along the z axis, and return the volume.

    A polygon placed in the RHS of the z-axis in the xz plane would have positive volume
    if it runs clockwise, and negative volume if it runs counter-clockwise.

    Similarly a polygon placed on the LHS of the z-axis in the xz plane would have
    negative volume if it runs clockwise, positive volume if it runs counter-clockwise.

    Parameters
    ----------
    polygon:
        Stores the x-z coordinate pairs of the four coordinates.

    Notes
    -----
    Consider one edge of the polygon, which has two vertices, $p$ and $c$.
    TODO: insert graphics

    When revolved around the z-axis, this trapezium forms a the frustum of a cone.
    The expression for the volume of this frustrum needs to be modified to avoid
    ZeroDivisionError, thus it is recast into the following (also the simplest) form:
    :math:`V = \\frac{\\pi}{3} (p_z - c_z) (p_x^2 + p_x c_x + c_x^2)`.

    Adding together the signed volume of all edges, the excess negative volume from one
    side would cancel out the excess positive volume from the other, such that
    abs(signed volume)= the volume of the polygon after being revolved around the z-axis.
    """
    polygon = np.array(polygon)
    if np.ndim(polygon) != 2 or np.shape(polygon)[1] != 2:  # noqa: PLR2004
        raise ValueError("This function takes in an np.ndarray of shape (N, 2).")
    previous_points, current_points = polygon, np.roll(polygon, -1, axis=0)
    px, pz = previous_points[:, 0], previous_points[:, -1]
    cx, cz = current_points[:, 0], current_points[:, -1]
    volume_3_over_pi = (pz - cz) * (px**2 + px * cx + cx**2)
    return np.pi / 3 * fsum(volume_3_over_pi)


def partial_diff_of_volume(
    three_vertices: Sequence[Sequence[float]],
    normalised_direction_vector: Iterable[float],
) -> float:
    """
    Gives the relationship between how the the solid volume varies with the position of
    one of its verticies. More precisely, it gives gives the the partial derivative of
    the volume of the solid revolved out of a polygon when one vertex of that polygon
    is moved in the direction specified by normalised_direction_vector.

    Parameters
    ----------
    three_vertices:
        Contain (x, z) coordinates of the polygon. It extracts only the vertex being
        moved, and the two vertices around it. three_vertices[0] and three_vertices[2]
        are anchor vertices that cannot be adjusted. shape (3, 2)
    normalised_direction_vector:
        Direction that the point is allowed to move in. shape = (2,)

    Notes
    -----
    Let there be 3 points, :math:`q`, :math:`r`, and :math:`s`, forming two edges of a
    polygon. When r is moved, the polygon's revolved solid volume changes.
    After a hefty amount of derivation, everything cancels out to give the expression
    .. math::

        \\frac{dV}{d r_z} = q_z q_x - r_z q_x + 2 q_z r_x - 2 s_z r_x + r_z s_x - s_z s_x
        \\frac{dV}{d r_x} = (q_x + r_x + s_x) (s_x - q_x)


    The dot product between the direction of motion and the vector :math:`\\frac{dV}{dr}`
    gives the required scalar derivative showing "how much does the volume change when
    r is moved in a certain direction by one unit length".
    """
    (qx, qz), (rx, rz), (sx, sz) = three_vertices
    x_component = qz * qx - rz * qx + 2 * qz * rx - 2 * sz * rx + rz * sx - sz * sx
    z_component = (qx + rx + sx) * (sx - qx)
    xz_derivatives = np.array([x_component, z_component]).T
    return np.pi / 3 * np.dot(normalised_direction_vector, xz_derivatives)


class CellWalls:
    """
    A list of start- and end-location vectors of all of the walls dividing neighbouring
    pre-cells.
    """

    def __init__(self, cell_walls: npt.ArrayLike):
        """
        Parameters
        ----------
        cell_walls
            array of shape (N+1, 2, 2)
            where N = number of cells, so the axis=0 describes the N+1 walls on either
            side of each cell, the axis=1 describes the the start and end points, and
            axis=2 describes the r and z coordinates.
        """
        self.cell_walls = np.array(cell_walls)
        if np.shape(self.cell_walls)[1:] != (2, 2):
            raise ValueError(
                "Expected N values of start and end xz coordinates, i.e. "
                f"shape = (N+1, 2, 2); got {np.shape(self.cell_walls)}."
            )
        self._starts = self.cell_walls[:, 0]  # shape = (N+1, 2)
        self._init_ends = self.cell_walls[:, 1]  # shape = (N+1, 2)
        vector = self._init_ends - self._starts  # shape = (N+1, 2)
        self.original_lengths = np.linalg.norm(vector, axis=-1)
        self.directions = (vector.T / self.original_lengths).T
        self.num_cells: int = len(self) - 1
        self.check_volumes_and_lengths()

    def __len__(self) -> int:
        """Number of cell wall panels"""
        return len(self.cell_walls)

    def __getitem__(self, index_or_slice) -> npt.NDArray | float:
        """Get cell wall panel"""
        return self.cell_walls[index_or_slice]

    def __setitem__(self, index_or_slice, new_coordinates: npt.NDArray | float):
        """
        self[:, :, :] = ... can completely reset some coordinates.
        However, a full-reset should be avoided because we don't want to mess with the
        start rz coordinates.
        """
        self.cell_walls[index_or_slice] = new_coordinates

    def __repr__(self) -> str:
        """String representation"""
        return (
            super().__repr__().replace(" at ", f" of {len(self.cell_walls)} walls at ")
        )

    def copy(self) -> CellWalls:
        """Copy cell wall"""
        return CellWalls(self.cell_walls.copy())

    @classmethod
    def from_pre_cell_array(cls, pre_cell_array: PreCellArray) -> CellWalls:
        """Use the corner vertices in an array of pre-cells to make a CellWalls."""
        # cut each coordinates down from having shape (3, 1) down to (2,)
        return cls([
            *(
                (c.vertex[:, Vert.int_end], c.vertex[:, Vert.ext_start])
                for c in pre_cell_array
            ),
            (
                pre_cell_array[-1].vertex[:, Vert.int_start],
                pre_cell_array[-1].vertex[:, Vert.ext_end],
            ),
        ])

    @classmethod
    def from_pre_cell_array_vv(cls, pre_cell_array: PreCellArray) -> CellWalls:
        """
        Use the corner vertices and the vacuum vessel vertices of the pre-cell array to
        make a CellWall.
        """
        return cls([
            *((c.vertex[:, Vert.int_end], c.vv_point[:, 0]) for c in pre_cell_array),
            (
                pre_cell_array[-1].vertex[:, Vert.int_start],
                pre_cell_array[-1].vv_point[:, 1],
            ),
        ])

    @property
    def starts(self) -> npt.NDArray:
        """The start points of each cell wall. shape = (N+1, 2)"""
        return self._starts  # the start points never change value.

    @property
    def ends(self) -> npt.NDArray:
        """The end point changes value depending on the user-set length."""
        return self.cell_walls[:, 1]  # shape = (N+1, 2)

    def calculate_new_end_points(
        self, lengths: float | npt.NDArray[np.float64]
    ) -> npt.NDArray:
        """
        Get the end points of each cell wall if they were changed to have the specified
        lengths. This is different to set_length in that the new end points are returned,
        and the object itself is not modified by the test length(s).

        Parameters
        ----------
        lengths
            np.ndarray of shape = (N+1,) where N+1 == number of cell walls.
            It can also be a scalar, which would then be broadcasted into (N+1,).

        Returns
        -------
        new end points
            an array of the same shape as self.starts
        """
        return self.starts + (self.directions.T * lengths).T

    def get_length(self, i: int) -> float:
        """
        Get the length of the i-th cell-wall

        Returns
        -------
        length: float
        """
        _end_i, _start_i = self.cell_walls[i]
        return np.linalg.norm(_end_i - _start_i)

    def set_length(self, i, new_length):
        """Set the length of the i-th cell-wall"""
        self.cell_walls[i, 1] = self.cell_walls[i, 0] + self.directions[i] * new_length

    @property
    def lengths(self):
        """Current lengths of the cell walls."""
        return np.linalg.norm(self.ends - self.starts, axis=-1)  # shape = (N+1)

    def get_volume(self, i):
        """
        Get the volume of the i-ith cell

        Returns
        -------
        volume: float
        """
        return polygon_revolve_signed_volume(
            np.concatenate([self.cell_walls[i], self.cell_walls[i + 1][::-1]])
        )

    @property
    def volumes(self):
        """
        Current volumes of the (simplified) cells created by joining straight lines
        between neighbouring cell walls.
        """
        # shape = (N+1,)
        return np.asarray([self.get_volume(i) for i in range(self.num_cells)])

    def check_volumes_and_lengths(self):
        """
        Ensure all cells have positive volumes, to minimise the risk of self-intersecting
        lines and negative lengths
        """
        if not (all(self.volumes > 0) and all(self.lengths > 0)):
            raise GeometryError("At least 1 cell has non-positive volumes!")

    def volume_of_cells_neighbouring(self, i, test_length):
        """
        Get the volume of cell[i-1] and cell[i] when cell_wall[i] is set to the
        test_length.

        Returns
        -------
        volume: float
        """
        start_i, dir_i = self.starts[i], self.directions[i]
        new_end = start_i + dir_i * test_length
        prev_wall, next_wall = self.cell_walls[i - 1 : i + 2 : 2]
        return polygon_revolve_signed_volume([
            prev_wall[0],
            prev_wall[1],
            new_end,
            start_i,
        ]) + polygon_revolve_signed_volume([
            start_i,
            new_end,
            next_wall[1],
            next_wall[0],
        ])

    def volume_derivative_of_cells_neighbouring(self, i, test_length):
        """
        Measure the derivative on the volume of cell[i-1] and cell[i] w.r.t. to
        length of cell_wall[i], at cell_wall[i] length = test_length.

        Returns
        -------
        dV/dl: float
        """
        start_i, dir_i = self.starts[i], self.directions[i]
        new_end = start_i + dir_i * test_length
        prev_end, next_end = self.ends[i - 1 : i + 2 : 2]
        return partial_diff_of_volume(
            [prev_end, new_end, start_i], dir_i
        ) + partial_diff_of_volume([start_i, new_end, next_end], dir_i)

    def optimise_to_match_individual_volumes(
        self, volume_list: Iterable[float], *, max_iter=1000
    ):
        """
        Allow the lengths of each wall to increase, so that the overall volumes are
        preserved as much as possible. Assuming the entire exterior curve is convex,
        then our linear approximation is only going to under-approximate. Therefore
        to achieve better approximation, we only need to increase the lengths.
        """
        if self.num_cells <= 1:
            return

        target_volumes = np.array(list(volume_list))
        if self.num_cells == 2:  # noqa: PLR2004
            # only one single step is required for the optimisation
            def volume_excess(new_length):
                return self.volume_of_cells_neighbouring(1, new_length) - sum(
                    target_volumes
                )

            def derivative(new_length):
                return self.volume_derivative_of_cells_neighbouring(1, new_length)

            self.set_length(1, newton(volume_excess, self.get_length(1), derivative))
            self.check_volumes_and_lengths()
            return

        # if more than 3 walls (more than 2 cells)
        step_direction, i, i_min, i_max = 1, 1, 1, self.num_cells - 1

        num_passes_counter = -1
        forward_pass_result = np.zeros(self.num_cells + 1)

        while num_passes_counter < max_iter:
            # do not allow length to decrease beyond their original value.

            def excess_volume(test_length, i=i):
                return self.volume_of_cells_neighbouring(i, test_length) - sum(
                    target_volumes[i - 1 : i + 1]
                )

            def dV_dl(test_length, i=i):  # noqa: N802
                return self.volume_derivative_of_cells_neighbouring(i, test_length)

            self.set_length(
                i,
                newton(excess_volume, self.get_length(i), dV_dl)
                if excess_volume(self.original_lengths[i]) < 0
                else self.original_lengths[i],
            )
            if i == i_min:
                # hitting the left end: bounce to the right
                step_direction = 1
                num_passes_counter += 1
                backward_pass_result = self.lengths.copy()
                # termination condition
                if np.allclose(
                    backward_pass_result, forward_pass_result, rtol=0, atol=EPS_FREECAD
                ):
                    bluemira_debug(
                        "Cell volume-matching optimisation successful."
                        "Terminating iterative cell wall length adjustment after "
                        f"{num_passes_counter} passes."
                    )
                    self.check_volumes_and_lengths()
                    return
            elif i == i_max:
                # hitting the right end: bounce to the left
                step_direction = -1
                num_passes_counter += 1
                forward_pass_result = self.lengths.copy()
            i += step_direction
        bluemira_warn(
            "Optimisation failed: Did not converge within"
            f"{num_passes_counter} iterations!"
        )

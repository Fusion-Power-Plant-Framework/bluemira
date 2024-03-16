# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Make pre-cells using bluemira wires."""
# ruff: noqa: PLR2004, D105

from __future__ import annotations

from typing import List, Union

import numpy as np
from numpy import typing as npt

from bluemira.display import plot_2d, show_cad
from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.coordinates import Coordinates, get_bisection_line
from bluemira.geometry.error import GeometryError
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import make_polygon, raise_error_if_overlap, revolve_shape
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.radial_wall import CellWalls, VerticesCoordinates


class PreCell:
    """
    A pre-cell is the BluemiraWire outlining the reactor cross-section
    BEFORE they have been simplified into straight-lines.
    Unlike a Cell, its outline may be constructed from curved lines.
    """

    def __init__(
        self,
        interior_wire: Union[BluemiraWire, Coordinates],
        exterior_wire: BluemiraWire,
    ):
        """
        Parameters
        ----------
        interior_wire

            Either: A wire representing the interior-boundary (i.e. plasma-facing side)
                of a blanket's pre-cell, running in the anti-clockwise direction when
                viewing the right hand side poloidal cross-section,
                i.e. downwards if inboard, upwards if outboard.
            or: a single Coordinates point, representing a point on the interior-boundary

        exterior_wire

            A wire representing the exterior-boundary (i.e. air-facing side) of a
                blanket's pre-cell, running in the clockwise direction when viewing the
                right hand side poloidal cross-section,
                i.e. upwards if inboard, downwards if outboard.

        Variables
        ---------
        _inner_curve
            A wire starting the end of the exterior_wire, and ending at the start of the
            exterior_wire. This should pass through the interior_wire.
        vertex
            :class:`~bluemira.neutronics.radial_wall.Vertices` of vertices, for
            convenient retrieval later.
        outline
            The wire outlining the PreCell
        half_solid
            The solid created by revolving the outline around the z-axis by 180°.
        """
        self.interior_wire = interior_wire
        self.exterior_wire = exterior_wire
        raise_error_if_overlap(
            self.exterior_wire, self.interior_wire, "interior wire", "exterior wire"
        )
        ext_start, ext_end = exterior_wire.start_point(), exterior_wire.end_point()
        if isinstance(interior_wire, Coordinates):
            int_start = int_end = interior_wire
            self._inner_curve = make_polygon(
                np.array([ext_end, interior_wire, ext_start]).T, closed=False
            )
        else:
            int_start, int_end = interior_wire.start_point(), interior_wire.end_point()
            self._out2in = make_polygon(np.array([ext_end, int_start]).T, closed=False)
            self._in2out = make_polygon(np.array([int_end, ext_start]).T, closed=False)
            self._inner_curve = BluemiraWire([
                self._out2in,
                self.interior_wire,
                self._in2out,
            ])
            raise_error_if_overlap(
                self._out2in,
                self._in2out,
                "cell-start cutting plane",
                "cell-end cutting plane",
            )
        self.vertex = VerticesCoordinates(ext_end, int_start, int_end, ext_start).to_2D()
        self.outline = BluemiraWire([self.exterior_wire, self._inner_curve])
        # Revolve only up to 180° for easier viewing
        self.half_solid = BluemiraSolid(revolve_shape(self.outline))

    def plot_2d(self, *args, **kwargs) -> None:
        """Plot the outline in 2D"""
        plot_2d(self.outline, *args, **kwargs)

    def show_cad(self, *args, **kwargs) -> None:
        """Plot the outline in 3D"""
        show_cad(self.half_solid, *args, **kwargs)

    @property
    def cell_walls(self):
        """
        The side (clockwise side and counter-clockwise) walls of this cell.
        Only create it when called, because some instances of PreCell will never use it.
        """
        if not hasattr(self, "_cell_walls"):
            self._cell_walls = CellWalls([
                [self.vertex.interior_end, self.vertex.exterior_start],
                [self.vertex.interior_start, self.vertex.exterior_end],
            ])
        return self._cell_walls

    @property
    def normal_to_interior(self):
        """
        The vector pointing from the interior_wire direction towards the exterior_wire,
        specifically, it's perpendicular to the interior_wire.
        Also only created when called, because it's not needed
        """
        if not hasattr(self, "_normal_to_interior"):
            if isinstance(self.interior_wire, Coordinates):
                self._normal_to_interior = get_bisection_line(
                    *self.cell_walls[:].reshape([4, 2])
                )[1]
            else:
                interior_vector = self.cell_walls.starts[0] - self.cell_walls.starts[1]
                normal = np.array([interior_vector[-1], -interior_vector[0]])
                self._normal_to_interior = normal / np.linalg.norm(normal)

        return self._normal_to_interior

    def get_cell_wall_cut_points_by_fraction(
        self, fraction: float
    ) -> npt.NDArray[float]:
        """
        Find the cut points on the cell's side walls by multiplying the original lengths
        by a fraction. When fraction=0, this returns the interior_start and interior_end.

        Parameters
        ----------
        fraction: float
            A scalar value

        Returns
        -------
        new end points
            The position of the pre-cell wall end points at the required fraction, array
            of shape (2, 2) [[cw_wall x, cw_wall z], [ccw_wall x, ccw_wall z]].
        """
        new_lengths = self.cell_walls.lengths * fraction
        return self.cell_walls.calculate_new_end_points(new_lengths)

    def get_cell_wall_cut_points_by_thickness(self, thickness):
        """
        Offset a line parallel to the interior_wire towards the exterior direction.
        Then, find where this line intersect the cell's side walls.

        Parameters
        ----------
        fraction: float
            A scalar value

        Returns
        -------
        new end points
            The position of the pre-cell wall end points at the required thickness, array
            of shape (2, 2) [[cw_wall x, cw_wall z], [ccw_wall x, ccw_wall z]].
        """
        projection_weight = self.cell_walls.directions @ self.normal_to_interior
        length_increases = thickness / projection_weight

        return self.cell_walls.calculate_new_end_points(length_increases)


class PreCellArray:
    """
    A list of pre-cells materials

    Parameters
    ----------
    list_of_pre_cells:
        An adjacent list of pre-cells

    Variables
    ---------
    volumes: List[float]
        Volume of each pre-cell
    """

    def __init__(self, list_of_pre_cells: List[PreCell]):
        """The list of pre-cells must be ajacent to each other."""
        self.pre_cells = list(list_of_pre_cells)
        for this_cell, next_cell in zip(self[:-1], self[1:]):
            # perform check that they are actually adjacent
            this_wall = (this_cell.vertex.exterior_end, this_cell.vertex.interior_start)
            next_wall = (next_cell.vertex.exterior_start, next_cell.vertex.interior_end)
            if not (
                np.allclose(this_wall[0], next_wall[0], atol=0, rtol=EPS_FREECAD)
                and np.allclose(this_wall[1], next_wall[1], atol=0, rtol=EPS_FREECAD)
            ):
                raise GeometryError(
                    "Adjacent pre-cells are expected to have matching"
                    f"corners; but instead we have {this_wall}!={next_wall}."
                )
        self.volumes = [pre_cell.half_solid.volume * 2 for pre_cell in self]

    def straighten_exterior(self, preserve_volume: bool = False) -> PreCellArray:
        """
        Turn the exterior curves of each cell into a straight edge.
        This is done at the PreCellArray level instead of the PreCell level to allow
        volume preservation, see Parameters below for more details.

        Parameters
        ----------
        preserve_volume: bool
            Whether to preserve the volume of each cell during the transformation from
            pre-cell with curved-edge to pre-cell with straight edges.
            If True, increase the length of the cut lines appropriately to compensate for
            the volume loss due to the straight line approximation.
        """
        cell_walls = CellWalls.from_pre_cell_array(self)
        if preserve_volume:
            cell_walls.optimize_to_match_individual_volumes(self.volumes)
        new_pre_cells = []
        for i, (this_wall, next_wall) in enumerate(zip(cell_walls[:-1], cell_walls[1:])):
            exterior = make_polygon(
                [
                    [this_wall[1, 0], next_wall[1, 0]],
                    [0, 0],
                    [this_wall[1, 1], next_wall[1, 1]],
                ],  # fill it back up to 3D to make the polygon
                label=f"straight edge approximation of the exterior of pre-cell {i}",
                closed=False,
            )
            new_pre_cells.append(PreCell(self[i].interior_wire, exterior))
        return PreCellArray(new_pre_cells)

    def plot_2d(self) -> None:  # noqa: D102
        plot_2d([c.outline for c in self])

    def show_cad(self) -> None:  # noqa: D102
        show_cad([c.half_solid for c in self])

    def __len__(self) -> int:
        return self.pre_cells.__len__()

    def __iter__(self):
        return self.pre_cells.__iter__()

    def __getitem__(self, index_or_slice) -> Union[List[PreCell], PreCell]:
        return self.pre_cells.__getitem__(index_or_slice)

    def __add__(self, other_array) -> PreCellArray:
        """Adding two list together to create a new one."""
        return PreCellArray(self.pre_cells + other_array)

    def __repr__(self) -> str:
        return super().__repr__().replace(" at ", f" of {len(self)} PreCells at ")

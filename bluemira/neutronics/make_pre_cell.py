# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Make pre-cells using bluemira wires."""
# ruff: noqa: PLR2004, D105

from __future__ import annotations

from collections import abc

import numpy as np
from numpy import typing as npt

from bluemira.base.constants import EPS
from bluemira.display import plot_2d, show_cad
from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.coordinates import (
    Coordinates,
    choose_direction,
    get_bisection_line,
)
from bluemira.geometry.error import GeometryError
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import make_polygon, raise_error_if_overlap, revolve_shape
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.radial_wall import CellWalls, Vertices, VerticesCoordinates
from bluemira.neutronics.wires import (
    CircleInfo,
    StraightLineInfo,
    WireInfo,
    WireInfoList,
)

CCW_90 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
CW_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])


class PreCell:  # TODO: Rename this as BlanketPreCell
    """
    A pre-cell is the BluemiraWire outlining the reactor cross-section
    BEFORE they have been simplified into straight-lines.
    Unlike a Cell, its outline may be constructed from curved lines.
    """

    def __init__(
        self,
        interior_wire: BluemiraWire | Coordinates,
        exterior_wire: BluemiraWire,
    ):
        """
        Parameters
        ----------
        interior_wire

            Either: A wire representing the interior-boundary (i.e. plasma-facing side)
                of a blanket's pre-cell, running in the counter-clockwise direction when
                viewing the right hand side poloidal cross-section,
                i.e. downwards if inboard, upwards if outboard.
            or: a single Coordinates point, representing a point on the interior-boundary

        exterior_wire

            A wire representing the exterior-boundary of a
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
    def cell_walls(self) -> CellWalls:
        """
        The side (clockwise side and counter-clockwise) walls of this cell.
        Only create it when called, because some instances of PreCell will never use it.

        it is of type :class:`~bluemira.neutronics.radial_wall.CellWalls`.
        """
        if not hasattr(self, "_cell_walls"):
            self._cell_walls = CellWalls([
                [self.vertex.interior_end, self.vertex.exterior_start],
                [self.vertex.interior_start, self.vertex.exterior_end],
            ])
        return self._cell_walls

    @property
    def normal_to_interior(self) -> npt.NDArray:
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

    def get_cell_wall_cut_points_by_fraction(self, fraction: float) -> npt.NDArray:
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


class PreCellArray(abc.Sequence):
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

    def __init__(self, list_of_pre_cells: list[PreCell]):
        """The list of pre-cells must be ajacent to each other."""
        self.pre_cells = list(list_of_pre_cells)
        for this_cell, next_cell in zip(self[:-1], self[1:], strict=False):
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
        for i, (this_wall, next_wall) in enumerate(
            zip(cell_walls[:-1], cell_walls[1:], strict=False)
        ):
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

    def plot_2d(self, *args, **kwargs) -> None:  # noqa: D102
        plot_2d([pc.outline for pc in self], *args, **kwargs)

    def show_cad(self, *args, **kwargs) -> None:  # noqa: D102
        show_cad([pc.half_solid for pc in self], *args, **kwargs)

    def get_exterior_vertices(self) -> npt.NDArray:
        """
        Returns all of the vertices on the exterior side of the pre-cell array.

        Returns
        -------
        exterior_vertices: npt.NDArray of shape (N+1, 3)
            Arranged clockwise (inboard to outboard).
        """
        cell_walls = CellWalls.from_pre_cell_array(self)
        return np.insert(cell_walls[:, 1], 1, 0, axis=-1)

    def get_interior_vertices(self) -> npt.NDArray:
        """
        Returns all of the vertices on the interior side of the pre-cell array.

        Returns
        -------
        interior_vertices: npt.NDArray of shape (N+1, 3)
            Arranged clockwise (inboard to outboard).
        """
        cell_walls = CellWalls.from_pre_cell_array(self)
        return np.insert(cell_walls[:, 0], 1, 0, axis=-1)

    def __len__(self) -> int:
        return self.pre_cells.__len__()

    def __getitem__(self, index_or_slice) -> list[PreCell] | PreCell:
        return self.pre_cells.__getitem__(index_or_slice)

    def __add__(self, other_array) -> PreCellArray:
        """Adding two list together to create a new one."""
        if isinstance(other_array, PreCellArray):
            return PreCellArray(self.pre_cells + other_array)
        raise TypeError(
            "Addition not implemented between PreCellArray and "
            f"{other_array.__class__}"
        )

    def __repr__(self) -> str:
        return super().__repr__().replace(" at ", f" of {len(self)} PreCells at ")


def ratio_of_distances(
    point_of_interest: npt.NDArray[float],
    anchor1: npt.NDArray[float],
    normal1: npt.NDArray[float],
    anchor2: npt.NDArray[float],
    normal2: npt.NDArray[float],
) -> npt.NDArray[float]:
    """
    Find how close a point is to line 1 and line 2, and then express that ratio as a
    tuple of floats that sums up to unit.
    Each line is given by the user by specifying any point on that line, and a direction
    NORMAL to that line. The point_of_interest must lie on the positive side of the line.

    Parameters
    ----------
    point_of_interest
        point to which we want to calculate the ratio of distances.
    anchor1, anchor2:
        Any point on line 1 and line 2 respectively.
    normal1, normal2:
        The positive distance direction of line 1 and line 2 respectively.

    Returns
    -------
    dist_to_1, dist_to_2:
        ratio of distances. Sum of these two numbers should yield unity (1.0).
    """
    dist_to_1 = (point_of_interest - anchor1) @ normal1
    dist_to_2 = (point_of_interest - anchor2) @ normal2
    if dist_to_1 < -EPS or dist_to_2 < -EPS:
        raise GeometryError(
            "Expecting point_of_interest to lie on the positive side of both lines!"
        )
    total_dist = dist_to_1 + dist_to_2
    return np.array([dist_to_1, dist_to_2]) / total_dist


def find_equidistant_point(
    point1: npt.NDArray[float], point2: npt.NDArray[float], distance: float
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Find the two (or 0) points on a 2D plane that are equidistant to each other.

    Parameters
    ----------
    point1, point2:
        2D points, each with shape (2,)
    distance:
        the distance that both points must obey by.

    Returns
    -------
    intersection1, intersection2:
        The two intersection points of circle1 and circle2.
    """
    mid_point = (point1 + point2) / 2
    sep = point2 - point1
    half_sep = np.linalg.norm(sep) / 2  # scalar
    if half_sep > distance:
        raise GeometryError("The two points are separated by > 2 * distance!")
    orth_length = np.sqrt(distance**2 - half_sep**2)
    orth_dir = np.array([-sep[1], sep[0]])
    orth_dir /= np.linalg.norm(orth_dir)
    return mid_point + (orth_dir * orth_length), mid_point - (orth_dir * orth_length)


def pick_higher_point(
    point1: npt.NDArray[float], point2: npt.NDArray[float], vector: npt.NDArray[float]
):
    """Pick the point that, when projected onto `vector`, gives a higher value."""
    if (vector @ point1) > (vector @ point2):
        return point1
    return point2


def calculate_new_circle(
    old_circle_info: CircleInfo, new_points: npt.NDArray
) -> CircleInfo:
    """
    Calculate how far does the new circle get shifted.

    Parameters
    ----------
    old_circle_info:
        an object accessed by WireInfoList[i].key_points.
        info on circle where the start_point and end_point are each of shape (3,).
    new_points
        array of shape (2, 3)

    Returns
    -------
    new_circle_info
        An instance of CircleInfo representing the new (scaled) arc of circle.
    """
    new_chord_vector = np.diff(new_points, axis=0)
    old_chord_vector = np.diff(old_circle_info[:2], axis=0)
    scale_factor = np.linalg.norm(new_chord_vector) / np.linalg.norm(old_chord_vector)
    new_radius = old_circle_info.radius * scale_factor
    possible_centers = find_equidistant_point(*new_points[:, ::2], new_radius)
    center1, center2 = np.insert(possible_centers, 1, 0, axis=1)

    old_chord_mid_point = np.mean(old_circle_info[:2], axis=0)
    old_radius_vector = np.array(old_circle_info.center) - old_chord_mid_point
    # chord should stay on the same side of the center after transformation.
    new_center = pick_higher_point(center1, center2, old_radius_vector)

    return CircleInfo(*new_points, new_center, new_radius)


class DivertorPreCell:
    """
    An intermediate class between the bluemira wire and the final csg product.
    A divertor pre-cell is the equivalent of a blanket's pre-cell, but for the divertor.
    """

    def __init__(
        self,
        interior_wire: WireInfoList,
        exterior_wire: WireInfoList,
    ):
        """
        Parameters
        ----------
        interior_wire
            WireInfoList of a wire on the interior side of the cell running
            counter-clockwise
        exterior_wire
            WireInfoList of a wire on the exterior side of the cell running clockwise

        Variables
        ---------
        cw_wall
            a WireInfoList of len==1, representing the straight line cut on the clockwise
            side of the divertor pre-cell
        ccw_wall
            a WireInfoList of len==1, representing the straight line cut on the
            counter-clockwise side of the divertor pre-cell
        vertex
            Vertices[cadapi.apiVector] denoting the four corners of the divertor pre-cell
        """
        self.interior_wire = interior_wire
        self.exterior_wire = exterior_wire
        # cw_wall and ccw_wall are of type WireInfoLists!!
        self.cw_wall = WireInfoList([
            WireInfo.from_2P(
                self.exterior_wire.end_point, self.interior_wire.start_point
            )
        ])
        self.ccw_wall = WireInfoList([
            WireInfo.from_2P(
                self.interior_wire.end_point, self.exterior_wire.start_point
            )
        ])
        self.vertex = Vertices(
            self.cw_wall.start_point,
            self.cw_wall.end_point,
            self.ccw_wall.start_point,
            self.ccw_wall.end_point,
        )

    def plot_2d(self, *args, **kwargs) -> None:  # noqa: D102
        plot_2d([self.outline], *args, **kwargs)

    @property
    def outline(self):
        """
        We don't need the volume value, so we're only going to generate the outline
        when the user wants to plot it.
        """
        if not hasattr(self, "_outline"):
            self._outline = BluemiraWire([
                self.cw_wall.restore_to_wire(),
                self.interior_wire.restore_to_wire(),
                self.ccw_wall.restore_to_wire(),
                self.exterior_wire.restore_to_wire(),
            ])
        return self._outline

    def offset_interior_wire(self, thickness: float) -> WireInfoList:
        """
        Offset the interior wire towards the exterior_wire.
        The true problem of expanding/shrinking a wire is a much more difficult one, so
        I've only opted for a simpler (but incorrect) approach of pushing the wire to a
        desired direction determined by how close it is to the wall.

        TODO: Re-write this method, as it currently do weird things to circles.

        New method should do the following:
        1. Find a point to be our new origin. (Name that point "p")
            This is likely to be a point near exterior_vertices.
        2. Scale down the interior_wire by x%.

        We should give more thought of how to derive/search for the optimal vector
        (p[0], p[-1], x), such that all lines are displaced by

        This proposed method has several new benefits:
        1. The circles will be scaled correctly (center moves towards the new origin
            by x%, radius scaled down by x%).
        2. All tangents are preserved, so no need to change them.
        """
        int_wire_pts = [w.key_points[0] for w in self.interior_wire]
        int_wire_pts.append(self.interior_wire[-1].key_points[1])
        int_wire_pts = np.array(int_wire_pts)  # shape (N+1, 3)
        cw_dir = choose_direction(
            self.cw_wall[0].tangents[0], self.cw_wall.end_point, self.cw_wall.start_point
        )  # assumed normalized
        ccw_dir = choose_direction(
            self.ccw_wall[0].tangents[1],
            self.ccw_wall.start_point,
            self.ccw_wall.end_point,
        )  # assumed normalized
        cw_norm = CCW_90 @ cw_dir
        cw_anchor = self.cw_wall.start_point
        ccw_norm = CW_90 @ ccw_dir
        ccw_anchor = self.ccw_wall.end_point

        shifted_pts = []
        for pt in int_wire_pts:
            weights = ratio_of_distances(pt, cw_anchor, cw_norm, ccw_anchor, ccw_norm)[
                ::-1
            ]
            new_dir = np.array([cw_dir, ccw_dir]).T @ weights
            step = new_dir / np.linalg.norm(new_dir) * thickness
            shifted_pts.append(pt + step)

        info_list = []
        for i, (new_start, new_end) in enumerate(
            zip(shifted_pts[:-1], shifted_pts[1:], strict=False)
        ):
            old_kp = self.interior_wire[i].key_points
            if isinstance(old_kp, StraightLineInfo):
                info_list.append(WireInfo.from_2P(new_start, new_end))
            else:
                info_list.append(
                    WireInfo(
                        calculate_new_circle(old_kp, np.array([new_start, new_end])),
                        # copy the old tangents, which are not going to be correct,
                        # but we won't be using them anyways.
                        self.interior_wire[i].tangents,
                    )
                )
        return WireInfoList(info_list)


class DivertorPreCellArray(abc.Sequence):
    """An array of Divertor pre-cells"""

    def __init__(self, list_of_div_pc: list[DivertorPreCell]):
        self.pre_cells = list(list_of_div_pc)
        # Perform check that they are adjacent
        for prev_cell, curr_cell in zip(self[:-1], self[1:], strict=False):
            if not np.allclose(
                prev_cell.vertex.exterior_start,
                curr_cell.vertex.exterior_end,
                atol=0,
                rtol=EPS_FREECAD,
            ):
                # # Don't need this check below, thus commented out.
                # and np.allclose(prev_cell.vertex.interior_end,
                #                 curr_cell.vertex.interior_start,
                #                 atol=0, rtol=EPS_FREECAD)
                raise GeometryError("Expect neighbouring cells to share corners!")

    def get_exterior_vertices(self) -> npt.NDArray:
        """
        Returns all of the tokamak's poloidal cross-section's outside corners'
        coordinates, in 3D.

        Returns
        -------
        exterior_vertices: npt.NDArray of shape (N+1, 3)
            Arranged counter-clockwise (inboard to outboard).
        """
        exterior_vertices = [
            stack.exterior_wire.get_3D_coordinates()[::-1]
            for stack in self
            # Because cells run counter-clockwise but the exterior_wire themselves runs
            # clockwise, we have to invert the wire during extraction to make it run
            # without double-backing onto itself.
        ]
        return np.concatenate(exterior_vertices)

    def get_interior_vertices(self) -> npt.NDArray:
        """
        Returns all of the tokamak's poloidal cross-section's inside corners'
        coordinates, in 3D.

        Parameters
        ----------
        interior_vertices: npt.NDArray of shape (N+1, 3)
            Arranged counter-clockwise (inboard to outboard).
        """
        interior_vertices = [stack.interior_wire.get_3D_coordinates() for stack in self]
        return np.concatenate(interior_vertices)

    def __len__(self) -> int:
        return self.pre_cells.__len__()

    def __getitem__(self, index_or_slice) -> list[DivertorPreCell] | DivertorPreCell:
        return self.pre_cells.__getitem__(index_or_slice)

    def __add__(self, other_array) -> DivertorPreCellArray:
        if isinstance(other_array, DivertorPreCellArray):
            return DivertorPreCellArray(self.pre_cell + self.pre_cell)
        raise TypeError(
            "Addition not implemented between DivertorPreCellArray and "
            f"{other_array.__class__}"
        )

    def __repr__(self) -> str:
        return (
            super().__repr__().replace(" at ", f" of {len(self)} DivertorPreCells at ")
        )

    def plot_2d(self, *args, **kwargs) -> None:  # noqa: D102
        plot_2d([dpc.outline for dpc in self], *args, **kwargs)

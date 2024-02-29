# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Create csg geometry from bluemira wires."""

from __future__ import annotations
import time
from dataclasses import dataclass
from collections import namedtuple
from typing import Optional, Union, List, Callable
from math import fsum

import numpy as np
from numpy import typing as npt

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display import plot_2d, show_cad  #, plot_3d
from bluemira.geometry.constants import EPS_FREECAD, D_TOLERANCE
from bluemira.geometry.coordinates import (Coordinates,
                                            get_bisection_line,
                                            vector_intersect)
from bluemira.geometry.error import GeometryError
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import (get_wire_plane_intersect,
                                    make_polygon,
                                    revolve_shape,
                                    raise_error_if_overlap,)
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.params import TokamakGeometry


class PreCell:
    """
    A pre-cell is the BluemiraWire outlining the reactor cross-section
    BEFORE they have been simplified into straight-lines.
    Unlike a Cell, it may be made of curved lines.
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
        """
        self.interior_wire = interior_wire
        self.exterior_wire = exterior_wire
        raise_error_if_overlap(self.exterior_wire, self.interior_wire,
                                "interior wire", "exterior wire")
        ext_start, ext_end = exterior_wire.start_point(), exterior_wire.end_point()
        if isinstance(interior_wire, Coordinates):
            int_start, int_end = interior_wire, interior_wire
            self._inner_curve = make_polygon(
                [ext_end, interior_wire, ext_start], closed=False
            )
        else:
            int_start, int_end = interior_wire.start_point(), interior_wire.end_point()
            self._out2in = make_polygon([ext_end, int_start], closed=False)
            self._in2out = make_polygon([int_end, ext_start], closed=False)
            self._inner_curve = BluemiraWire([
                self._out2in,
                self.interior_wire,
                self._in2out,
            ])
            raise_error_if_overlap(self._out2in, self._in2out,
                    "cell-start cutting plane", "cell-end cutting plane")
        self.vertex = PreCellWireVerticesNames(ext_end, int_start, int_end, ext_start)
        self.outline = BluemiraWire([self.exterior_wire, self._inner_curve])
        # Revolve only up to 180° for easier viewing
        self.half_solid = BluemiraSolid(revolve_shape(self.outline))

    def plot_2d(self, *args, **kwargs) -> None:
        """Plot the outline in 2D"""
        plot_2d(self.outline, *args, **kwargs)

    def show_cad(self, *args, **kwargs) -> None:
        """Plot the outline in 3D"""
        show_cad(self.half_solid, *args, **kwargs)

    def _get_volume_approximation_error(self) -> float:
        """
        Get the volume lost by by approximating the interior and exterior curve with
        straight lines.

        Returns
        -------
        self.lost_volume: float
            The volume lost. [m^3]
        """
        self._approximator_exterior_straightline = make_polygon([
                self._inner_curve.end_point(),
                self._inner_curve.start_point()
            ], closed=False)
        self.approximator_outline = BluemiraWire([
                self._approximator_exterior_straightline, self._inner_curve
            ])
        self.approximator_half_solid = revolve_shape(self.approximator_outline)
        self.lost_volume = self.half_solid.volume - self.approximator_half_solid.volume
        return self.lost_volume*2 # volume lost by half-

    def _get_volume_approximation_error_fraction(self) -> float:
        lost_volume = self._get_volume_approximation_error()
        return lost_volume/self.volume


def polygon_revolve_signed_volume(polygon: NDArray[NDArray[float]]) -> float:
    """
    Revolve a polygon along the z axis, and return the volume.

    A polgyon placed in the RHS of the z-xis in the xz plane would have positive volume
    if it runs clockwise, and negative volume if it runs anticlockwise.

    Similarly a polygon placed on the LHS of the z-axis in the xz plane would have
    negative volume if it runs clockwise, positive volume if it runs anti-clockwise.

    Parameters
    ----------
    polygon: ndarray of shape (N, 2)
        Stores the x-z coordinate pairs of the four coordinates.

    Notes
    -----
    TODO: add formula later.
    """
    polygon = np.array(polygon)
    if np.ndim(polygon)!=2 or np.shape(polygon)[1]!=2:
        raise ValueError("This function takes in an np.ndarray of shape (N, 2).")
    previous_points, current_points = polygon, np.roll(polygon, -1, axis=0)
    px, pz = previous_points[:, 0], previous_points[:, -1]
    cx, cz = current_points[:, 0], current_points[:, -1]
    volume_3_over_pi = (pz - cz) * (px**2 + px*cx + cx**2)
    return np.pi/3 * fsum(volume_3_over_pi)

PreCellWireVerticesNames = namedtuple('PreCellWireVerticesNames', ["exterior_end", "interior_start", "interior_end", "exterior_start"])


def partial_diff_of_volume(three_vertices: Iterable[Iterable[float]],
        normalized_direction_vector: Iterable[float]
    ) -> float:
    """
    This function gives the relationship between how the the solid volume varies with
    the position of one of its verticies. More precisely, it gives gives the the partial
    derivative of the volume of the solid revolved out of a polygon when one vertex of
    that polygon is moved in the specified direction normalized_direction_vector.

    Parameters
    ----------
    three_vertices: NDArray with shape (3, 2)
        Contain (x, z) coordinates of the polygon. It extracts only the vertex being
        moved, and the two vertices around it. three_vertices[0] and three_vertices[2]
        are anchor vertices that cannot be adjusted.
    normalized_direction_vector: NDArray with shape (2,)
        Direction that the point is allowed to move in.

    Notes
    -----
    TODO: add formula later.
    """
    (qx, qz), (rx, rz), (sx, sz) = three_vertices
    x_component = qz*qx - rz*qx + 2*qz*rx - 2*sz*rx + rz*sx - sz*sx
    z_component = (qx+rx+sx) * (sx-qx)
    xz_derivatives = np.array([x_component, z_component]).T
    return np.pi/3 * np.dot(normalized_direction_vector, xz_derivatives)

def Newtons_method(objective: Callable[[float], float],
                x_guess: float,
                dobjective_dx: Callable[[float], float],
                atol: float=EPS_FREECAD,
                # relaxation_factor: float=0.5,
                ) -> float:
    """
    Try to find the root of a strictly monotonic 1D function.

    Writing our own since we don't want to use scipy.

    Parameters
    ----------
    objective:
        Objective function to be minimized. Takes in float x.
    x_guess:
        Starting guess.
    dobjective_dx:
        Derivative of objective function w.r.t. x.
    atol:
        Absolute value of objective function must be smaller than this to terminate
        optimization successfully.
    """
    deviation, x, dy_dx = objective, x_guess, dobjective_dx
    for i in range(100):
        x -= deviation(x)/dy_dx(x) #* relaxation_factor
        if np.isclose(objective(x), 0, rtol=0, atol=atol):
            return x
    bluemira_warn("Optimization failed: Newton's method did not converge after"
                    f"{i+1} iterations!")
    return x


class PreCellArray:
    def __init__(self, list_of_pre_cells):
        """The list of pre-cells must be ajacent to each other."""
        self.pre_cells = list_of_pre_cells
        for this_cell, next_cell in zip(self.pre_cells[:-1], self.pre_cells[1:]):
            this_wall = (this_cell.vertex.exterior_end, this_cell.vertex.interior_start)
            next_wall = (next_cell.vertex.exterior_start, next_cell.vertex.interior_end)
            if not (np.allclose(this_wall[0], next_wall[0], atol=0, rtol=EPS_FREECAD)
                and np.allclose(this_wall[1], next_wall[1], atol=0, rtol=EPS_FREECAD)):
                raise GeometryError("Adjacent pre-cells are expected to have matching"
                    f"corners; but instead we have {this_wall}!={next_wall}.")
        self.volumes = [pre_cell.half_solid.volume * 2 for pre_cell in self.pre_cells]
        # perform check that they are adjacent
            
    def to_csg(self, preserve_volume: bool=False):
        """
        Convert to planes defined in the CSG array

        Parameters
        ----------
        preserve_volume: bool
            If True, change the length of the cut-lines to preserve the volume of each
            cell as much as possible.
        """
        # pre_cell_walls: shape = (N+1, 2, 2) for N+1 walls of N pre-cells (left and
        # right walls), 2 points per wall, xz coordinates per point. 
        # pre_cell_walls is not used outside of here, hence does not warrant to be made
        # into a class of its own yet.
        pre_cell_walls = [(c.vertex.interior_end, c.vertex.exterior_start) for c in self.pre_cells]
        pre_cell_walls.append( (self.pre_cells[-1].vertex.interior_start,
                                self.pre_cells[-1].vertex.exterior_end) )
        pre_cell_walls = np.array(pre_cell_walls)[:,:,::2,0]
        _start, _end = pre_cell_walls[:, 0], pre_cell_walls[:, 1] # shape (N, 2), (N, 2)
        _vector = _end - _start # shape (N, 2)
        direction_of_wall = (_vector.T/np.linalg.norm(_vector, axis=-1)).T # shape (N, 2)

        def set_length(i: int, length: float) -> None:
            pre_cell_walls[i][1] = pre_cell_walls[i][0] + direction_of_wall[i] * length

        def get_length(i: int) -> float:
            _start, _end = pre_cell_walls[i]
            return np.linalg.norm(_end - _start, axis=-1)

        def get_all_length() -> List[float]:
            return np.array([get_length(i) for i in range(len(pre_cell_walls))])

        def get_volume_of_cell(i: int) -> float:
            """Return the volume of the cell"""
            outline = np.concatenate([pre_cell_walls[i], pre_cell_walls[i+1][::-1]])
            return polygon_revolve_signed_volume(outline)

        def get_all_volumes() -> List[float]:
            return np.array([get_volume_of_cell(i) for i in range(len(self.pre_cells))])

        if preserve_volume and len(self.pre_cells)>1:
            step_direction = +1 # +1 = adjust leading edge, -1 = adjust lagging edge
            num_passes_counter = -1
            i = 1
            i_range = range(1, len(self.pre_cells))
            forward_pass_result = np.array([0.0 for _ in pre_cell_walls])
            while True:
                target_volumes = self.volumes[i-1:i+1]

                def excess_volume(new_length):
                    set_length(i, new_length)
                    return sum([get_volume_of_cell(i-1) - target_volumes[0],
                                get_volume_of_cell(i) - target_volumes[1]])

                def derivative(new_length):
                    set_length(i, new_length)

                    p1 = pre_cell_walls[i-1][1]
                    p3, p2 = pre_cell_walls[i]
                    d1 = partial_diff_of_volume([p1, p2, p3], direction_of_wall[i])

                    p1, p2 = pre_cell_walls[i]
                    p3 = pre_cell_walls[i+1][1]
                    d2 = partial_diff_of_volume([p1, p2, p3], direction_of_wall[i])

                    return d1+d2

                new_length = Newtons_method(excess_volume, get_length(i), derivative)
                set_length(i, new_length)
                if min(i_range)==max(i_range):
                    break
                elif i==min(i_range):
                    step_direction = +1
                    num_passes_counter += 1
                    backward_pass_result = get_all_length()
                    print("Change in length in the last pass =")
                    print(backward_pass_result - forward_pass_result)
                    if np.allclose(backward_pass_result, forward_pass_result, rtol=0, atol=D_TOLERANCE):
                        print("Success! Terminating cell wall length adjustment.")
                        break
                    print("\nIterating forward...")
                    time.sleep(1)
                elif i==max(i_range):
                    step_direction = -1
                    num_passes_counter += 1
                    forward_pass_result = get_all_length()
                    print("Change in length in the last pass =")
                    print(forward_pass_result - backward_pass_result)
                    print("\nIterating backward...")
                    time.sleep(1)
                print("length = [ "+ ", ".join(f"{l:8.5f}" for l in get_all_length()) + "]")
                print("volume_difference = "+", ".join(f"{l:8.5f}" for l in get_all_volumes()-self.volumes)+"]")
                time.sleep(0.05)
                i += step_direction
        if not all(get_all_volumes()>0):
            raise GeometryError("At least 1 cell has non-positive volume!")
        return pre_cell_walls

    def plot_2d(self) -> None:
        plot_2d([c.outline for c in self.pre_cells])

    def show_cad(self) -> None:
        show_cad([c.half_solid for c in self.pre_cells])

    def __add__(self, other_array: PreCellArray) -> PreCellArray:
        return PreCellArray(self.pre_cells.__add__(other_array.pre_cells))
    def __eq__(self, other_array: PreCellArray) -> bool:
        if isinstance(other_array, PreCellArray):
            return self.pre_cells.__eq__(other_array.pre_cells)
        raise TypeError(f"__eq__ not implemented between {self.__class__} and {other_array.__class__}")
    def __getitem__(self, idx):
        return self.pre_cells.__getitem__(idx)
    def __iter__(self) -> Iterable[PreCell]:
        return self.pre_cells.__iter__()
    def __ne__(self) -> bool:
        if isinstance(other_array, PreCellArray):
            return self.pre_cells.__ne__(other_array.pre_cells)
        raise TypeError(f"__eq__ not implemented between {self.__class__} and {other_array.__class__}")
    def __len__(self) -> int:
        return self.pre_cells.__len__()


# create an xy-plane simply by drawing an L.
x_plane = lambda _x: BluemiraPlane.from_3_points([_x, 0, 0], [_x, 0, 1], [_x, 1, 0])  # noqa: E731
z_plane = lambda _z: BluemiraPlane.from_3_points([0, 0, _z], [0, 1, _z], [1, 0, _z])  # noqa: E731


def fill_xz_to_3d(xz):
    """
    Bloat up a 2D/ a list of 2D coordinates to 3D, by filling out the y-coords with 0's.
    """
    return np.array([xz[0], np.zeros_like(xz[0]), xz[1]])


def cut_exterior_curve(
    interior_panels: npt.NDArray[float],
    exterior_curve: BluemiraWire,
    snap_to_horizontal_angle: float = 30.0,
    starting_cut: Optional[npt.NDArray[float]] = None,
    ending_cut: Optional[npt.NDArray[float]] = None,
    discretization_level: int = 50,
):
    """
    Cut up an exterior boundary (a BluemiraWire) curve according to interior panels'
    breakpoints.

    Parameters
    ----------
        exterior_curve:
            The BluemiraWire representing the outside surface of the vacuum vessel.
        interior_panels:
            numpy array of shape==(N, 2), showing the XZ coordinates of joining points
            between adjacent first wall panels.
        snap_to_horizontal_angle:
            If the cutting plane is less than x degrees (°) away from horizontal,
            then snap it to horizontal.
        starting_cut, ending_cut:
            The program cannot deduce what angle to cut the exterior curve at without
            extra user input. Therefore the user is able to use these options to specify
            the cut line for the first and final point respectively.

            For the first cut line,
                the cut line would start from interior_panels[0] and reach starting_cut,
            and for the final cut line,
                the cut line would start from interior_panels[-1] and reach ending_cut.
            Both arguments have shape (2,) if given,
                as they represent the XZ coordinates.

        discretization_level:
            how many points to use to approximate the curve.
            TODO: remove this! when issue #3038 is fixed. The raw wire can be used
                without discretization then.

    Returns
    -------
    A list of BluemiraWires.
    """
    cut_points = []
    if abs(interior_panels[0][0]) < abs(interior_panels[-1][0]):
        _start_x, _end_x = 0, interior_panels[-1] * 2
    else:
        _start_x, _end_x = interior_panels[0] * 2, 0
    if starting_cut is None:
        starting_cut = np.array([_start_x, interior_panels[0][-1]])
    if ending_cut is None:
        ending_cut = np.array([_end_x, interior_panels[-1][-1]])

    # first point
    p2, p4 = fill_xz_to_3d(interior_panels[0]), fill_xz_to_3d(starting_cut)
    cut_points = [
        get_wire_plane_intersect(
            exterior_curve,
            BluemiraPlane.from_3_points(p2, p4, p2 + np.array([0, 1, 0])),
            cut_direction=p4 - p2,
        )
    ]

    for i in range(1, len(interior_panels) - 1):
        p1, p2, p3 = interior_panels[i - 1 : i + 2]
        origin_2d, direction_2d = get_bisection_line(p1, p2, p3, p2)
        origin, cut_direction = fill_xz_to_3d(origin_2d), fill_xz_to_3d(direction_2d)
        if direction_2d[0]==0:
            _plane = x_plane(p2[0])  # vertical cut line, i.e. surface of a cylinder.
        elif abs(np.arctan(direction_2d[-1] / direction_2d[0]))<np.deg2rad(snap_to_horizontal_angle):
            _plane = z_plane(p2[-1])  # flat cut line, i.e. horizontal plane.
        else:
            _plane = BluemiraPlane.from_3_points(
                origin, origin + cut_direction, origin + np.array([0, 1, 0])
            )  # draw an L
        cut_points.append(
            get_wire_plane_intersect(exterior_curve, _plane, cut_direction=cut_direction)
        )

    # last point
    p2, p4 = fill_xz_to_3d(interior_panels[-1]), fill_xz_to_3d(ending_cut)
    cut_points.append(
        get_wire_plane_intersect(
            exterior_curve,
            BluemiraPlane.from_3_points(p2, p4, p2 + np.array([0, 1, 0])),
            cut_direction=p4 - p2,
        )
    )

    alpha = exterior_curve.parameter_at(cut_points[0])
    wire_segments = []
    for i, cp in enumerate(cut_points[1:]):
        beta = exterior_curve.parameter_at(cp)
        if alpha > beta:
            alpha -= 1.0  # rollover.
        param_range = np.linspace(alpha, beta, discretization_level) % (1.0)
        wire_segments.append(
            make_polygon(
                [exterior_curve.value_at(i) for i in param_range],
                label=f"exterior curve {i + 1}",
                closed=False,
            )
        )
        # `make_polygon` here shall be replaced when issue #3038 gets resolved.
        alpha = beta

    return wire_segments


def grow_blanket_into_cell_array(interior_panels: npt.NDArray[float],
                                thicknesses: TokamakGeometry,
                                in_out_board_breakpoint):
    """
    Simply grow 4 shells around the interior panels according to specified thicknesses.
    The thicknesses of these shells in the inboard side are constant, and the thicknesses
    of these shells in the outboard side are also constant, unlike the ones produced by
    exterior_curve
    """
    return
    

def split_blanket_into_pre_cell_array(
    interior_panels: npt.NDArray[float],
    exterior_curve: BluemiraWire,
    snap_to_horizontal_angle: float = 30.0,
    starting_cut: Optional[npt.NDArray[float]] = None,
    ending_cut: Optional[npt.NDArray[float]] = None,
    discretization_level: int = 50,
):
    """
    Cut up an exterior boundary (a BluemiraWire) curve according to interior panels'
    breakpoints.

    Parameters
    ----------
        exterior_curve:
            The BluemiraWire representing the outside surface of the vacuum vessel.
        interior_panels:
            numpy array of shape==(N, 2), showing the XZ coordinates of joining points
            between adjacent first wall panels.
        snap_to_horizontal_angle:
            If the cutting plane is less than x degrees (°) away from horizontal,
            then snap it to horizontal.
        starting_cut, ending_cut:
            The program cannot deduce what angle to cut the exterior curve at without
            extra user input. Therefore the user is able to use these options to specify
            the cut line for the first and final point respectively.

            For the first cut line,
                the cut line would start from interior_panels[0] and reach starting_cut,
            and for the final cut line,
                the cut line would start from interior_panels[-1] and reach ending_cut.
            Both arguments have shape (2,) if given,
                as they represent the XZ coordinates.

        discretization_level:
            how many points to use to approximate the curve.
            TODO: remove this! See cut_exterior_curve.__doc__

    Returns
    -------
    PreCellArray
    """
    pre_cell_list = []
    for i, exterior_curve_segment in enumerate(
        cut_exterior_curve(
            interior_panels,
            exterior_curve,
            snap_to_horizontal_angle=snap_to_horizontal_angle,
            starting_cut=starting_cut,
            ending_cut=ending_cut,
            discretization_level=discretization_level,
        )
    ):
        pre_cell_list.append(
            PreCell(
                make_polygon(
                    [fill_xz_to_3d(interior_panels[i : i + 2][::-1].T).T], closed=False
                ),
                exterior_curve_segment,
            )
        )
    return PreCellArray(pre_cell_list)

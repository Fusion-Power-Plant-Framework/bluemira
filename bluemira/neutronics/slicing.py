# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Oversees the conversion from bluemira wires into pre-cells, then into csg."""

# ruff: noqa: PLR2004
from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np
from numpy import typing as npt

from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.coordinates import get_bisection_line
from bluemira.geometry.error import GeometryError
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import get_wire_plane_intersect, make_polygon
from bluemira.neutronics.make_pre_cell import (
    DivertorPreCell,
    DivertorPreCellArray,
    PreCell,
    PreCellArray,
)
from bluemira.neutronics.wires import (
    CircleInfo,
    StraightLineInfo,
    WireInfo,
    WireInfoList,
)

if TYPE_CHECKING:
    from bluemira.geometry.wire import BluemiraWire

TOLERANCE_DEGREES = 6.0
DISCRETIZATION_LEVEL = 10


def make_plane_from_2_points(
    point1: npt.NDArray[float], point2: npt.NDArray[float]
) -> BluemiraPlane:
    """Make a plane that is perpendicular to the RZ plane using only 2 points."""
    # Draw an extra leg in the Y-axis direction to make an L shape.
    point3 = point1 + np.array([0, 1, 0])
    return BluemiraPlane.from_3_points(point1, point2, point3)


def x_plane(x: float):
    """Make a vertical plane (perpendicular to Y)."""
    # Simply draw an L shape in the YZ plane
    return BluemiraPlane.from_3_points([x, 0, 0], [x, 0, 1], [x, 1, 0])


def z_plane(z: float):
    """Make a horizontal plane."""
    # Simply draw an L shape in the XY plane
    return BluemiraPlane.from_3_points([0, 0, z], [0, 1, z], [1, 0, z])


def grow_blanket_into_pre_cell_array(
    interior_panels: npt.NDArray[float],  # noqa: ARG001
    inboard_thickness: float,  # noqa: ARG001
    outboard_thickness: float,  # noqa: ARG001
    in_out_board_transition_radius: float,  # noqa: ARG001
):
    """
    Simply grow a shell around the interior panels according to specified thicknesses.
    The thicknesses of these shells in the inboard side are constant, and the thicknesses
    of these shells in the outboard side are also constant. This would be the equivalent
    of making curves that are constant in the inboard and outboard side.
    """
    return


def calculate_plane_dir(
    start_point, end_point
) -> Tuple[BluemiraPlane, npt.NDArray[float]]:
    """
    Calcullate the cutting plane and the direction of the cut from 2 points.
    Both points must lie on the RZ plane.

    Parameters
    ----------
    start_point, end_point:
        3D arrays of single points (shape = (3,))

    Returns
    -------
    plane: BluemiraPlane
        a 3D object
    cut_direction: npt.NDArray[float]
        a 3D array of a single point (shape = (3,))
    """
    plane = make_plane_from_2_points(start_point, end_point)
    cut_direction = end_point - start_point
    return plane, cut_direction


class PanelsAndExteriorCurve:
    """
    A collection of two objects, the first wall panels and the exterior curve.

    Parameters
    ----------
    panel_break_points:
        numpy array of shape==(N, 2), showing the RZ coordinates of joining points
        between adjacent first wall panels, running in the clockwise direction,
        from the inboard divertor to the top, then down to the outboard divertor.
    exterior_curve:
        The BluemiraWire representing the outside surface of the vacuum vessel.
    """

    def __init__(
        self, panel_break_points: npt.NDArray[float], exterior_curve: BluemiraWire
    ):
        """Instantiate from a break point curve and the exterior_curve.

        Parameters
        ----------
        panel_break_points
            A series of 2D coordinate (of shape = (N+1, 2)) representing the N panels
            of the blanket. It (also) runs clockwise (inboard side to outboard side),
            same as exterior_curve
        exterior_curve
            A BluemiraWire that runs clockwise, showing the vacuum vessel on the RHHP
            cross-section of the tokamak.
        """
        self.exterior_curve = exterior_curve
        self.interior_panels = np.insert(
            panel_break_points, 1, 0, axis=-1
        )  # shape = (N+1, 3)
        if len(self.interior_panels[0]) != 3 or np.ndim(self.interior_panels) != 2:
            raise ValueError(
                "Expected an input np.ndarray of breakpoints of shape = "
                f"(N+1, 2). Instead received shape = {np.shape(panel_break_points)}."
            )
        self.cut_points = []
        self.exterior_curve_segments = []

    def get_bisection_line(
        self, index: int
    ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Calculate the bisection line that separates two panels at breakpoint[i].

        Parameters
        ----------
        index: int
            the n-th breakpoint that we want the bisection line to project from.
            (N.B. There are N+1 breakpoints for N panels.)
            Thus this is the last point of the n-th panel.
        """
        p1, p2, p3 = self.interior_panels[index - 1 : index + 2, ::2]
        origin_2d, direction_2d = get_bisection_line(p1, p2, p3, p2)
        line_origin = np.insert(origin_2d, 1, 0, axis=-1)
        line_direction = np.insert(direction_2d, 1, 0, axis=-1)
        return line_origin, line_direction

    def add_cut_point(self, cutting_plane: BluemiraPlane, cut_direction: npt.NDArray):
        """
        Find a point that lies on the exterior curve, which would be used to cut up the
        exterior_curve eventually.

        N.B. These cut points must be sequentially added, i.e. they should
        follow the exterior curve in the clockwise direction.
        """
        self.cut_points.append(
            get_wire_plane_intersect(self.exterior_curve, cutting_plane, cut_direction)
        )

    def calculate_exterior_cut_points(
        self,
        snap_to_horizontal_angle: float,
        starting_cut: Optional[npt.NDArray[float]],
        ending_cut: Optional[npt.NDArray[float]],
    ) -> List[npt.NDArray[float]]:
        """
        Cut the exterior curve according to some specified criteria:
        In general, a line would be drawn from each panel break point outwards towards
        the exterior curve. This would form the cut line.

        The space between two neighbouring cut line makes a PreCell.

        Usually these cut's angle is the bisection of the normal angles of its
        neighbouring panels, but in special rules applies when snap_to_horizontal_angle
        is set to a value >0.

        Since there is only one neighbouring panel at the start break point and end break
        point, starting_cut and end_cut must be specified. If not, they are chosen to be
        horizontal cuts.

        Parameters
        ----------
        snap_to_horizontal_angle:
            If the cutting plane is less than x degrees (°) away from horizontal,
            then snap it to horizontal.
            allowed range: [0, 90]
        starting_cut, ending_cut:
            Each is an ndarray with shape (2,), denoting the destination of the cut
            lines.

            The program cannot deduce what angle to cut the exterior curve at these
            locations without extra user input. Therefore the user is able to use these
            options to specify the cut line's destination point for the first and final
            points respectively.

            For the first cut line,
                the cut line would start from interior_panels[0] and reach starting_cut;
            and for the final cut line,
                the cut line would start from interior_panels[-1] and reach ending_cut.
            Both arguments have shape (2,) if given, representing the RZ coordinates.


        Returns
        -------
        None
            (results are stored in self.cut_points)
        """
        threshold_angle = np.deg2rad(snap_to_horizontal_angle)
        if len(self.cut_points) > 0:
            raise RuntimeError("self.cut_points be cleared first before reuse!")

        # initial cut point

        _plane, _dir = calculate_plane_dir(
            self.interior_panels[0], [starting_cut[0], 0, starting_cut[-1]]
        )
        self.add_cut_point(_plane, _dir)

        for i in range(1, len(self.interior_panels) - 1):
            _origin, _dir = self.get_bisection_line(i)

            if _dir[0] == 0:
                _plane = x_plane(self.interior_panels[i][0])  # vertical cut plane
            elif abs(np.arctan(_dir[-1] / _dir[0])) < threshold_angle:
                _plane = z_plane(self.interior_panels[i][-1])  # horizontal cut plane
            else:
                _plane = make_plane_from_2_points(_origin, _origin + _dir)
            self.add_cut_point(_plane, _dir)

        # final cut point
        _plane, _dir = calculate_plane_dir(
            self.interior_panels[-1], [ending_cut[0], 0, ending_cut[-1]]
        )
        self.add_cut_point(_plane, _dir)

        return self.cut_points

    def execute_exterior_curve_cut(
        self, discretization_level: int, increasing=True
    ) -> List[BluemiraWire]:
        """
        Cut the exterior curve into a series and store them in
        self.exterior_curve_segments.
        This is the slowest part of the entire csg-creation process, because of the
        discretization.

        Parameters
        ----------
        discretization_level:
            how many points to use to approximate the curve.
            TODO: remove this when issue #3038 is fixed. The raw wire can be used
                without discretization then.

        Returns
        -------
        None
            results are stored in self.exterior_curve_segments
        """
        if len(self.exterior_curve_segments) != 0:
            raise RuntimeError("self.exterior_curve_segments must be cleared first!")

        alpha = self.exterior_curve.parameter_at(self.cut_points[0])  # t_start
        for i, cp in enumerate(self.cut_points[1:]):
            beta = self.exterior_curve.parameter_at(cp)  # t_end
            if increasing:
                if alpha > beta:  # alpha is expected to be smaller than beta
                    alpha -= 1.0
            elif alpha < beta:  # alpha is expected to be larger than beta.
                beta -= 1.0
            param_range = np.linspace(alpha, beta, discretization_level) % 1.0
            self.exterior_curve_segments.append(
                make_polygon(
                    np.array([self.exterior_curve.value_at(i) for i in param_range]).T,
                    label=f"exterior curve {i + 1}",
                    closed=False,
                )
            )
            # TODO: `make_polygon` here shall be replaced when issue #3038 gets resolved.
            alpha = beta  # t_end becomes t_start in the next iteration
        return self.exterior_curve_segments

    def make_quadrilateral_pre_cell_array(
        self,
        snap_to_horizontal_angle: float = 30.0,
        starting_cut: Optional[npt.NDArray[float]] = None,
        ending_cut: Optional[npt.NDArray[float]] = None,
        discretization_level: int = DISCRETIZATION_LEVEL,
    ) -> PreCellArray:
        """
        Cut the exterior curve up, so that would act as the exterior side of the
        quadrilateral pre-cell. Then, the panel would act as the interior side of the
        pre-cell. The two remaining sides are the counter-clockwise and clockwise walls.
        """
        _start_r = 0  # starting cut's final destination's r value
        _end_r = (
            self.interior_panels[-1, 0] * 2
        )  # ending cut's final destination's r value

        # make horizontal cuts if the starting and ending cuts aren't provided.
        if starting_cut is None:
            starting_cut = np.array([_start_r, self.interior_panels[0][-1]])
        if ending_cut is None:
            ending_cut = np.array([_end_r, self.interior_panels[-1][-1]])

        self.cut_points = []
        self.calculate_exterior_cut_points(
            snap_to_horizontal_angle, starting_cut, ending_cut
        )

        self.exterior_curve_segments = []
        self.execute_exterior_curve_cut(discretization_level)

        pre_cell_list = []
        for i, exterior_curve_wire_segment in enumerate(self.exterior_curve_segments):
            _inner_wire = make_polygon(
                self.interior_panels[i : i + 2][::-1].T, closed=False
            )
            pre_cell_list.append(PreCell(_inner_wire, exterior_curve_wire_segment))

        return PreCellArray(pre_cell_list)


def check_and_breakdown_bmwire(bmwire: BluemiraWire) -> WireInfoList:
    """
    Raise GeometryError if the BluemiraWire has an unexpected data storage structure.
    Then, get only the key information (start/end points and tangent) of each segment of
    the wire.
    """
    wire_container = []

    def add_line(
        edge: cadapi.apiEdge,
        wire: BluemiraWire,
        start_vector: cadapi.apiVector,
        end_vector: cadapi.apiVector,
    ):
        """Function to record a line"""
        wire_container.append(
            WireInfo(
                StraightLineInfo(start_vector, end_vector),
                [
                    edge.tangentAt(edge.FirstParameter),
                    edge.tangentAt(edge.LastParameter),
                ],
                wire,
            )
        )

    def add_circle(
        edge: cadapi.apiEdge,
        wire: BluemiraWire,
        start_vector: cadapi.apiVector,
        end_vector: cadapi.apiVector,
    ):
        """Function to record the arc of a circle."""
        wire_container.append(
            WireInfo(
                CircleInfo(
                    start_vector,
                    end_vector,
                    np.array(edge.Curve.Center),
                    np.array(edge.Curve.Radius),
                ),
                [
                    edge.tangentAt(edge.FirstParameter),
                    edge.tangentAt(edge.LastParameter),
                ],
                wire,
            )
        )

    for _bmw_edge in bmwire.edges:
        if len(_bmw_edge.boundary) != 1 or len(_bmw_edge.boundary[0].OrderedEdges) != 1:
            raise GeometryError("Expected each boundary to contain only 1 curve!")
        edge = _bmw_edge.boundary[0].OrderedEdges[0]

        # Create aliases for easier referring to variables.
        # The following line may become `edge.start_point(), edge.end_point()`
        # when PR # 3095 is merged
        current_start, current_end = edge.firstVertex().Point, edge.lastVertex().Point
        curve_type = edge.Curve

        # Get the info about this segment of wire
        if isinstance(curve_type, (cadapi.Part.Line, cadapi.Part.LineSegment)):
            add_line(edge, _bmw_edge, current_start, current_end)

        elif isinstance(curve_type, (cadapi.Part.ArcOfCircle, cadapi.Part.Circle)):
            add_circle(edge, _bmw_edge, current_start, current_end)

        elif isinstance(curve_type, (cadapi.Part.BSplineCurve, cadapi.Part.BezierCurve)):
            sample_points = _bmw_edge.discretize(DISCRETIZATION_LEVEL)
            discretized_wire = make_polygon(sample_points, closed=False)
            for __bmw_edge, _start, _end in zip(
                discretized_wire.edges, sample_points.T[:-1], sample_points.T[1:]
            ):
                add_line(
                    __bmw_edge.boundary[0].OrderedEdges[0], __bmw_edge, _start, _end
                )
        elif isinstance(curve_type, (cadapi.Part.ArcOfEllipse, cadapi.Part.Ellipse)):
            raise NotImplementedError("Conversion for ellipses are not available yet.")
            # TODO: implement this feature
        else:
            raise NotImplementedError(f"Conversion for {curve_type} not available yet.")

    return WireInfoList(wire_container)


def turned_morethan_180(
    xyz_vector1: Sequence[float], xyz_vector2: Sequence[float], direction_sign: int
) -> bool:
    """
    Checked if one needs to rotate vector 1 by more than 180° in the specified direction
    by more than 180° to align with vector 2.

    Parameters
    ----------
    xyz_vector1, xyz_vector2: Sequence[float]
        xyz array of where the vector points.
    direction_sign: signed integer
        +1: evaluate rotation required in the counter-clockwise direction.
        -1: evaluate rotation required in the clockwise direction.
    """
    if xyz_vector1[1] != 0 or xyz_vector2[1] != 0:
        raise GeometryError("Tangent vector points out of plane!")
    angle1 = np.arctan2(xyz_vector1[2], xyz_vector1[0])
    angle2 = np.arctan2(xyz_vector2[2], xyz_vector2[0])
    if direction_sign == 1:
        if angle2 < angle1:
            angle2 += 2 * np.pi
        return np.rad2deg(angle2 - angle1) >= 180
    if angle2 > angle1:
        angle2 -= 2 * np.pi
    return np.rad2deg(angle1 - angle2) >= 180


def deviate_less_than(
    xyz_vector1: Sequence[float], xyz_vector2: Sequence[float], threshold_degrees: float
):
    """Check if two vector's angles less than a certain threshold angle (in degrees)."""
    angle1 = np.arctan2(xyz_vector1[2], xyz_vector1[0])
    angle2 = np.arctan2(xyz_vector2[2], xyz_vector2[0])
    return np.rad2deg(abs(angle2 - angle1)) < threshold_degrees


def straight_lines_deviate_less_than(
    info1: WireInfo, info2: WireInfo, threshold_degrees: float
):
    """
    Check that both lines are straight lines, then check if deviation is less than
    threshold_degrees or not
    """
    if not (
        isinstance(info1.key_points, StraightLineInfo)
        and isinstance(info2.key_points, StraightLineInfo)
    ):
        return False
    return deviate_less_than(info1.tangents[1], info2.tangents[0], threshold_degrees)


def break_wire_into_convex_chunks(bmwire, curvature_sign=-1) -> List[WireInfoList]:
    """
    Break a wire up into several convex wires.
    Merge if they are almost collinear.

    Parameters
    ----------
    bmwire: BluemiraWire
    curvature_sign: int, [-1, 1]
        if it's -1: we allow each convex chunk's wire to turn right only.
        if it's 1: we allow each convex chunk's wire to turn left only.

    Returns
    -------
    convex_chunks:
        a list of WireInfos
    """
    wire_segments = list(check_and_breakdown_bmwire(bmwire))
    convex_chunks = []
    # initializing the first chunk
    this_chunk = []
    chunk_start_tangent = wire_segments[0].tangents[0]

    def add_to_chunk(this_seg: WireInfo) -> None:
        """Add the info and wire into the current chunk and current wire."""
        if this_chunk and straight_lines_deviate_less_than(
            this_chunk[-1], this_seg, TOLERANCE_DEGREES
        ):
            # modify the previous line directly
            this_chunk[-1].key_points = StraightLineInfo(
                this_chunk[-1].key_points.start_point, this_seg.key_points.end_point
            )
            return

        this_chunk.append(WireInfo(this_seg.key_points, this_seg.tangents))

    def conclude_chunk():
        """Wrap up the current chunk"""
        nonlocal chunk_start_tangent
        convex_chunks.append(WireInfoList(this_chunk.copy()))
        this_chunk.clear()
        if wire_segments:  # if there are still segments left
            chunk_start_tangent = wire_segments[0].tangents[0]

    while len(wire_segments) > 1:
        add_to_chunk(wire_segments.pop(0))
        prev_end_tangent = this_chunk[-1].tangents[-1]
        next_start_tangent = wire_segments[0].tangents[0]
        if deviate_less_than(
            this_chunk[-1].tangents[1], wire_segments[0].tangents[0], TOLERANCE_DEGREES
        ):
            continue
        if turned_morethan_180(
            chunk_start_tangent, next_start_tangent, curvature_sign
        ) or turned_morethan_180(prev_end_tangent, next_start_tangent, curvature_sign):
            conclude_chunk()
    add_to_chunk(wire_segments.pop(0))
    conclude_chunk()

    return convex_chunks


class DivertorWireAndExteriorCurve:
    """A class to store a wire with an exterior curve"""

    def __init__(self, divertor_wire: BluemiraWire, exterior_curve: BluemiraWire):
        """
        Instantiate from a BluemiraWire of the divertor and a BluemiraWire of the
        exterior (outside of vacuum vessem) curve.

        Parameters
        ----------
        divertor_wire
            A BluemiraWire that runs from the inboard side to the outboard side of the
            tokamak, representing the plasma facing surface of the divertor.
        exterior_curve
            A BluemiraWire that runs clockwise, showing the vacuum vessel on the RHHP
            cross-section of the tokamak.
        """
        convex_segments = break_wire_into_convex_chunks(divertor_wire)
        self.convex_segments = convex_segments
        all_key_points = [
            seg.key_points[0] for seg in chain.from_iterable(convex_segments)
        ]
        all_key_points.append(convex_segments[-1][-1].key_points[1])
        self.key_points = np.array(all_key_points)  # shape = (N+1, 3)
        self.tangents = [
            seg.tangents for seg in chain.from_iterable(convex_segments)
        ]  # shape = (N, 2, 3)
        self.center_point = np.array([  # center of the divertor
            (self.key_points[0, 0] + self.key_points[-1, 0]) / 2,  # mean x
            self.key_points[:, -1].max(),  # highest z
        ])
        self.exterior_curve = exterior_curve
        self.cut_points = []
        self.exterior_curve_segments = []

    def get_bisection_line(
        self, prev_index: int
    ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Get the the line bisecting the x and the y, represented as origin and direction.

        Parameters
        ----------
        prev_index
            two convex chunks are involved when making a bisection line. This index
            refers to the smaller of these two chunks' indices.

        Returns
        -------
        line_origin: a point on this line
        line_direction: a normal vector pointing in the direction of this line.
        """
        # Get the vectors as arrays of shape = (2,)
        direct1 = np.array(self.convex_segments[prev_index][-1].tangents[1])[::2]
        anchor1 = np.array(self.convex_segments[prev_index][-1].key_points[1])[::2]
        direct2 = np.array(self.convex_segments[prev_index + 1][0].tangents[0])[::2]
        anchor2 = np.array(self.convex_segments[prev_index + 1][0].key_points[0])[::2]
        # force the directions to point outwards.
        if (direct1 @ self.center_point) > (direct1 @ anchor1):
            direct1 = -direct1
        if (direct2 @ self.center_point) > (direct2 @ anchor2):
            direct2 = -direct2
        origin_2d, direction_2d = get_bisection_line(
            anchor1 - direct1, anchor1, anchor2 - direct2, anchor2
        )
        line_origin = np.insert(origin_2d, 1, 0, axis=-1)
        line_direction = np.insert(direction_2d, 1, 0, axis=-1)
        return line_origin, line_direction

    def add_cut_point(self, cutting_plane: BluemiraPlane, cut_direction: npt.NDArray):
        """
        Find a point that lies on the exterior curve, which would be used to cut up the
        exterior_curve eventually.

        N.B. These cut points must be sequentially added, i.e. they should
        follow the exterior curve in the clockwise direction.

        TODO: this is identical to :class:`~PanelsAndExteriorCurve.add_cut_point`.
        Should be refactored away.
        """
        self.cut_points.append(
            get_wire_plane_intersect(self.exterior_curve, cutting_plane, cut_direction)
        )

    def calculate_exterior_cut_points(
        self,
        starting_cut: Optional[npt.NDArray[float]],
        ending_cut: Optional[npt.NDArray[float]],
    ) -> List[npt.NDArray[float]]:
        """
        Cut the exterior curve up into N segments to match the N convex chunks of the
        divertor.
        The space between two neighbouring cut line makes a PreCell.

        Parameters
        ----------
        starting_cut, ending_cut:
            Each is an ndarray with shape (2,), denoting the destination of the cut
            lines.

            The program cannot deduce what angle to cut the exterior curve at these
            locations without extra user input. Therefore the user is able to use these
            options to specify the cut line's destination point for the first and final
            points respectively.

            For the first cut line,
                the cut line would start from interior_panels[0] and reach starting_cut;
            and for the final cut line,
                the cut line would start from interior_panels[-1] and reach ending_cut.
            Both arguments have shape (2,) if given, representing the RZ coordinates.


        Returns
        -------
        None
            (results are stored in self.cut_points)
        """
        if len(self.cut_points) > 0:
            raise RuntimeError("self.cut_points be cleared first before reuse!")

        # initial cut point

        _plane, _dir = calculate_plane_dir(
            np.array(self.convex_segments[0][0].key_points[0]),
            [starting_cut[0], 0, starting_cut[-1]],
        )
        self.add_cut_point(_plane, _dir)

        for i in range(len(self.convex_segments) - 1):
            _origin, _dir = self.get_bisection_line(i)
            _plane = make_plane_from_2_points(_origin, _origin + _dir)
            self.add_cut_point(_plane, _dir)

        # final cut point
        _plane, _dir = calculate_plane_dir(
            np.array(self.convex_segments[-1][-1].key_points[1]),
            [ending_cut[0], 0, ending_cut[-1]],
        )
        self.add_cut_point(_plane, _dir)

        return self.cut_points

    def execute_exterior_curve_cut(
        self, discretization_level: int, increasing=False, reverse=True
    ) -> List[WireInfoList]:
        """
        Cut the exterior curve into a series and store them in
        self.exterior_curve_segments. Use the following table to decide whether the
        segments run clockwise or counter-clockwise.
        |                 | increasing = True | increasing = False |
        ------------------------------------------------------------
        | reverse = False |     clockwise     | counter-clockwise  |
        | reverse = False | counter-clockwise |     clockwise      |

        This is the slowest part of the entire csg-creation process, because of the
        discretization.

        TODO: when refactoring: possibly abstract this away because this is a shared
        functionality with :class:`~PanelsAndExteriorCurve.execute_exterior_curve_cut`.

        Parameters
        ----------
        discretization_level:
            how many points to use to approximate the curve.
            TODO: remove this when issue #3038 is fixed. The raw wire can be used
                without discretization then.

        Returns
        -------
        None
            results are stored in self.exterior_curve_segments
        """
        if len(self.exterior_curve_segments) != 0:
            raise RuntimeError("self.exterior_curve_segments must be cleared first!")

        alpha = self.exterior_curve.parameter_at(self.cut_points[0])  # t_start
        for i, cp in enumerate(self.cut_points[1:]):
            beta = self.exterior_curve.parameter_at(cp)  # t_end
            if increasing:
                if alpha > beta:  # alpha is expected to be smaller than beta
                    alpha -= 1.0
            elif alpha < beta:  # alpha is expected to be larger than beta.
                beta -= 1.0
            param_range = np.linspace(alpha, beta, discretization_level) % 1.0
            if reverse:
                param_range = param_range[::-1]

            sample_coords_3d = [self.exterior_curve.value_at(i) for i in param_range]
            this_curve = []
            for start_point, end_point in zip(
                sample_coords_3d[:-1], sample_coords_3d[1:]
            ):
                this_curve.append(WireInfo.from_2P(start_point, end_point))
            self.exterior_curve_segments.append(WireInfoList(this_curve))
            # TODO: `make_polygon` here shall be replaced when issue #3038 gets resolved.
            alpha = beta  # t_end becomes t_start in the next iteration
        return self.exterior_curve_segments

    def make_divertor_pre_cell_array(
        self,
        starting_cut: Optional[npt.NDArray[float]] = None,
        ending_cut: Optional[npt.NDArray[float]] = None,
        discretization_level: int = DISCRETIZATION_LEVEL,
    ):
        """
        Cut the exterior curve up, so that would act as the exterior side of the
        quadrilateral pre-cell. Then, the panel would act as the interior side of the
        pre-cell. The two remaining sides are the counter-clockwise and clockwise walls.

        Ordering: The cells should run from inoboard to outboard.

        Parameters
        ----------
        starting_cut:
            shape = (2,)
        ending_cut:
            shape = (2,)
        discretization_level:
            integer: how many points to use to approximate each exterior curve segment.
        """
        # deduced starting_cut and ending_cut from the divertor itself.
        if not starting_cut:
            first_point = np.array(self.convex_segments[0][0].key_points[0])[::2]
            tangent = np.array(self.convex_segments[0][0].tangents[0])[::2]
            if (tangent @ self.center_point) > (tangent @ first_point):
                tangent = -tangent
            starting_cut = first_point + tangent
        if not ending_cut:
            last_point = np.array(self.convex_segments[-1][-1].key_points[1])[::2]
            tangent = np.array(self.convex_segments[-1][-1].tangents[1])[::2]
            if (tangent @ self.center_point) > (tangent @ last_point):
                tangent = -tangent
            ending_cut = last_point + tangent

        self.cut_points = []
        self.calculate_exterior_cut_points(starting_cut, ending_cut)

        self.exterior_curve_segments = []
        self.execute_exterior_curve_cut(discretization_level)

        pre_cell_list = []
        for i, exterior_wire in enumerate(self.exterior_curve_segments):
            interior_wire = self.convex_segments[i]
            # make a new line joining the start and end.
            if np.isclose(
                exterior_wire.end_point,
                interior_wire.start_point,
                rtol=0,
                atol=EPS_FREECAD,
            ).all():
                cw_line = WireInfoList([interior_wire.pop(0)])
            else:
                cw_line = WireInfoList([
                    WireInfo.from_2P(exterior_wire.end_point, interior_wire.start_point)
                ])
                # merge lines if collinear
            while straight_lines_deviate_less_than(cw_line[-1], interior_wire[0], 0.5):
                cw_line.end_point = interior_wire.start_point
                interior_wire.pop(0)

            if np.isclose(
                interior_wire.end_point,
                exterior_wire.start_point,
                rtol=0,
                atol=EPS_FREECAD,
            ).all():
                ccw_line = WireInfoList([interior_wire.pop(-1)])
            else:
                ccw_line = WireInfoList([
                    WireInfo.from_2P(interior_wire.end_point, exterior_wire.start_point)
                ])
                # merge lines if collinear
            while straight_lines_deviate_less_than(interior_wire[-1], ccw_line[0], 0.5):
                ccw_line.start_point = interior_wire.end_point
                interior_wire.pop(-1)

            pre_cell_list.append(DivertorPreCell(interior_wire, exterior_wire))

        return DivertorPreCellArray(pre_cell_list)

# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Oversees the conversion from bluemira wires into pre-cells, then into csg."""

# ruff: noqa: PLR2004
from __future__ import annotations

from itertools import chain, pairwise, starmap
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.coordinates import choose_direction, get_bisection_line
from bluemira.geometry.error import GeometryError
from bluemira.geometry.plane import (
    BluemiraPlane,
    calculate_plane_dir,
    x_plane,
    xz_plane_from_2_points,
    z_plane,
)
from bluemira.geometry.tools import get_wire_plane_intersect, make_polygon
from bluemira.neutronics.constants import DISCRETISATION_LEVEL, TOLERANCE_DEGREES
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
    from collections.abc import Generator, Sequence

    from numpy import typing as npt

    from bluemira.geometry.wire import BluemiraWire


def cut_curve(
    cut_points: list[npt.NDArray],
    wire: BluemiraWire,
    discretisation_level: int,
    *,
    reverse: bool = False,
) -> Generator[npt.NDArray[float]]:
    """
    Generator to cut a closed BluemiraWire to size.
    Current implementation is to yield a parameter range for every sequential cut-point
    pair. Subject to change of implementation in the future when issue 3038 is fixed.

    Parameters
    ----------
    cut_points
        A list of at least 3 points, presumed to be lying on the wire, that we want to
        cut the wire at.

    wire
        The wire that we want to cut up.

    discretisation_level [current implementation]
        We yield a list of points of len==discretisation_level, such that we can build a
        wire made of (discretisation_level-1) straight lines to approximate each segment.

    reversed:
        Whether we want the neutron spectrum to go in the increasing or the decreasing
        direction.

    Yields
    ------
    params_range [current implementation]
        For each segment, yield the values of the parametric variable t such that
        `[wire.value_at(t) for t in params_range]`
        is a list of points (sufficient to built up a series of (discretisation_level-1)
        straight lines that approximate that segment)
    """
    cut_params = [wire.parameter_at(cp) for cp in cut_points]
    # for reference: cut_params has range:
    # [0, 1] for an open wire, [0, 1) for a closed wire.

    # determine whether we're going up or down in parameter (t) space.
    finite_difference = np.diff(cut_params)
    if len(finite_difference) <= 2:
        raise GeometryError(
            "Too few points! I.e. discretization_level parameter too low. "
            "Can't determine the cut direction!"
        )
    if (finite_difference <= 0).sum() <= 1:
        # strictly monotonically increasing except for 1 wrap-around point
        increasing = True
    elif (finite_difference >= 0).sum() <= 1:
        # strictly monotonically decreasing except for 1 wrap-around point
        increasing = False
    else:  # no discrenable pattern in the increase/decrease
        raise GeometryError("Points are too disordered!")

    # generator function
    for alpha, beta in pairwise(cut_params):
        if increasing:
            if alpha > beta:  # alpha is expected to be smaller than beta
                alpha -= 1.0  # noqa: PLW2901
        elif alpha < beta:  # alpha is expected to be larger than beta.
            alpha += 1.0  # noqa: PLW2901
        param_range = np.linspace(alpha, beta, discretisation_level) % 1.0
        if reverse:
            param_range = param_range[::-1]
        yield param_range


class PanelsAndExteriorCurve:
    """
    A collection of three objects, the first wall panels, the vacuum vessel interior
    curve, and the exterior curve.

    Parameters
    ----------
    panel_break_points:
        numpy array of shape==(N, 2), showing the RZ coordinates of joining points
        between adjacent first wall panels, running in the clockwise direction,
        from the inboard divertor to the top, then down to the outboard divertor.
    vv_interior:
        The BluemiraWire representing the inside surface of the vacuum vessel.
    vv_exterior:
        The BluemiraWire representing the outside surface of the vacuum vessel.
    """

    def __init__(
        self,
        panel_break_points: npt.NDArray[float],
        vv_interior: BluemiraWire,
        vv_exterior: BluemiraWire,
    ):
        """Instantiate from a break point curve and the vv_interior.

        Parameters
        ----------
        panel_break_points
            A series of 2D coordinate (of shape = (N+1, 2)) representing the N panels
            of the blanket. It (also) runs clockwise (inboard side to outboard side),
            same as vv_interior
        vv_interior
            A BluemiraWire that runs clockwise, showing the vacuum vessel on the RHHP
            cross-section of the tokamak.
        """
        self.vv_interior = vv_interior
        self.vv_exterior = vv_exterior
        self.interior_panels = np.insert(
            panel_break_points, 1, 0, axis=-1
        )  # shape = (N+1, 3)
        if len(self.interior_panels[0]) != 3 or np.ndim(self.interior_panels) != 2:
            raise ValueError(
                "Expected an input np.ndarray of breakpoints of shape = "
                f"(N+1, 2). Instead received shape = {np.shape(panel_break_points)}."
            )

    def get_bisection_line(
        self, index: int
    ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
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

    def add_cut_points(self, cutting_plane: BluemiraPlane, cut_direction: npt.NDArray):
        """
        Find where the cutting_plane intersect the self.vv_interior and self.vv_exterior;
        these points will eventually be used to cut up self.vv_interior and
        self.vv_exterior.

        N.B. These cut points must be sequentially added, i.e. they should
        follow the curves in the clockwise direction.
        """
        self.vv_cut_points.append(
            get_wire_plane_intersect(self.vv_interior, cutting_plane, cut_direction)
        )
        self.exterior_cut_points.append(
            get_wire_plane_intersect(self.vv_exterior, cutting_plane, cut_direction)
        )

    def calculate_cut_points(
        self,
        starting_cut: npt.NDArray[float] | None,
        ending_cut: npt.NDArray[float] | None,
        snap_to_horizontal_angle: float,
    ) -> list[npt.NDArray[float]]:
        """
        Cut the curves according to some specified criteria:
        In general, a line would be drawn from each panel break point outwards towards
        the curves. This would form the cut line.

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

            The program cannot deduce what angle to cut the curves at these
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
        self.vv_cut_points

            cut points where the vacuum vessel interior wire is split

        self.exterior_cut_points

            cut points where the vacuum vessel exterior wire is split
        """
        self.vv_cut_points, self.exterior_cut_points = [], []

        threshold_angle = np.deg2rad(snap_to_horizontal_angle)

        # initial cut point

        _plane, _dir = calculate_plane_dir(
            self.interior_panels[0], [starting_cut[0], 0, starting_cut[-1]]
        )
        self.add_cut_points(_plane, _dir)

        for i in range(1, len(self.interior_panels) - 1):
            _origin, _dir = self.get_bisection_line(i)

            if _dir[0] == 0:
                _plane = x_plane(self.interior_panels[i][0])  # vertical cut plane
            elif abs(np.arctan(_dir[-1] / _dir[0])) < threshold_angle:
                _plane = z_plane(self.interior_panels[i][-1])  # horizontal cut plane
            else:
                _plane = xz_plane_from_2_points(_origin, _origin + _dir)
            self.add_cut_points(_plane, _dir)

        # final cut point
        _plane, _dir = calculate_plane_dir(
            self.interior_panels[-1], [ending_cut[0], 0, ending_cut[-1]]
        )
        self.add_cut_points(_plane, _dir)

        return self.vv_cut_points, self.exterior_cut_points

    def execute_curve_cut(
        self,
        discretisation_level: int,
        starting_cut: npt.NDArray[float] | None = None,
        ending_cut: npt.NDArray[float] | None = None,
        snap_to_horizontal_angle: float = 30.0,
    ) -> list[BluemiraWire]:
        """
        Cut the exterior curve into a series and return them.
        This is the slowest part of the entire csg-creation process, because of the
        discretisation.

        Parameters
        ----------
        snap_to_horizontal_angle, starting_cut, ending_cut
            See
            :meth:`~bluemira.neutronics.slicing.PanelsAndExteriorCurve.calculate_cut_points`

        discretisation_level:
            how many points to use to approximate the curve.
            TODO: remove this when issue #3038 is fixed. The raw wire can be used
                without discretisation then.

        Returns
        -------
        vv_curve_segments

            segments of the vacuum vessl interior curve forming each pre-cell's interior
            curve.

        exterior_curve_segments

            segments of the vacuum vessl exterior curve forming each pre-cell's exterior
            curve.
        """
        vv_curve_segments, exterior_curve_segments = [], []
        vv_cut_points, exterior_cut_points = self.calculate_cut_points(
            starting_cut, ending_cut, snap_to_horizontal_angle
        )

        for i, t_range in enumerate(
            cut_curve(
                vv_cut_points, self.vv_interior, discretisation_level, reverse=False
            )
        ):
            vv_curve_segments.append(
                self.approximate_curve(
                    self.vv_interior,
                    t_range,
                    label=f"vacuum vessel interior curve {i + 1}",
                )
            )

        for i, t_range in enumerate(
            cut_curve(
                exterior_cut_points,
                self.vv_exterior,
                discretisation_level,
                reverse=False,
            )
        ):
            exterior_curve_segments.append(
                self.approximate_curve(
                    self.vv_exterior,
                    t_range,
                    label=f"vacuum vessel exterior curve {i + 1}",
                )
            )

        return vv_curve_segments, exterior_curve_segments

    @staticmethod
    def approximate_curve(
        curve: BluemiraWire, param_range: npt.NDArray[float], label=""
    ) -> BluemiraWire:
        """
        Given params_range (an iterable of floats of len n between 0 to 1), create n-1
        straight lines such that the n-th point corresponds to
        `curve.value_at(params_range[n])`.

        This implementation shall be replaced when issue #3038 gets resolved.
        """
        return make_polygon(
            np.array([curve.value_at(t) for t in param_range]).T,
            label=label,
            closed=False,  # we expect it to form an open curve.
        )

    def make_quadrilateral_pre_cell_array(
        self,
        discretisation_level: int = DISCRETISATION_LEVEL,
        starting_cut: npt.NDArray[float] | None = None,
        ending_cut: npt.NDArray[float] | None = None,
        snap_to_horizontal_angle: float = 30.0,
    ) -> PreCellArray:
        """
        Cut the exterior curve up, so that it would act as the exterior side of the
        quadrilateral pre-cell. Then, the panel would act as the interior side of the
        pre-cell. The two remaining sides are the counter-clockwise and clockwise walls.

        The vacuum vessel interior surface is sandwiched between them.
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

        pre_cell_list = []
        for i, (vv_segment, exterior_segment) in enumerate(
            zip(
                *self.execute_curve_cut(
                    discretisation_level,
                    starting_cut,
                    ending_cut,
                    snap_to_horizontal_angle,
                ),
                strict=True,
            )
        ):
            _inner_wire = make_polygon(
                self.interior_panels[i : i + 2][::-1].T, closed=False
            )
            pre_cell_list.append(PreCell(_inner_wire, vv_segment, exterior_segment))

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
                    edge.Curve.Radius,
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
        if isinstance(curve_type, cadapi.Part.Line | cadapi.Part.LineSegment):
            add_line(edge, _bmw_edge, current_start, current_end)

        elif isinstance(curve_type, cadapi.Part.ArcOfCircle | cadapi.Part.Circle):
            add_circle(edge, _bmw_edge, current_start, current_end)

        elif isinstance(curve_type, cadapi.Part.BSplineCurve | cadapi.Part.BezierCurve):
            sample_points = _bmw_edge.discretise(DISCRETISATION_LEVEL)
            discretised_wire = make_polygon(sample_points, closed=False)
            for __bmw_edge, _start, _end in zip(
                discretised_wire.edges,
                sample_points.T[:-1],
                sample_points.T[1:],
                strict=True,
            ):
                add_line(
                    __bmw_edge.boundary[0].OrderedEdges[0], __bmw_edge, _start, _end
                )
        elif isinstance(curve_type, cadapi.Part.ArcOfEllipse | cadapi.Part.Ellipse):
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


def break_wire_into_convex_chunks(bmwire, curvature_sign=-1) -> list[WireInfoList]:
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
        interior_curve_turned_over_180 = turned_morethan_180(
            chunk_start_tangent, next_start_tangent, curvature_sign
        )
        concave_turning_point = turned_morethan_180(
            prev_end_tangent, next_start_tangent, curvature_sign
        )
        if concave_turning_point:
            conclude_chunk()
            continue
        if interior_curve_turned_over_180:
            # curled in on itself too much.
            bluemira_warn(
                "Divertor wire geometry possibly too extreme for program "
                "to handle. Check pre-cell visually by using the .plot_2d() methods "
                "on the relevant DivertorPreCell and DivertorPreCellArray."
            )
    add_to_chunk(wire_segments.pop(0))
    conclude_chunk()

    return convex_chunks


class DivertorWireAndExteriorCurve:
    """
    A class to store a wire with a vacuum vessel interior curve and an exterior curve.
    """

    def __init__(
        self,
        divertor_wire: BluemiraWire,
        vv_interior: BluemiraWire,
        vv_exterior: BluemiraWire,
    ):
        """
        Instantiate from a BluemiraWire of the divertor and a BluemiraWire of the
        exterior (outside of vacuum vessem) curve. Also save the BluemiraWire that goes
        between these two (inside of vacuum vessel).

        Parameters
        ----------
        divertor_wire
            A BluemiraWire that runs from the inboard side to the outboard side of the
            tokamak, representing the plasma facing surface of the divertor.
        vv_interior
            A BluemiraWire that runs clockwise, showing the inside of the vacuum vessel
            on the RHHP cross-section of the tokamak.
        vv_exterior
            A BluemiraWire that runs clockwise, showing the outside of the vacuum vessel
            on the RHHP cross-section of the tokamak.
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
        self.vv_interior = vv_interior
        self.vv_exterior = vv_exterior

    def get_bisection_line(
        self, prev_index: int
    ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
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
        direct1 = choose_direction(direct1, self.center_point, anchor1)
        direct2 = choose_direction(direct2, self.center_point, anchor2)
        origin_2d, direction_2d = get_bisection_line(
            anchor1 - direct1, anchor1, anchor2 - direct2, anchor2
        )
        line_origin = np.insert(origin_2d, 1, 0, axis=-1)
        line_direction = np.insert(direction_2d, 1, 0, axis=-1)
        return line_origin, line_direction

    def add_cut_points(self, cutting_plane: BluemiraPlane, cut_direction: npt.NDArray):
        """
        Find where the cutting_plane intersect the self.vv_interior and self.vv_exterior;
        these points will eventually be used to cut up self.vv_interior and
        self.vv_exterior.

        N.B. These cut points must be sequentially added, i.e. they should
        follow the curves in the clockwise direction.

        While identical to :meth:`~PanelsAndExteriorCurve.add_cut_points`, this can't be
        refactored away because they're specific to the class.
        """
        self.vv_cut_points.append(
            get_wire_plane_intersect(self.vv_interior, cutting_plane, cut_direction)
        )
        self.exterior_cut_points.append(
            get_wire_plane_intersect(self.vv_exterior, cutting_plane, cut_direction)
        )

    def calculate_cut_points(
        self,
        starting_cut: npt.NDArray[float] | None,
        ending_cut: npt.NDArray[float] | None,
    ) -> list[npt.NDArray[float]]:
        """
        Cut the curves up into N segments to match the N convex chunks of the
        divertor.
        The space between two neighbouring cut line makes a DivertorPreCell.

        Parameters
        ----------
        starting_cut, ending_cut:
            Each is an ndarray with shape (2,), denoting the destination of the cut
            lines.

            The program cannot deduce what angle to cut the curves at these
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
        self.vv_cut_points

            cut points where the vacuum vessel interior wire is split

        self.exterior_cut_points

            cut points where the vacuum vessel exterior wire is split
        """
        self.vv_cut_points, self.exterior_cut_points = [], []

        # initial cut point

        _plane, _dir = calculate_plane_dir(
            np.array(self.convex_segments[0][0].key_points[0]),
            [starting_cut[0], 0, starting_cut[-1]],
        )
        self.add_cut_points(_plane, _dir)

        for i in range(len(self.convex_segments) - 1):
            _origin, _dir = self.get_bisection_line(i)
            _plane = xz_plane_from_2_points(_origin, _origin + _dir)
            self.add_cut_points(_plane, _dir)

        # final cut point
        _plane, _dir = calculate_plane_dir(
            np.array(self.convex_segments[-1][-1].key_points[1]),
            [ending_cut[0], 0, ending_cut[-1]],
        )
        self.add_cut_points(_plane, _dir)

        return self.vv_cut_points, self.exterior_cut_points

    def execute_curve_cut(
        self,
        discretisation_level: int,
        starting_cut: npt.NDArray[float] | None,
        ending_cut: npt.NDArray[float] | None,
    ) -> list[WireInfoList]:
        """
        Cut the vacuum vessel curves into a series return these segments.

        (Obsolete)
        ----------
        Use the following table to decide whether the
        segments run clockwise or counter-clockwise.
        |                 | INCREASING = True | INCREASING = False |
        ------------------------------------------------------------
        | reverse = False |     clockwise     | counter-clockwise  |
        | reverse = False | counter-clockwise |     clockwise      |

        This is the slowest part of the entire csg-creation process, because of the
        discretisation.

        Parameters
        ----------
        starting_cut, ending_cut:
            See
            :meth:`~bluemira.neutronics.slicing.DivertorWireAndExteriorCurve.calculate_cut_points`
        discretisation_level:
            how many points to use to approximate the curve.
            TODO: remove this when issue #3038 is fixed. The raw wire can be used
                without discretisation then.

        Returns
        -------
        vv_curve_segments

            list of WireInfoList describing the each pre-cell's vacuum vessel interior
            curve

        exterior_curve_segments

            list of WireInfoList describing the each pre-cell's vacuum vessel exterior
            curve
        """
        vv_curve_segments, exterior_curve_segments = [], []
        vv_cut_points, exterior_cut_points = self.calculate_cut_points(
            starting_cut, ending_cut
        )

        for t_range in cut_curve(
            vv_cut_points, self.vv_interior, discretisation_level, reverse=True
        ):
            vv_curve_segments.append(self.approximate_curve(self.vv_interior, t_range))  # noqa: PERF401

        for t_range in cut_curve(
            exterior_cut_points, self.vv_exterior, discretisation_level, reverse=True
        ):
            exterior_curve_segments.append(  # noqa: PERF401
                self.approximate_curve(self.vv_exterior, t_range)
            )

        return vv_curve_segments, exterior_curve_segments

    @staticmethod
    def approximate_curve(
        curve: BluemiraWire, param_range: npt.NDArray[float]
    ) -> WireInfoList:
        """
        Given params_range (an iterable of floats of len n between 0 to 1), create n-1
        straight lines such that the n-th point corresponds to
        `curve.value_at(params_range[n])`.

        This implementation shall be updated/replaced when issue #3038 gets resolved.
        """
        sample_coords_3d = [curve.value_at(t) for t in param_range]
        this_curve = list(starmap(WireInfo.from_2P, pairwise(sample_coords_3d)))
        return WireInfoList(this_curve)

    def make_divertor_pre_cell_array(
        self,
        starting_cut: npt.NDArray[float] | None = None,
        ending_cut: npt.NDArray[float] | None = None,
        discretisation_level: int = DISCRETISATION_LEVEL,
    ):
        """
        Cut the exterior curve up, so that would act as the exterior side of the
        quadrilateral pre-cell. Then, the panel would act as the interior side of the
        pre-cell. The two remaining sides are the counter-clockwise and clockwise walls.
        The vv_interior curve would be sandwiched between these two.

        Ordering: The cells should run from inoboard to outboard.

        Parameters
        ----------
        starting_cut:
            shape = (2,)
        ending_cut:
            shape = (2,)
        discretisation_level:
            integer: how many points to use to approximate each curve segment.
        """
        # deduced starting_cut and ending_cut from the divertor itself.
        if not starting_cut:
            first_point = np.array(self.convex_segments[0][0].key_points[0])[::2]
            tangent = np.array(self.convex_segments[0][0].tangents[0])[::2]
            tangent = choose_direction(tangent, self.center_point, first_point)
            starting_cut = first_point + tangent
        if not ending_cut:
            last_point = np.array(self.convex_segments[-1][-1].key_points[1])[::2]
            tangent = np.array(self.convex_segments[-1][-1].tangents[1])[::2]
            tangent = choose_direction(tangent, self.center_point, last_point)
            ending_cut = last_point + tangent

        pre_cell_list = []
        for i, (vv_segment, exterior_segment) in enumerate(
            zip(
                *self.execute_curve_cut(discretisation_level, starting_cut, ending_cut),
                strict=True,
            )
        ):
            interior_wire = self.convex_segments[i]
            # make a new line joining the start and end.
            if np.isclose(
                exterior_segment.end_point,
                interior_wire.start_point,
                rtol=0,
                atol=EPS_FREECAD,
            ).all():
                cw_line = WireInfoList([interior_wire.pop(0)])
            else:
                cw_line = WireInfoList([
                    WireInfo.from_2P(
                        exterior_segment.end_point, interior_wire.start_point
                    )
                ])
                # merge lines if collinear
            while straight_lines_deviate_less_than(cw_line[-1], interior_wire[0], 0.5):
                cw_line.end_point = interior_wire.start_point
                interior_wire.pop(0)

            if np.isclose(
                interior_wire.end_point,
                exterior_segment.start_point,
                rtol=0,
                atol=EPS_FREECAD,
            ).all():
                ccw_line = WireInfoList([interior_wire.pop(-1)])
            else:
                ccw_line = WireInfoList([
                    WireInfo.from_2P(
                        interior_wire.end_point, exterior_segment.start_point
                    )
                ])
                # merge lines if collinear
            while straight_lines_deviate_less_than(interior_wire[-1], ccw_line[0], 0.5):
                ccw_line.start_point = interior_wire.end_point
                interior_wire.pop(-1)

            pre_cell_list.append(
                DivertorPreCell(interior_wire, vv_segment, exterior_segment)
            )

        return DivertorPreCellArray(pre_cell_list)
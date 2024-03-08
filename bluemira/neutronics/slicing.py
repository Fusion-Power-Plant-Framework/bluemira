# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Oversees the conversion from bluemira wires into pre-cells, then into csg."""

# ruff: noqa: PLR2004
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy import typing as npt

from bluemira.geometry.coordinates import get_bisection_line
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import get_wire_plane_intersect, make_polygon
from bluemira.neutronics.make_pre_cell import PreCell, PreCellArray

if TYPE_CHECKING:
    from bluemira.geometry.wire import BluemiraWire


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
        """Instantiate from a break curve and a thing."""
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

    @staticmethod
    def calculate_plane_dir(
        start_point, end_point
    ) -> Tuple[BluemiraPlane, npt.NDArray[float]]:
        """
        Calcullate the cutting plane and the direction of the cut from 2 points.
        Both points must lie on the RZ plane.
        """
        plane = make_plane_from_2_points(start_point, end_point)
        cut_direction = end_point - start_point
        return plane, cut_direction

    def get_bisection_line(self, index):
        """Calculate the bisection line that separates two panels at breakpoint[i]."""
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
    ) -> PreCellArray:
        """
        Cut the exterior curve according to some specified criteria:
        In general, a line would be drawn from each panel break point outwards towards
        the exterior curve. This would form the cut line.

        The space between two neighbouring cut line makes a PreCell.

        Usually these cut's angle is the bisection of the normal angles of its
        neighbouring panels, but in special rules applies when snap_to_horizontal_angle
        is set to >0.

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
            raise RuntimeError("self.cut_points be cleared first!")

        # initial cut point

        _plane, _dir = self.calculate_plane_dir(
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
        _plane, _dir = self.calculate_plane_dir(
            self.interior_panels[-1], [ending_cut[0], 0, ending_cut[-1]]
        )
        self.add_cut_point(_plane, _dir)

        return self.cut_points

    def execute_exterior_curve_cut(self, discretization_level: int):
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

        alpha = self.exterior_curve.parameter_at(self.cut_points[0])  # t-start
        for i, cp in enumerate(self.cut_points[1:]):
            beta = self.exterior_curve.parameter_at(cp)  # t-end
            if alpha > beta:
                alpha -= 1.0
            param_range = np.linspace(alpha, beta, discretization_level) % 1.0
            self.exterior_curve_segments.append(
                make_polygon(
                    np.array([self.exterior_curve.value_at(i) for i in param_range]).T,
                    label=f"exterior curve {i + 1}",
                    closed=False,
                )
            )
            # `make_polygon` here shall be replaced when issue #3038 gets resolved.
            alpha = beta  # t-end becomes t-start in the next iteration
        return self.exterior_curve_segments

    def make_quadrilateral_pre_cell_array(
        self,
        snap_to_horizontal_angle: float = 30.0,
        starting_cut: Optional[npt.NDArray[float]] = None,
        ending_cut: Optional[npt.NDArray[float]] = None,
        discretization_level: int = 10,
    ):
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

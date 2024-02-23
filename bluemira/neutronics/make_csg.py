# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Create csg geometry from bluemira wires."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy import typing as npt

from bluemira.base.constants import EPS
from bluemira.display import plot_2d
from bluemira.geometry.coordinates import Coordinates, get_bisection_line
from bluemira.geometry.error import GeometryError
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import get_wire_plane_intersect, make_polygon, revolve_shape
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
                of a blanket's precell, running in the anti-clockwise direction when
                viewing the right hand side poloidal cross-section,
                i.e. downwards if inboard, upwards if outboard.
            or: a single Coordinates point, representing a point on the interior-boundary

        exterior_wire

            A wire representing the exterior-boundary (i.e. air-facing side) of a
                blanket's precell, running in the clockwise direction when viewing the
                right hand side poloidal cross-section,
                i.e. upwards if inboard, downwards if outboard.
        """
        self._interior_wire = interior_wire
        self._exterior_wire = exterior_wire
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
                self._interior_wire,
                self._in2out,
            ])
        self.outline = BluemiraWire([self._exterior_wire, self._inner_curve])
        self.cross_section = BluemiraSolid(revolve_shape(self.outline))
        self.volume = self.cross_section.volume

    def plot_2d(self, *args, **kwargs):
        """Plot the outline in 2D"""
        plot_2d(self.outline, *args, **kwargs)

    def finer_division(self, bending_limit):
        """Further divide up the cell's exterior wire experiences more bending than
        a certain limit.
        """
        new_wire_list = list(...)
        return [PreCell(new_wire) for new_wire in new_wire_list]


@dataclass
class WallThicknessFraction:
    """List of thickness of various sections of the blanket as fractions"""

    first_wall: float  # [m]
    breeder_zone: float  # [m]
    manifold: float  # [m]
    vacuum_vessel: float  # [m]

    def __post_init__(self):
        """Check fractions are between 0 and 1 and sums to unity."""
        for section in ("first_wall", "manifold", "vacuum_vessel"):
            if getattr(self, section) <= 0:
                raise GeometryError(f"Thickness fraction of {section} must be non-zero")
        if self.breeder_zone < 0:  # can be zero, but not negative.
            raise GeometryError("Thickness fraction of breeder_zone must be nonnegative")
        if not np.isclose(
            sum([self.first_wall, self.manifold, self.breeder_zone, self.vacuum_vessel]),
            1.0,
            rtol=0,
            atol=EPS,
        ):
            raise GeometryError(
                "Thickness fractions of all four sections " "must add up to unity!"
            )


class InboardThicknessFraction(WallThicknessFraction):
    """Thickness fraction list of the inboard wall of the blanket"""

    pass


class OutboardThicknessFraction(WallThicknessFraction):
    """Thickness fraction list of the outboard wall of the blanket"""

    pass


@dataclass
class ThicknessFractions:
    """
    A dataclass containing info. on both
    inboard and outboard blanket thicknesses as fractions.
    """

    inboard: InboardThicknessFraction
    outboard: OutboardThicknessFraction

    @classmethod
    def from_TokamakGeometry(cls, tokamak_geometry: TokamakGeometry):
        """
        Create this dataclass by
        translating from our existing tokamak_geometry dataclass.
        """
        inb_sum = sum([
            tokamak_geometry.inb_fw_thick,
            tokamak_geometry.inb_bz_thick,
            tokamak_geometry.inb_mnfld_thick,
            tokamak_geometry.inb_vv_thick,
        ])
        inb = InboardThicknessFraction(
            tokamak_geometry.inb_fw_thick / inb_sum,
            tokamak_geometry.inb_bz_thick / inb_sum,
            tokamak_geometry.inb_mnfld_thick / inb_sum,
            tokamak_geometry.inb_vv_thick / inb_sum,
        )
        outb_sum = sum([
            tokamak_geometry.outb_fw_thick,
            tokamak_geometry.outb_bz_thick,
            tokamak_geometry.outb_mnfld_thick,
            tokamak_geometry.outb_vv_thick,
        ])
        outb = OutboardThicknessFraction(
            tokamak_geometry.outb_fw_thick / outb_sum,
            tokamak_geometry.outb_bz_thick / outb_sum,
            tokamak_geometry.outb_mnfld_thick / outb_sum,
            tokamak_geometry.outb_vv_thick / outb_sum,
        )
        return cls(inb, outb)


# create an xy-plane simply by drawing an L.
z_plane = lambda _z: BluemiraPlane.from_3_points([0, 0, _z], [0, 1, _z], [1, 0, _z])  # noqa: E731


def fill_xz_to_3d(xz):
    """
    Bloat up a 2D/ a list of 2D coordinates to 3D, by filling out the y-coords with 0's.
    """
    return np.array([xz[0], np.zeros_like(xz[0]), xz[1]])


def cut_exterior_curve(
    exterior_curve: BluemiraWire,
    interior_panels: npt.NDArray[float],
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
        angle = np.rad2deg(np.arctan(direction_2d[-1] / direction_2d[0]))
        if any(
            abs(np.array([angle - 180, angle, angle + 180])) < snap_to_horizontal_angle
        ):
            _plane = z_plane(p2[-1])
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


def split_blanket_into_precells(
    exterior_curve: BluemiraWire,
    interior_panels: npt.NDArray[float],
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
    A list of BluemiraWires.
    """
    precell_list = []
    for i, exterior_curve_segment in enumerate(
        cut_exterior_curve(
            exterior_curve,
            interior_panels,
            snap_to_horizontal_angle=snap_to_horizontal_angle,
            starting_cut=starting_cut,
            ending_cut=ending_cut,
            discretization_level=discretization_level,
        )
    ):
        precell_list.append(
            PreCell(
                make_polygon(
                    [fill_xz_to_3d(interior_panels[i : i + 2][::-1].T).T], closed=False
                ),
                exterior_curve_segment,
            )
        )
    return precell_list

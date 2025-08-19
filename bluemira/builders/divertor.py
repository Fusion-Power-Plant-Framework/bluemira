# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Builder for making a parameterised EU-DEMO divertor.
"""

from __future__ import annotations

import enum
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import (
    apply_component_display_options,
    circular_pattern_component,
    get_n_sectors,
    pattern_revolved_silhouette,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.equilibria.find import find_flux_surface_through_point
from bluemira.equilibria.find_legs import LegFlux
from bluemira.geometry.tools import (
    interpolate_bspline,
    make_circle,
    make_polygon,
)
from bluemira.geometry.wire import BluemiraWire

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from bluemira.base.builder import BuildConfig
    from bluemira.base.parameter_frame.typed import ParameterFrameLike
    from bluemira.equilibria import Equilibrium
    from bluemira.geometry.face import BluemiraFace


class LegPosition(enum.Enum):
    """
    Enum classifying divertor/separatrix leg positions
    """

    INNER = enum.auto()
    OUTER = enum.auto()


def get_separatrix_legs(
    equilibrium: Equilibrium,
) -> dict[LegPosition, list[BluemiraWire]]:
    """
    Find the separatrix legs for the given equilibrium.

    Returns
    -------
    :
        Legs for given position on separatrix
    """
    # A flag specifying which end of the plasma (i.e., upper or lower)
    # we want the legs from will need to be added
    legs = LegFlux(equilibrium).get_legs()
    return {
        LegPosition.INNER: [make_polygon(loop.xyz) for loop in legs["lower_inner"]],
        LegPosition.OUTER: [make_polygon(loop.xyz) for loop in legs["lower_outer"]],
    }


class WireEndAxis(enum.Enum):
    """
    Enum for wire end axis
    """

    X = enum.auto()
    Z = enum.auto()


@dataclass
class DivertorDesignerParams(ParameterFrame):
    """
    Divertor designer parameters
    """

    # Length of divertor legs
    div_L2D_ib: Parameter[float]
    div_L2D_ob: Parameter[float]
    # Target length
    div_Ltarg_ib: Parameter[float]  # noqa: N815
    div_Ltarg_ob: Parameter[float]  # noqa: N815
    # Target Location - fraction of target length from the PFR side of target
    strike_loc_ib: Parameter[float]
    strike_loc_ob: Parameter[float]
    # Target angle (ccw)
    div_targ_angle_ib: Parameter[float]
    div_targ_angle_ob: Parameter[float]
    # Horizontal ot Vertical
    div_targ_type_ib: Parameter[str]
    div_targ_type_ob: Parameter[str]
    # Type of divertor baffles
    div_baffle_type: Parameter[str]


@dataclass
class DivertorBuilderParams(ParameterFrame):
    """
    Divertor builder parameters
    """

    n_TF: Parameter[int]
    n_div_cassettes: Parameter[int]
    c_rm: Parameter[float]


class DivertorDesigner(Designer[tuple[BluemiraWire, ...]]):
    """
    Divertor Designer
    """

    INNER_BAFFLE = "inner_baffle"
    OUTER_BAFFLE = "outer_baffle"
    DOME = "dome"
    INNER_TARGET = "inner_target"
    OUTER_TARGET = "outer_target"
    OPEN_BAFFLE = "open_baffle"
    CIRCLE_BAFFLE = "circle_baffle"
    FLUXLINE_BAFFLE = "fluxline_baffle"

    params: DivertorDesignerParams
    param_cls: type[DivertorDesignerParams] = DivertorDesignerParams

    def __init__(
        self,
        params: dict | ParameterFrame,
        equilibrium: Equilibrium,
        x_limits: tuple[float],
        z_limits: tuple[float],
        build_config: BuildConfig | None = None,
        keep_in_zone_wire: BluemiraWire | None = None,
        keep_out_zone_wires: list[BluemiraWire] | BluemiraWire | None = None,
    ):
        """
        Parameters
        ----------
        params:
            Divertor designer parameters
        equilibrium:
            The equilibrium to design around
        wall:
            wall boundary keep out zone (cut at divertor)
        build_config:
            Build configuration options for the divertor designer.
        keep_in_zone_wire:
            divertor keep in zone
        keep_out_zone_wires:
            List of any additional keep out zones
        """
        super().__init__(params, build_config)
        self.equilibrium = equilibrium
        self.x_limits = x_limits
        self.z_limits = z_limits
        self.kiz_wire = keep_in_zone_wire
        self.koz_wires = keep_out_zone_wires
        self.leg_length = {
            LegPosition.INNER: self.params.div_L2D_ib,
            LegPosition.OUTER: self.params.div_L2D_ob,
        }
        self.separatrix_legs = get_separatrix_legs(self.equilibrium)

    def _make_target(self, leg: LegPosition, label: str):
        """
        Divertor designer method for making the target.

        Parameters
        ----------
        leg:
            Position of leg to make target for
        label:
           The label to give the returned target

        Returns
        -------
        :
            Target
        """
        target_coords = self._make_angled_target(leg)
        return make_polygon(target_coords, label=label)

    def _make_angled_target(self, leg: LegPosition):
        """
        Method for making a target with the angle and lengths
        specified in the params (DivertorDesignerParams).

        Parameters
        ----------
        leg:
            Position of leg to make target for

        Returns
        -------
        :
            Coordinates of the target
        """
        sol = self._get_sols_for_leg(leg)[0]

        target_point = sol.value_at(distance=self.leg_length[leg].value)
        # use a point slightly further along the leg to get a vector
        # tangent to the sep. leg in the expect direction
        # i.e. towards the increasing leg length
        post_target_point = sol.value_at(distance=self.leg_length[leg].value + 0.1)

        a = post_target_point - target_point
        a_hat = a / np.linalg.norm(a)

        # ccw angle
        theta = (
            np.deg2rad(self.params.div_targ_angle_ib.value)
            if leg is LegPosition.INNER
            else np.deg2rad(self.params.div_targ_angle_ob.value)
        )

        rot_matrix = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 0, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ])  # ccw rotation about y-axis

        b_hat = rot_matrix @ a_hat

        if leg is LegPosition.INNER:
            pfr_side_length = (
                self.params.strike_loc_ib.value * self.params.div_Ltarg_ib.value
            )
            sol_side_length = (
                1 - self.params.strike_loc_ib.value
            ) * self.params.div_Ltarg_ib.value
            p1 = target_point - b_hat * sol_side_length
            p2 = target_point + b_hat * pfr_side_length
        else:
            pfr_side_length = (
                self.params.strike_loc_ob.value * self.params.div_Ltarg_ob.value
            )
            sol_side_length = (
                1 - self.params.strike_loc_ob.value
            ) * self.params.div_Ltarg_ob.value
            p1 = target_point - b_hat * pfr_side_length
            p2 = target_point + b_hat * sol_side_length

        return np.array([p1, p2]).T

    def _get_sols_for_leg(
        self, leg: LegPosition, layers: Iterable[int] = (0, -1)
    ) -> list[BluemiraWire]:
        """
        Get the selected scrape-off-leg layers from the separatrix legs.

        Returns
        -------
        :
            Separatrix legs
        """
        return [self.separatrix_legs[leg][layer] for layer in layers]

    def make_dome(
        self,
        start: np.ndarray,
        end: np.ndarray,
        label: str,
        start_picked: bool | None = None,  # noqa: FBT001
    ) -> BluemiraWire:
        """
        Make a dome between the two given points.

        Returns
        -------
        :
            Divertor dome

        Notes
        -----
        The dome shape follows a constant line of flux that is closest to the input
        coordinates.
        The nearest point on the flux surface to the start point and the end point are
        joined.
        The default is that the flux surface is picked based on the lowest z coordinate
        of the start and end point to ensure a continuous divertor shape is produced.
        """
        if start_picked is None:
            start_picked = start[1] < end[1]
        return self.make_flux_line_wire(start, end, start_picked, label)

    def make_flux_line_wire(
        self,
        start: np.ndarray,
        end: np.ndarray,
        start_picked: bool,  # noqa: FBT001
        label: str,
    ):
        """
        Get a constant line of flux that is closest to the input coordinates.

        The nearest point on the flux surface to the start point and the end point
        are joined.

        Returns
        -------
        :
            Selected flux line as a wire.
        """
        # Get the flux surface that crosses the through the start or end point.
        # We can use this surface to guide the shape of the wire.
        pick_point = start if start_picked else end
        psi_start = self.equilibrium.psi(*pick_point)
        flux_surface = find_flux_surface_through_point(
            self.equilibrium.x,
            self.equilibrium.z,
            self.equilibrium.psi(),
            start[0],
            start[1],
            psi_start,
        )
        # Get the indices of the closest points on the flux surface to
        # the input start and end points
        start_coord = start.reshape((2, 1))
        end_coord = end.reshape((2, 1))
        idx = np.array([
            np.argmin(np.hypot(*(flux_surface - start_coord))),
            np.argmin(np.hypot(*(flux_surface - end_coord))),
        ])

        # Make sure the start and end are in the right order
        # idx[0] should be smaller than idx[1]
        if idx[0] > idx[1]:
            # swap
            idx = idx[::-1]
            # cut
            flux_contour = flux_surface[:, idx[0] + 1 : idx[1]]
            # reverse, so that the contour ends at the end point
            flux_contour = flux_contour[:, ::-1]
        else:
            flux_contour = flux_surface[:, idx[0] + 1 : idx[1]]

        # Build the coords of the flux contour in 3D (all(y == 0))
        contour_coords = np.zeros((3, flux_contour.shape[1] + 1))
        if start_picked:
            contour_coords[(0, 2), 0] = start
            contour_coords[(0, 2), 1:] = flux_contour
            # replace last point with end
            contour_coords[(0, 2), -1] = end
        else:
            contour_coords[(0, 2), 0:-1] = flux_contour
            contour_coords[(0, 2), -1] = end
            # replace first point with start
            contour_coords[(0, 2), 0] = start

        return interpolate_bspline(contour_coords, label=label)

    def make_baffle(
        self,
        label: str,
        target_baffle_join_point: np.ndarray,
        target_dome_join_point: np.ndarray,
        target_start: bool | None = None,  # noqa: FBT001
    ) -> BluemiraWire:
        """
        Divertor designer method for making the baffles.

        A baffle wire joins the target wire to the wall wire.

        Parameters
        ----------
        label:
            The label to give the returned Component.
        target_baffle_join_point:
            The position (in x-z) where the target connects to the baffle.
        target_dome_join_point:
            The position (in x-z) where the target connects to the dome.
        target_start:
            Determines which flux surface is selected to create the baffle shape
            when using a fluxline baffle design (see div_baffle_type parameter).
            True -> use flux surface closest to target join point.
            False -> use flux surface closest to wall join point.
            Default (None) will mean that the point with the lowest z value is selected.

        Returns
        -------
        :
            The baffle shape.
        """

        def grad_xz(p1: np.ndarray, p2: np.ndarray):
            """
            Calculate gradient in xz.

            Returns
            -------
            :
                gradient
            """
            x1, z1 = p1[0], p1[1]
            x2, z2 = p2[0], p2[1]
            if np.isclose(z1, z2):
                return 0
            if np.isclose(x1, x2):
                return np.inf
            return (z1 - z2) / (x1 - x2)

        idx_inner = np.argmin(self.x_limits)
        idx_outer = np.argmax(self.x_limits)

        wall_join_point = (
            np.array([
                self.x_limits[idx_inner],
                self.z_limits[idx_inner],
            ])
            if label == self.INNER_BAFFLE
            else np.array([
                self.x_limits[idx_outer],
                self.z_limits[idx_outer],
            ])
        )

        # Different methods used to create the baffle shape.
        if self.params.div_baffle_type.value == self.OPEN_BAFFLE:
            raise NotImplementedError("Open baffle not implemented")
        if self.params.div_baffle_type.value == self.FLUXLINE_BAFFLE:
            return self._make_fluxline_baffle(
                label,
                wall_join_point,
                target_baffle_join_point,
                target_start,
            )
        target_gradient = grad_xz(target_baffle_join_point, target_dome_join_point)
        return self._make_circular_baffle(
            label, wall_join_point, target_baffle_join_point, target_gradient
        )

    def _make_fluxline_baffle(
        self,
        label: str,
        wall_join_point: np.ndarray,
        target_join_point: np.ndarray,
        target_start: bool | None = None,  # noqa: FBT001
    ) -> BluemiraWire:
        """
        Make a baffle using the divertor leg flux line shape.

        Parameters
        ----------
        label:
            The label to give the returned Component.
        wall_join_point:
            The position (in x-z) where the wall connects to the baffle.
        target_baffle_join_point:
            The position (in x-z) where the target connects to the baffle.
        target_start:
            Determines which flux surface is selected to create the baffle shape.
            True -> use flux surface closest to target join point.
            False -> use flux surface closest to wall join point.

        Returns
        -------
        :
            The baffle shape.

        Notes
        -----
        The baffle shape follows a constant line of flux that is closest to the input
        coordinates.
        The nearest point on the flux surface to the start point and the end point are
        joined.
        The default is that the flux surface is picked based on the lowest z coordinate
        of the start and end point.
        """
        return self.make_flux_line_wire(
            start=target_join_point,
            end=wall_join_point,
            start_picked=target_start,
            label=label,
        )

    def _make_circular_baffle(
        self,
        label: str,
        wall_join_point: np.ndarray,
        target_join_point: np.ndarray,
        target_gradient: float,
    ) -> BluemiraWire:
        """
        Make a circular baffle which is tangent to the target.

        Parameters
        ----------
        label:
            The label to give the returned Component.
        target_baffle_join_point:
            The position (in x-z) where the target connects to the baffle.
        target_dome_join_point:
            The position (in x-z) where the target connects to the dome.
        target_gradient:
            The gradient (in x-z) of the divertor target.

        Returns
        -------
        :
            The baffle shape.

        Raises
        ------
        ValueError
            Baffle internal and external radii are not equal
        """
        bx, bz = wall_join_point[0], wall_join_point[1]
        tx, tz = target_join_point[0], target_join_point[1]
        mt = target_gradient

        def solve(l1: tuple[float, ...], l2: tuple[float, ...]):
            A = np.array([[l1[0], l1[1]], [l2[0], l2[1]]])
            b = np.array([l1[2], l2[2]])
            return np.linalg.solve(A, b)

        # Solve the two linear equations below, in the form Ax + Bz = C
        # This comes from solving for where a circle
        # will intersect both the target and blanket join points
        l1 = (
            2 * bx - 2 * tx,  # A
            2 * bz - 2 * tz,  # B
            (bx**2) + bz**2 - (tx**2 + tz**2),  # C
        )

        # These are co-effs for a line perpendicular to the target
        # (through the target join point), found using
        # the fact that the gradient of the line is the negative
        # reciprocal of the target gradient
        # the equ: x + mt*z = tx + mt*tz
        l2 = (
            0 if mt == np.inf else 1,  # A
            1 if mt == np.inf else mt,  # B
            tz if mt == np.inf else tx + mt * tz,  # C
        )
        arc_center_point = solve(l1, l2)
        ox, oz = arc_center_point[0], arc_center_point[1]
        if oz > bz:
            raise ValueError(
                "Arc center point is above the blanket join point, "
                f"make the target angle sharper: {label}"
            )

        deg_t = np.rad2deg(np.arctan2(tz - oz, tx - ox))
        deg_b = np.rad2deg(np.arctan2(bz - oz, bx - ox))

        start_angle = deg_t
        end_angle = deg_b
        if label == self.OUTER_BAFFLE:
            start_angle = deg_b
            end_angle = deg_t

        radius_t = float(np.linalg.norm(target_join_point - arc_center_point))
        radius_b = float(np.linalg.norm(wall_join_point - arc_center_point))

        if not np.isclose(radius_b, radius_t):
            raise ValueError("radii must be equal")

        # make_circle_arc_3P would have been used but it put the arc center
        # somewhere weird so could not be used
        return make_circle(
            radius=radius_t,
            center=(ox, 0, oz),
            axis=(0, -1, 0),
            start_angle=start_angle,
            end_angle=end_angle,
            label=label,
        )

    def _get_wire_ends_by_psi(self, wire: BluemiraWire) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the coordinates of the ends of a wire where the end
        with higher psi is returned first

        Returns
        -------
        :
            higher psi point
        :
            lower psi point
        """
        start_point = wire.start_point()
        end_point = wire.end_point()

        psi_start = self.equilibrium.psi(x=start_point.x[0], z=start_point.z[0])
        psi_end = self.equilibrium.psi(x=end_point.x[0], z=end_point.z[0])

        if psi_start < psi_end:
            return start_point.xz.flatten(), end_point.xz.flatten()
        return end_point.xz.flatten(), start_point.xz.flatten()

    @staticmethod
    def _get_wire_end_with_smallest(wire: BluemiraWire, axis: str) -> np.ndarray:
        """
        Get the coordinates of the end of a wire with largest value in
        the given dimension

        Returns
        -------
        :
            Wire end point
        """
        return DivertorDesigner._get_wire_end(wire, axis, operator.lt)

    @staticmethod
    def _get_wire_end_with_largest(wire: BluemiraWire, axis: str) -> np.ndarray:
        """
        Get the coordinates of the end of a wire with largest value in
        the given dimension

        Returns
        -------
        :
            Wire end point
        """
        return DivertorDesigner._get_wire_end(wire, axis, operator.gt)

    @staticmethod
    def _get_wire_end(wire: BluemiraWire, axis: str, comp: Callable) -> np.ndarray:
        """
        Get the coordinates of the end of a wire whose coordinate in the
        given axis satisfies the comparison function.

        Returns
        -------
        :
            Wire end point
        """
        axis = WireEndAxis[axis.upper()].name.lower()

        start_point = wire.start_point()
        end_point = wire.end_point()
        if comp(getattr(start_point, axis), getattr(end_point, axis)):
            return start_point.xz.flatten()
        return end_point.xz.flatten()


class DivertorBuilder(Builder):
    """
    Divertor builder
    """

    DIV = "DIV"
    BODY = "Body"
    CASETTES = "cassettes"
    SEGMENT_PREFIX = "segment"
    param_cls: type[DivertorBuilderParams] = DivertorBuilderParams
    params: DivertorBuilderParams

    def __init__(
        self,
        params: ParameterFrameLike,
        build_config: BuildConfig,
        divertor_silhouette: BluemiraFace,
    ):
        super().__init__(params, build_config)
        self.div_koz = divertor_silhouette

    def build(self) -> Component:
        """
        Build the divertor component.
        """  # noqa: DOC201
        return self.component_tree(
            xz=[self.build_xz()],
            xy=[],
            xyz=self.build_xyz(degree=0),
        )

    def build_xz(self) -> PhysicalComponent:
        """
        Build the x-z components of the divertor.
        """  # noqa: DOC201
        body = PhysicalComponent(self.BODY, self.div_koz)
        apply_component_display_options(body, color=BLUE_PALETTE[self.DIV][0])

        return body

    def build_xyz(self, degree: float = 360.0) -> list[PhysicalComponent]:
        """
        Build the x-y-z components of the divertor.
        """  # noqa: DOC201
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)
        shapes = pattern_revolved_silhouette(
            self.div_koz,
            self.params.n_div_cassettes.value,
            self.params.n_TF.value,
            self.params.c_rm.value,
        )

        segments = []
        for no, shape in enumerate(shapes):
            segment = PhysicalComponent(
                f"{self.SEGMENT_PREFIX}_{no}",
                shape,
                material=self.get_material(),
            )
            apply_component_display_options(segment, BLUE_PALETTE[self.DIV][no])
            segments.append(segment)

        return circular_pattern_component(
            Component(self.CASETTES, children=segments),
            n_sectors,
            degree=sector_degree * n_sectors,
        )

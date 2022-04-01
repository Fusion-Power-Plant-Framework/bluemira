# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
"""
Define builder for divertor
"""

import enum
import operator
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np

from bluemira.base.builder import BuildConfig, Builder, Component
from bluemira.base.components import PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.base.error import BuilderError
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import find_flux_surface_through_point, get_legs
from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire


class LegPosition(enum.Enum):
    """
    Enum classifying divertor/separatrix leg positions
    """

    INNER = enum.auto()
    OUTER = enum.auto()


def get_separatrix_legs(
    equilibrium: Equilibrium,
) -> Dict[LegPosition, List[BluemiraWire]]:
    """
    Find the separatrix legs for the given equilibrium.
    """
    # A flag specifying which end of the plasma (i.e., upper or lower)
    # we want the legs from will need to be added
    legs = get_legs(equilibrium)
    separatrix_legs = {
        LegPosition.INNER: [make_polygon(loop.xyz) for loop in legs["lower_inner"]],
        LegPosition.OUTER: [make_polygon(loop.xyz) for loop in legs["lower_outer"]],
    }
    return separatrix_legs


class DivertorSilhouetteBuilder(Builder):
    """
    Build an EUDEMO divertor.

    The builder outputs a component with the structure:

    .. code-block::

        divertor (Component)
        └── xz (Component)
            ├── inner_target (PhysicalComponent)
            ├── outer_target (PhysicalComponent)
            ├── dome (PhysicalComponent)
            ├── inner_baffle (PhysicalComponent)
            └── outer_baffle (PhysicalComponent)
    """

    COMPONENT_INNER_BAFFLE = "inner_baffle"
    COMPONENT_OUTER_BAFFLE = "outer_baffle"
    COMPONENT_DOME = "dome"
    COMPONENT_INNER_TARGET = "inner_target"
    COMPONENT_OUTER_TARGET = "outer_target"

    _required_params = [
        "div_L2D_ib",
        "div_L2D_ob",
        "div_Ltarg",
        "div_open",
    ]
    _required_config: List[str] = []
    _params: Configuration

    def __init__(
        self,
        params: Dict[str, Any],
        build_config: BuildConfig,
        equilibrium: Equilibrium,
        x_limits: Sequence[float],
        z_limits: Sequence[float],
    ):
        super().__init__(params, build_config)

        self._shape = None

        self.equilibrium = equilibrium
        self.x_limits = x_limits
        self.z_limits = z_limits
        self.leg_length = {
            LegPosition.INNER: self.params["div_L2D_ib"],
            LegPosition.OUTER: self.params["div_L2D_ob"],
        }
        self.separatrix_legs = get_separatrix_legs(self.equilibrium)
        _, self.x_points = self.equilibrium.get_OX_points()

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        return super().reinitialise(params, **kwargs)

    def build(self) -> Component:
        """
        Build the divertor component.
        """
        component = super().build()
        component.add_child(self.build_xz())
        return component

    def build_xz(self) -> Component:
        """
        Build the divertor's components in the xz-plane.
        """
        if not self.x_limits:
            raise BuilderError("Divertor's x-limits not set.")

        component = Component("xz")

        # Build the targets for each separatrix leg
        inner_target = self.make_target(LegPosition.INNER, self.COMPONENT_INNER_TARGET)
        outer_target = self.make_target(LegPosition.OUTER, self.COMPONENT_OUTER_TARGET)
        for target in [inner_target, outer_target]:
            component.add_child(target)

        # Build the dome based on target positions
        inner_target_end = self._get_wire_end_with_smallest(inner_target.shape, "z")
        outer_target_start = self._get_wire_end_with_smallest(outer_target.shape, "z")
        dome = self.make_dome(
            inner_target_end, outer_target_start, label=self.COMPONENT_DOME
        )
        component.add_child(dome)

        # Build the baffles
        idx_inner = np.argmin(self.x_limits)
        x_lim_inner = self.x_limits[idx_inner]
        z_lim_inner = self.z_limits[idx_inner]
        component.add_child(
            self._make_inner_baffle(inner_target.shape, x_lim_inner, z_lim_inner)
        )
        idx_outer = np.argmax(self.x_limits)
        x_lim_outer = self.x_limits[idx_outer]
        z_lim_outer = self.z_limits[idx_outer]
        component.add_child(
            self._make_outer_baffle(outer_target.shape, x_lim_outer, z_lim_outer)
        )
        return component

    def make_target(self, leg: LegPosition, label: str) -> PhysicalComponent:
        """
        Make a divertor target for a the given leg.
        """
        sols = self._get_sols_for_leg(leg)
        leg_length = self._get_length_for_leg(leg)
        # Just use the first scrape-off layer for now
        point = sols[0].value_at(distance=leg_length)

        # Create some vertical targets for now. Eventually the target
        # angle will be derived from the grazing-angle parameter
        target_length = self.params.div_Ltarg
        target_coords = np.array(
            [
                [point[0], point[0]],
                [point[1], point[1]],
                [point[2] - target_length / 2, point[2] + target_length / 2],
            ]
        )
        return PhysicalComponent(label, make_polygon(target_coords))

    def make_dome(self, start: Sequence[float], end: Sequence[float], label: str):
        """
        Make a dome between the two given points

        The dome shape follows a constant line of flux that is closest
        to the input start coordinate. Finally, the nearset point on the
        flux surface to the end point and the end point are joined.
        """
        # Get the flux surface that crosses the through the start point
        # We can use this surface to guide the shape of the dome
        psi_start = self.equilibrium.psi(*start)
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
        start_coord = np.array([[start[0]], [start[1]]])  # [[x], [z]]
        end_coord = np.array([[end[0]], [end[1]]])
        idx = np.array(
            [
                np.argmin(np.hypot(*(flux_surface - start_coord))),
                np.argmin(np.hypot(*(flux_surface - end_coord))),
            ]
        )

        # Make sure the start and end are in the right order
        if idx[0] > idx[1]:
            idx = idx[::-1]
            dome_contour = flux_surface[:, idx[0] + 1 : idx[1]]
            dome_contour = dome_contour[:, ::-1]
        else:
            dome_contour = flux_surface[:, idx[0] + 1 : idx[1]]

        # Build the coords of the dome in 3D (all(y == 0))
        dome = np.zeros((3, dome_contour.shape[1] + 2))
        dome[(0, 2), 0] = start_coord.T
        dome[(0, 2), 1:-1] = dome_contour
        dome[(0, 2), -1] = end_coord.T

        return PhysicalComponent(label, make_polygon(dome))

    def make_baffle(
        self,
        label: str,
        start: Sequence[float],
        end: Sequence[float],
    ) -> PhysicalComponent:
        """
        Make a baffle.

        The baffle shape is a straight line between the given start and
        end points.

        Parameters
        ----------
        label: str
            The label to give the returned Component.
        start: Sequence
            The position (in x-z) to start drawing the baffle from,
            e.g., the outside end of a target.
        end: Sequence
            The position (in x-z) to stop drawing the baffle, e.g., the
            postion to the upper part of the first wall.
        """
        coords = np.array([[start[0], end[0]], [0, 0], [start[1], end[1]]])
        return PhysicalComponent(label, make_polygon(coords))

    def _make_inner_baffle(
        self,
        target: BluemiraWire,
        x_lim: float,
        z_lim: float,
    ) -> PhysicalComponent:
        """
        Build the inner baffle to join with the given target.
        """
        if self.params.div_open:
            raise NotImplementedError("Open divertor baffles not yet supported")
        else:
            inner_target_start = self._get_wire_end_with_largest(target, "x")
        return self.make_baffle(
            label=self.COMPONENT_INNER_BAFFLE,
            start=np.array([x_lim, z_lim]),
            end=inner_target_start,
        )

    def _make_outer_baffle(
        self,
        target: BluemiraWire,
        x_lim: float,
        z_lim: float,
    ) -> PhysicalComponent:
        """
        Build the outer baffle to join with the given target.
        """
        if self.params.div_open:
            raise NotImplementedError("Open divertor baffles not yet supported")
        else:
            outer_target_end = self._get_wire_end_with_largest(target, "x")
        return self.make_baffle(
            label=self.COMPONENT_OUTER_BAFFLE,
            start=outer_target_end,
            end=np.array([x_lim, z_lim]),
        )

    def _get_length_for_leg(self, leg: LegPosition):
        """
        Retrieve the length of the given leg from the parameters.
        """
        if leg is LegPosition.INNER:
            return self.params.div_L2D_ib
        elif leg is LegPosition.OUTER:
            return self.params.div_L2D_ob

    def _get_sols_for_leg(
        self, leg: LegPosition, layers: Iterable[int] = (0, -1)
    ) -> List[BluemiraWire]:
        """
        Get the selected scrape-off-leg layers from the separatrix legs.
        """
        sols = []
        for layer in layers:
            sols.append(self.separatrix_legs[leg][layer])
        return sols

    @staticmethod
    def _get_wire_end_with_smallest(wire: BluemiraWire, axis: str) -> np.ndarray:
        """
        Get the coordinates of the end of a wire with largest value in
        the given dimension
        """
        return DivertorSilhouetteBuilder._get_wire_end(wire, axis, operator.lt)

    @staticmethod
    def _get_wire_end_with_largest(wire: BluemiraWire, axis: str) -> np.ndarray:
        """
        Get the coordinates of the end of a wire with largest value in
        the given dimension
        """
        return DivertorSilhouetteBuilder._get_wire_end(wire, axis, operator.gt)

    @staticmethod
    def _get_wire_end(wire: BluemiraWire, axis: str, comp: Callable):
        """
        Get the coordinates of the end of a wire whose coordinate in the
        given axis satisfies the comparision function.
        """
        allowed_axes = ["x", "z"]
        if axis not in allowed_axes:
            raise ValueError(
                f"Unrecognised axis '{axis}'. Must be one of: {allowed_axes}."
            )

        start_point = wire.start_point()
        end_point = wire.end_point()
        if comp(getattr(start_point, axis), getattr(end_point, axis)):
            return start_point.xz.flatten()
        return end_point.xz.flatten()

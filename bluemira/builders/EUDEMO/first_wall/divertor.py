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
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

from bluemira.base.builder import BuildConfig, Builder, Component
from bluemira.base.components import PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import find_flux_surface_through_point, get_legs
from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry.tools import find_point_along_wire_at_length, make_polygon
from bluemira.geometry.wire import BluemiraWire


class LegPosition(enum.Enum):
    """
    Enum classifying divertor/separatrix leg positions
    """

    INNER = enum.auto()
    OUTER = enum.auto()
    CORE1 = enum.auto()
    CORE2 = enum.auto()


def parse_legs(legs: Dict[str, List[Loop]]):
    separatrix_legs = {
        LegPosition.INNER: [make_polygon(loop.xyz) for loop in legs["lower_inner"]],
        LegPosition.OUTER: [make_polygon(loop.xyz) for loop in legs["lower_outer"]],
    }
    return separatrix_legs


class DivertorBuilder(Builder):
    """
    Build an EUDEMO divertor.
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
    _default_runmode: str = "mock"

    def __init__(
        self,
        params: Dict[str, Any],
        build_config: BuildConfig,
        equilibrium: Equilibrium,
        x_lims: Iterable[float],
        **kwargs,
    ):
        super().__init__(params, build_config, **kwargs)

        self._shape = None
        self.boundary: BluemiraWire = None

        self.x_lims = sorted(x_lims)
        self.equilibrium = equilibrium
        self.leg_length = {
            LegPosition.INNER: self.params["div_L2D_ib"],
            LegPosition.OUTER: self.params["div_L2D_ob"],
        }
        self.separatrix_legs = parse_legs(get_legs(self.equilibrium))
        _, self.x_points = self.equilibrium.get_OX_points()

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        return super().reinitialise(params, **kwargs)

    def mock(self) -> Component:
        """
        Create a basic shape for the wall's boundary.
        """
        pass

    def build(self, **kwargs) -> Component:
        """
        Build the divertor component.
        """
        component = super().build(**kwargs)
        component.add_child(self.build_xz())
        return component

    def build_xz(self) -> Component:
        """
        Build the divertor's components in the xz-plane.
        """
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

        # Build the inner baffle
        if self.params.div_open:
            pass
        else:
            inner_target_start = self._get_wire_end_with_largest(inner_target.shape, "x")
        inner_baffle = self.make_baffle(
            label=self.COMPONENT_INNER_BAFFLE,
            start=np.array([self.x_lims[0], self.x_points[0].z]),
            end=inner_target_start,
            initial_vec=None,
        )
        component.add_child(inner_baffle)

        # Build the outer baffle
        if self.params.div_open:
            outer_target_end = self._get_wire_end_with_smallest(outer_target.shape, "x")
        else:
            outer_target_end = self._get_wire_end_with_largest(outer_target.shape, "x")
        outer_baffle = self.make_baffle(
            label=self.COMPONENT_OUTER_BAFFLE,
            start=outer_target_end,
            end=np.array([self.x_lims[1], self.x_points[0].z]),
            initial_vec=None,
        )
        component.add_child(outer_baffle)

        return component

    def make_target(self, leg: LegPosition, label: str) -> PhysicalComponent:
        """
        Make a divertor target for a the given leg.
        """
        sol = self._get_sol_for_leg(leg)
        try:
            leg_length = self._get_length_for_leg(leg)
        except ValueError as exc:
            raise ValueError(
                f"Cannot make target for leg '{leg}'. Only inner and outer legs "
                "supported."
            ) from exc

        # We need to work out which SOL to use here
        point, _ = find_point_along_wire_at_length(sol[0], leg_length)

        # Just create some vertical targets for now. Eventually the
        # target angle will be set using a grazing-angle parameter
        target_length = self.params.div_Ltarg
        target_coords = np.array(
            [
                [point[0], point[0]],
                [point[1], point[1]],
                [point[2] - target_length / 2, point[2] + target_length / 2],
            ]
        )
        return PhysicalComponent(label, make_polygon(target_coords))

    def make_dome(
        self, start: Tuple[float, float], end: Tuple[float, float], label: str
    ):
        """
        Make a dome between the two given points
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

        # Build the coords of the dome in 3-D (all(y == 0))
        dome = np.zeros((3, dome_contour.shape[1] + 2))
        dome[(0, 2), 0] = start_coord.T
        dome[(0, 2), 1:-1] = dome_contour
        dome[(0, 2), -1] = end_coord.T

        return PhysicalComponent(label, make_polygon(dome))

    def make_baffle(
        self,
        label: str,
        start: np.ndarray,
        end: np.ndarray,
        initial_vec: np.ndarray,
    ) -> PhysicalComponent:
        """
        Make a baffle.

        Parameters
        ----------
        leg: LegPosition
            The position of the leg the baffle should join to (e.g.,
            inner).
        start: np.ndarray(2, 1)
            The position (in x-z) to start drawing the baffle from,
            e.g., the outside end of a target.
        end: np.ndarray(2, 1)
            The position (in x-z) to stop drawing the baffle, e.g., the
            postion to the upper part of the first wall.
        initial_vec: np.ndarray(2, 1)
            A vector specifying the direction the start of the baffle
            should be drawn in, e.g., the orientation of a target.
            This is converted to a unit vector, so the magnitude is
            ignored.
        """
        # initial_unit_vec = initial_vec / np.hypot(initial_vec)

        # Just generate the most basic baffle: a straight line
        coords = np.array([[start[0], end[0]], [0, 0], [start[1], end[1]]])
        return PhysicalComponent(label, make_polygon(coords))

    def _get_length_for_leg(self, leg: LegPosition):
        """
        Retrieve the length of the given leg from the parameters.
        """
        if leg is LegPosition.INNER:
            return self.params.div_L2D_ib
        elif leg is LegPosition.OUTER:
            return self.params.div_L2D_ob
        raise ValueError(f"No length exists for leg '{leg}'.")

    def _get_sol_for_leg(
        self, leg: LegPosition, layers: Iterable[int] = (0, -1)
    ) -> BluemiraWire:
        """
        Get the selected scrape-off-leg layers from the separatrix legs.
        """
        sol = []
        for layer in layers:
            sol.append(self.separatrix_legs[leg][layer])
        return sol

    @staticmethod
    def _get_wire_end_with_smallest(wire: BluemiraWire, axis: str) -> np.ndarray:
        """
        Get the coordinates of the end of a wire with largest value in
        the given dimension
        """
        return DivertorBuilder._get_wire_end(wire, axis, operator.lt)

    @staticmethod
    def _get_wire_end_with_largest(wire: BluemiraWire, axis: str) -> np.ndarray:
        """
        Get the coordinates of the end of a wire with largest value in
        the given dimension
        """
        return DivertorBuilder._get_wire_end(wire, axis, operator.gt)

    @staticmethod
    def _get_wire_end(wire: BluemiraWire, axis: str, comp: Callable):
        """
        Get the coordinates of the end of a wire whose coordinate in the
        given axis satisfies the comparision function.
        """
        allowed_axes = ["x", "z"]
        if axis not in allowed_axes:
            raise ValueError("Unrecognised axis '{}'. Must be one of '{}'.").format(
                axis, "', '".join(allowed_axes)
            )

        start_point = wire.start_point()
        end_point = wire.end_point()
        axis_idx = 0 if axis == "x" else -1
        if comp(start_point[axis_idx], end_point[axis_idx]):
            return start_point[[0, -1]]
        return end_point[[0, -1]]


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt

    from bluemira.base.file import get_bluemira_path, get_bluemira_root
    from bluemira.display.plotter import plot_2d

    DATA = get_bluemira_path("bluemira/equilibria/test_data", subfolder="tests")
    eq_filename = os.path.join(DATA, "eqref_OOB.json")
    eq = Equilibrium.from_eqdsk(eq_filename)
    # eq.plot()  # throws exception

    _, x_points = eq.get_OX_points()
    # # print(ox_points)

    fig, ax = plt.subplots()
    # o_points_x = [p.x for p in ox_points[0]]
    # o_points_y = [p.z for p in ox_points[0]]
    # x_points_x = [p.x for p in ox_points[1]]
    # x_points_y = [p.z for p in ox_points[1]]
    # ax.plot(o_points_x, o_points_y, color="blue", marker="o", linestyle="none")
    # ax.plot(x_points_x, x_points_y, color="g", marker="x", linestyle="none")

    separatrix = eq.get_separatrix()
    separatrix.plot(ax=ax, linestyle="--", linewidth=0.3)
    # separatrix[0].plot(ax=ax, linestyle="--", linewidth=0.3)
    # separatrix[1].plot(ax=ax, linestyle="--", linewidth=0.3)
    # # plt.show()

    params = {
        "Name": "Divertor example",
        "div_L2D_ib": (1.1, "Input"),
        "div_L2D_ob": (1.45, "Input"),
        "div_Ltarg": (0.5, "Input"),
        "div_open": (False, "Input"),
    }
    build_config = {"name": "divertor", "runmode": "mock"}

    div = DivertorBuilder(
        params,
        build_config=build_config,
        equilibrium=eq,
        x_lims=[x_points[0].x - 4, x_points[0].x + 4],
    )
    component = div()
    component.plot_2d(ax=ax)

    # for leg_set in div.separatrix_legs.values():
    #     plot_2d(leg_set[0], ax=ax, show=False)
    # plot_2d(leg_set[1], ax=ax, show=False, color="g")

    plt.show()

    # div.make_target(LegPosition.INNER)

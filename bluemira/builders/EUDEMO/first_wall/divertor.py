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
from typing import Any, Dict, Iterable, List

import numpy as np

from bluemira.base.builder import BuildConfig, Builder, Component
from bluemira.base.components import PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry.tools import find_point_along_wire_at_length, make_polygon
from bluemira.geometry.wire import BluemiraWire
from BLUEPRINT.nova.stream import StreamFlow


class Leg(enum.Enum):
    """
    Enum classifying divertor/separatrix leg positions
    """

    INNER = enum.auto()
    OUTER = enum.auto()
    CORE1 = enum.auto()
    CORE2 = enum.auto()


def _equilibrium_to_stream_flow(equilibrium: Equilibrium) -> StreamFlow:
    """
    Convert an equilibrium object to a StreamFlow.
    """
    import os
    import tempfile

    tmp_file_name = "tmp_eq.eqdsk"
    tmp_path = os.path.join(tempfile.gettempdir(), "tmp_eq.eqdsk")
    equilibrium.to_eqdsk(tmp_file_name, directory=tempfile.gettempdir())
    try:
        sf = StreamFlow(tmp_path + ".json")
    finally:
        os.remove(tmp_path + ".json")
    return sf


def get_legs(equilibrium: Equilibrium) -> Dict[Leg, List[BluemiraWire]]:
    """
    Hacky implementation leveraging StreamFlow to find the legs of the
    given equilibrium's separatrix.
    """
    from bluemira.geometry.tools import make_polygon

    stream_flow = _equilibrium_to_stream_flow(equilibrium)
    stream_flow.sol()
    stream_flow.get_legs()

    def _parse_legs(sf_leg):
        """
        Convert the given list of leg 'structs' into BluemiraWires.
        """
        x_legs = sf_leg["X"]
        z_legs = sf_leg["Z"]
        legs = []
        for x_leg, z_leg in zip(x_legs, z_legs):
            leg = np.array([x_leg, np.zeros(x_leg.shape), z_leg])
            legs.append(make_polygon(leg))
        return legs

    legs = {
        Leg.INNER: _parse_legs(stream_flow.legs["inner"]),
        Leg.OUTER: _parse_legs(stream_flow.legs["outer"]),
        Leg.CORE1: _parse_legs(stream_flow.legs["core1"]),
        Leg.CORE2: _parse_legs(stream_flow.legs["core2"]),
    }
    return legs


def point_along_wire_at_length(wire: BluemiraWire, length: float):
    """
    Find the point that is a given length along a wire, and the
    unit tanget vector along that point and its neighbour.

    This method discretizes the wire in order to find the desired point.
    Because of this, the error in this calculation will depend on the
    discretization's step size.
    """
    # TODO(hsaunders1904): magic number here needs justification
    coords = wire.discretize(ndiscr=2000)
    segment_lengths = np.linalg.norm(np.diff(coords, axis=1), axis=0)
    cumulative_lengths = np.cumsum(segment_lengths)
    if length > cumulative_lengths[-1]:
        raise ValueError(
            "Given length ({length}) greater than wire length ({wire.length})."
        )
    index = np.searchsorted(cumulative_lengths, length)

    tangent_vec = coords[:, index] - coords[:, index - 1]
    unit_tangent_vec = tangent_vec / np.linalg.norm(tangent_vec)

    # Could potentially use the calculated coordinate as the starting
    # point for some sort of optimization/root finder if this needs to
    # be more accurate in the future

    return coords[:, index], unit_tangent_vec


class DivertorBuilder(Builder):
    """
    Build an EUDEMO divertor.
    """

    _required_params = [
        "div_L2D_ib",
        "div_L2D_ob",
        "div_Ltarg",
    ]
    _required_config: List[str] = []
    _params: Configuration
    _default_runmode: str = "mock"

    def __init__(
        self,
        params: Dict[str, Any],
        build_config: BuildConfig,
        equilibrium: Equilibrium,
        **kwargs,
    ):
        super().__init__(params, build_config, **kwargs)

        self._shape = None
        self.boundary: BluemiraWire = None

        self.equilibrium = equilibrium
        self.o_points, self.x_points = self._get_OX_points()
        self.leg_length = {
            Leg.INNER: self.params["div_L2D_ib"],
            Leg.OUTER: self.params["div_L2D_ob"],
        }
        self.separatrix_legs = get_legs(self.equilibrium)

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
        component = super().build(**kwargs)
        component.add_child(self.build_xz())
        return component

    def build_xz(self, **kwargs) -> Component:
        """
        Build the divertor's components in the xz-plane.
        """
        component = Component("xz")
        for leg in [Leg.INNER, Leg.OUTER]:
            component.add_child(self.make_target(leg, f"target {leg}"))
        return component

    def make_target(self, leg: Leg, label: str) -> Component:
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

    def _get_length_for_leg(self, leg: Leg):
        """
        Retrieve the length of the given leg from the parameters.
        """
        if leg is Leg.INNER:
            return self.params.div_L2D_ib
        elif leg is Leg.OUTER:
            return self.params.div_L2D_ob
        raise ValueError(f"No length exists for leg '{leg}'.")

    def _get_sol_for_leg(
        self, leg: Leg, layers: Iterable[int] = (0, -1)
    ) -> BluemiraWire:
        """
        Get the selected scrape-off-leg layers from the separatrix legs.
        """
        sol = []
        for layer in layers:
            sol.append(self.separatrix_legs[leg][layer])
        return sol

    @property
    def separatrix(self):
        # Use a cached property for now.
        # We may want to pass in the separatrix directly, not an Equilibrium instance
        if not hasattr(self, "_separatrix"):
            self._separatrix = self.equilibrium.get_separatrix()
        return self._separatrix

    def _get_OX_points(self):
        """
        Get the OX points from this object's equilibrium.
        """
        o_points, x_points = self.equilibrium.get_OX_points()
        return (
            np.array([[point[0], point[1]] for point in o_points]),
            np.array([[point[0], point[1]] for point in x_points]),
        )

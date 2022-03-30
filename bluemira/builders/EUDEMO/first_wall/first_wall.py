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
Builders for the first wall of the reactor, including divertor
"""

from copy import deepcopy
from typing import Any, Dict

import numpy as np

from bluemira.base.builder import BuildConfig, Component
from bluemira.base.components import PhysicalComponent
from bluemira.base.error import BuilderError
from bluemira.builders.EUDEMO.first_wall.divertor import DivertorBuilder
from bluemira.builders.EUDEMO.first_wall.wall import WallBuilder
from bluemira.builders.shapes import Builder
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import find_OX_points
from bluemira.geometry.tools import (
    boolean_cut,
    convex_hull_wires_2d,
    make_polygon,
    offset_wire,
)
from bluemira.geometry.wire import BluemiraWire


def _cut_wall_below_x_point(shape: BluemiraWire, x_point_z: float) -> BluemiraWire:
    """
    Remove the parts of the wire below the given value in the z-axis.
    """
    # Create a box that surrounds the wall below the given z
    # coordinate, then perform a boolean cut to remove that portion
    # of the wall's shape.
    bounding_box = shape.bounding_box
    cut_box_points = np.array(
        [
            [bounding_box.x_min, 0, bounding_box.z_min],
            [bounding_box.x_min, 0, x_point_z],
            [bounding_box.x_max, 0, x_point_z],
            [bounding_box.x_max, 0, bounding_box.z_min],
            [bounding_box.x_min, 0, bounding_box.z_min],
        ]
    )
    cut_zone = make_polygon(cut_box_points, label="_shape_cut_exclusion")
    # For a single-null, we expect three 'pieces' from the cut: the
    # upper wall shape and the two separatrix legs
    pieces = boolean_cut(shape, [cut_zone])

    wall_piece = pieces[np.argmax([p.center_of_mass[2] for p in pieces])]
    if wall_piece.center_of_mass[2] < x_point_z:
        raise ValueError(
            "Could not cut wall shape below x-point. "
            "No parts of the wall found above x-point."
        )
    return wall_piece


class FirstWallBuilder(Builder):
    """
    Build a first wall with a divertor.

    This class runs the builders for the wall shape and the divertor,
    then combines the two.

    For a single-null plasma, the builder outputs a Component with the
    structure:

    .. code-block::

        first_wall (Component)
        └── xz (Component)
            └── wall (Component)
                └── wall_boundary (PhysicalComponent)
            └── divertor (Component)
                ├── inner_target (PhysicalComponent)
                ├── outer_target (PhysicalComponent)
                ├── dome (PhysicalComponent)
                ├── inner_baffle (PhysicalComponent)
                └── outer_baffle (PhysicalComponent)
    """

    COMPONENT_DIVERTOR = "divertor"
    COMPONENT_FIRST_WALL = "first_wall"
    COMPONENT_WALL = "wall"

    def __init__(
        self,
        params: Dict[str, Any],
        build_config: BuildConfig,
        equilibrium: Equilibrium,
        **kwargs,
    ):
        super().__init__(params, build_config, **kwargs)

        self.wall: Component
        self.divertor: Component

        self.equilibrium = equilibrium
        _, self.x_points = find_OX_points(
            self.equilibrium.x, self.equilibrium.z, self.equilibrium.psi()
        )
        self._wall_builder = self._init_wall_builder(params, build_config)
        self._divertor_builder = self._init_divertor_builder(params, build_config)

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        return super().reinitialise(params, **kwargs)

    def mock(self):
        """
        Create a basic shape for the wall's boundary.
        """
        pass

    def run(self):
        """Run the builder design problem."""
        pass

    def build(self) -> Component:
        """
        Build the component.
        """
        self.wall = self._build_wall()
        wall_shape = self.wall.get_component(WallBuilder.COMPONENT_WALL_BOUNDARY).shape
        self._divertor_builder.x_limits = [
            wall_shape.start_point().x[0],
            wall_shape.end_point().x[0],
        ]
        self.divertor = self._divertor_builder()

        first_wall = Component(self.name)
        first_wall.add_child(self.build_xz())
        return first_wall

    def build_xz(self) -> Component:
        """
        Build the component in the xz-plane.
        """
        parent_component = Component("xz")

        # Extract the xz components in the wall
        wall_xz = self.wall.get_component("xz")
        Component(
            self.COMPONENT_WALL,
            parent=parent_component,
            children=list(wall_xz.children),
        )

        # Extract the xz components in the divertor
        divertor_xz_component = self.divertor.get_component("xz")
        Component(
            self.COMPONENT_DIVERTOR,
            parent=parent_component,
            children=list(divertor_xz_component.children),
        )
        return parent_component

    def _init_wall_builder(
        self, params: Dict[str, Any], build_config: BuildConfig
    ) -> WallBuilder:
        """
        Initialize the wall builder.
        """
        build_config = deepcopy(build_config)
        build_config.update({"name": self.COMPONENT_WALL})
        keep_out_zone = self._make_wall_keep_out_zone(geom_offset=0.2, psi_n=1.05)
        return WallBuilder(
            params, build_config=build_config, keep_out_zone=keep_out_zone
        )

    def _init_divertor_builder(
        self, params: Dict[str, Any], build_config: BuildConfig
    ) -> DivertorBuilder:
        """
        Initialize the divertor builder.
        """
        build_config = deepcopy(build_config)
        build_config.update({"name": self.COMPONENT_DIVERTOR})
        build_config.pop("runmode", None)
        return DivertorBuilder(
            params,
            build_config,
            equilibrium=self.equilibrium,
            # no limits for now, we need to build the wall shape before
            # we know them
            x_limits=[],
        )

    def _build_wall(self):
        """
        Build the component for the wall, excluding the divertor.

        This uses the WallBuilder class to create a (optionally
        optimised) first wall shape. It then cuts the wall below the
        equilibrium's x-point, to make space for a divertor.
        """
        # Build a full, closed, wall shape
        wall = self._wall_builder()

        # Cut wall below x-point in xz, a divertor will be put in the
        # space
        wall_xz = wall.get_component("xz")
        wall_boundary = wall_xz.get_component(WallBuilder.COMPONENT_WALL_BOUNDARY)
        x_point_z = self.x_points[0].z
        if wall_boundary.shape.bounding_box.z_min >= x_point_z:
            raise BuilderError(
                "First wall boundary does not inclose separatrix x-point."
            )
        cut_shape = _cut_wall_below_x_point(wall_boundary.shape, x_point_z)

        # Replace the "uncut" wall boundary with the new shape
        wall_xz.prune_child(WallBuilder.COMPONENT_WALL_BOUNDARY)
        wall_xz.add_child(
            PhysicalComponent(WallBuilder.COMPONENT_WALL_BOUNDARY, cut_shape)
        )
        return wall

    def _make_wall_keep_out_zone(self, geom_offset, psi_n) -> BluemiraWire:
        """
        Create a "keep-out zone" to be used as a constraint in the
        wall shape optimiser.
        """
        geom_offset_zone = self._make_geometric_keep_out_zone(geom_offset)
        flux_surface_zone = self._make_flux_surface_keep_out_zone(psi_n)
        return convex_hull_wires_2d(
            [geom_offset_zone, flux_surface_zone], ndiscr=200, plane="xz"
        )

    def _make_geometric_keep_out_zone(self, offset: float) -> BluemiraWire:
        """
        Make a "keep-out zone" from a geometric offset of the LCFS.
        """
        lcfs = make_polygon(self.equilibrium.get_LCFS().xyz, closed=True)
        return offset_wire(lcfs, offset, join="arc")

    def _make_flux_surface_keep_out_zone(self, psi_n: float) -> BluemiraWire:
        """
        Make a "keep-out zone" from an equilibrium's flux surface.
        """
        flux_surface_zone = self.equilibrium.get_flux_surface(psi_n)
        flux_surface_zone = make_polygon(flux_surface_zone.xyz, closed=True)
        # Remove the "legs" from the keep-out zone, we want the wall to
        # intersect these
        return _cut_wall_below_x_point(flux_surface_zone, self.x_points[0].z)

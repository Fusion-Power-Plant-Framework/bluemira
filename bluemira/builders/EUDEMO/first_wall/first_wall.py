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
from typing import Any, Dict, Iterable, List

import numpy as np

from bluemira.base.builder import BuildConfig, Component
from bluemira.base.components import PhysicalComponent
from bluemira.builders.EUDEMO.first_wall.divertor import DivertorBuilder
from bluemira.builders.EUDEMO.first_wall.wall import WallBuilder
from bluemira.builders.shapes import Builder
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import find_OX_points
from bluemira.geometry.tools import boolean_cut, make_polygon, offset_wire
from bluemira.geometry.wire import BluemiraWire

_WALL_MODULE_REF = "bluemira.builders.EUDEMO.first_wall.wall"


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

    wall_piece = pieces[np.argmax([p.center_of_mass.z for p in pieces])]
    if wall_piece.center_of_mass.z < x_point_z:
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

        self.equilibrium = equilibrium
        _, self.x_points = find_OX_points(
            self.equilibrium.x, self.equilibrium.z, self.equilibrium.psi()
        )
        self.wall: Component = self._build_wall(params, build_config)
        wall_shape = self.wall.get_component(WallBuilder.COMPONENT_WALL_BOUNDARY).shape
        self.divertor: Component = self._build_divertor(
            params,
            build_config,
            [wall_shape.start_point()[0], wall_shape.end_point()[0]],
        )

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

    def _build_wall(self, params: Dict[str, Any], build_config: BuildConfig):
        """
        Build the component for the wall, excluding the divertor.

        This uses the WallBuilder class to create a (optionally
        optimised) first wall shape. It then cuts the wall below the
        equilibrium's x-point, to make space for a divertor.
        """
        build_config = deepcopy(build_config)
        build_config.update(
            {
                "algorithm_name": "SLSQP",
                "class": f"{_WALL_MODULE_REF}::WallBuilder",
                "label": self.COMPONENT_WALL,
                "name": self.COMPONENT_WALL,
                "opt_conditions": {
                    "ftol_rel": 1e-6,
                    "xtol_rel": 1e-8,
                    "xtol_abs": 1e-8,
                    "max_eval": 100,
                },
                "param_class": f"{_WALL_MODULE_REF}::WallPrincetonD",
                "problem_class": f"{_WALL_MODULE_REF}::MinimiseLength",
            }
        )

        # Keep-out zone to constrain the wall around the plasma
        keep_out_zones = self._make_wall_keep_out_zones(geom_offset=0.2, psi_n=1.05)

        # Build a full, closed, wall shape
        builder = WallBuilder(
            params, build_config=build_config, keep_out_zones=keep_out_zones
        )
        wall = builder()

        # Cut wall below x-point in xz, a divertor will be put in the
        # space
        wall_xz = wall.get_component("xz")
        wall_boundary = wall_xz.get_component(WallBuilder.COMPONENT_WALL_BOUNDARY)
        x_point_z = self.x_points[0].z
        cut_shape = _cut_shape_in_z(wall_boundary.shape, x_point_z)

        # Replace the "uncut" wall boundary with the new shape
        wall_xz.prune_child(WallBuilder.COMPONENT_WALL_BOUNDARY)
        wall_xz.add_child(
            PhysicalComponent(WallBuilder.COMPONENT_WALL_BOUNDARY, cut_shape)
        )
        return wall

    def _build_divertor(
        self, params: Dict[str, Any], build_config, x_lims: Iterable[float]
    ) -> Component:
        """
        Build divertor component below the first x-point in the
        separatrix of the equilibrium.
        """
        # This currently only supports building a divertor at the lower
        # end of the plasma. We will need to add a 'Location' switch
        # here when we start supporting double-null plasmas
        build_config = deepcopy(build_config)
        build_config.update({"name": self.COMPONENT_DIVERTOR})
        builder = DivertorBuilder(params, build_config, self.equilibrium, x_lims)
        return builder()

    def _make_wall_keep_out_zones(self, geom_offset, psi_n) -> List[BluemiraWire]:
        """
        Create a "keep-out zone" to be used as a constraint in the
        shape optimiser.
        """
        geom_offset_zone = self._make_geometric_keep_out_zone(geom_offset)
        flux_surface_zone = self._make_flux_surface_keep_out_zone(psi_n)
        return [geom_offset_zone, flux_surface_zone]

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

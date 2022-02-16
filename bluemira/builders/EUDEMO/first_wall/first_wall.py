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
from typing import Any, Dict, Iterable

import numpy as np

from bluemira.base.builder import BuildConfig, Component
from bluemira.base.components import PhysicalComponent
from bluemira.builders.EUDEMO.first_wall.divertor import DivertorBuilder
from bluemira.builders.EUDEMO.first_wall.wall import WallBuilder
from bluemira.builders.shapes import Builder
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import find_OX_points
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.tools import boolean_cut, make_polygon
from bluemira.geometry.wire import BluemiraWire

_WALL_MODULE_REF = "bluemira.builders.EUDEMO.first_wall.wall"


def _cut_shape_in_z(shape: BluemiraWire, z_max: float):
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
            [bounding_box.x_min, 0, z_max],
            [bounding_box.x_max, 0, z_max],
            [bounding_box.x_max, 0, bounding_box.z_min],
            [bounding_box.x_min, 0, bounding_box.z_min],
        ]
    )
    cut_zone = make_polygon(cut_box_points, label="_shape_cut_exclusion")
    return boolean_cut(shape, [cut_zone])[0]


class FirstWallBuilder(Builder):
    """
    Build a first wall with a divertor.

    This class runs the builders for the wall shape and the divertor,
    then combines the two.

    For a single-null plasma, the builder outputs a Component with the
    structure:

    first_wall (Component)
    └── xz (Component)
        ├── wall (PhysicalComponent)
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
        self.wall: PhysicalComponent = self._build_wall(params, build_config)

        wall_shape: BluemiraWire = self.wall.shape
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
        first_wall = Component(FirstWallBuilder.COMPONENT_FIRST_WALL)
        first_wall.add_child(self.build_xz())
        return first_wall

    def build_xz(self) -> Component:
        """
        Build the component in the xz-plane.
        """
        parent_component = Component("xz")

        # Extract the xz components in the wall
        # TODO(hsaunders1904): add "xz" to wall component
        parent_component.add_child(self.wall)

        # Extract the xz components in the divertor
        divertor_xz_component = self.divertor.get_component("xz")
        Component(
            self.COMPONENT_DIVERTOR,
            parent=parent_component,
            children=divertor_xz_component.children,
        )
        return parent_component

    def _build_wall(self, params: Dict[str, Any], build_config: BuildConfig):
        """
        Build the component for the wall, excluding the divertor.
        """
        build_config = deepcopy(build_config)
        build_config.update(
            {
                "class": f"{_WALL_MODULE_REF}::WallBuilder",
                "param_class": f"{_WALL_MODULE_REF}::WallPolySpline",
                "problem_class": f"{_WALL_MODULE_REF}::MinimiseLength",
                "label": self.COMPONENT_WALL,
                "name": self.COMPONENT_WALL,
            }
        )

        # Keep-out-zone to stop the wall intersecting the plasma
        keep_out_zone = self._make_wall_keep_out_zone()
        # Build a full, closed, wall shape
        builder = WallBuilder(
            params, build_config=build_config, keep_out_zones=(keep_out_zone,)
        )
        wall = builder()
        wall_boundary = wall.get_component(WallBuilder.COMPONENT_WALL_BOUNDARY)
        # Cut wall below x-point, a divertor will be put in the space
        x_point_z = self.x_points[0].z
        cut_shape = _cut_shape_in_z(wall_boundary.shape, x_point_z)
        return PhysicalComponent(FirstWallBuilder.COMPONENT_WALL, cut_shape)

    def _build_divertor(
        self, params: Dict[str, Any], build_config, x_lims: Iterable[float]
    ) -> Component:
        """
        Build the divertor component.
        """
        build_config = deepcopy(build_config)
        build_config.update({"name": self.COMPONENT_DIVERTOR})
        builder = DivertorBuilder(params, build_config, self.equilibrium, x_lims)
        return builder()

    def _make_wall_keep_out_zone(self) -> BluemiraWire:
        """
        Create a "keep-out-zone" to be used as a constraint in the
        shape optimiser
        """
        # The keep-out-zone is generated from the flux surface of the
        # separatrix above the x-point
        coords = self.equilibrium.get_separatrix().xyz
        coords = coords[:, coords[2] > self.x_points[0].z]
        return make_polygon(coords, closed=True)

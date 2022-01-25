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

from typing import Any, Dict

import numpy as np

from bluemira.base.builder import BuildConfig, Component
from bluemira.builders.EUDEMO.first_wall import ClosedFirstWallBuilder
from bluemira.builders.shapes import ParameterisedShapeBuilder
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.tools import boolean_cut, make_polygon
from bluemira.geometry.wire import BluemiraWire


class FirstWallBuilder(ParameterisedShapeBuilder):
    """
    Build a first wall with a divertor.

    This class runs the builders for the first wall shape and the
    divertor, then combines the two.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        build_config: BuildConfig,
        x_point: np.ndarray,
        # separatrix: Loop,
        **kwargs,
    ):
        super().__init__(params, build_config, **kwargs)

        self.x_point = x_point  # anything below z is divertor, anything above is wall
        # self.separatrix: Loop = separatrix

        self.wall_part: Component = self._build_wall_no_divertor(params, build_config)
        self.divertor: Component = None

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        return super().reinitialise(params, **kwargs)

    def mock(self):
        """
        Create a basic shape for the wall's boundary.
        """
        self.boundary = self._shape.create_shape()

    def build(self, **kwargs) -> Component:
        """
        Build the component.
        """
        pass

    def _build_wall_no_divertor(self, params: Dict[str, Any], build_config: BuildConfig):
        """
        Build the component for the wall, excluding the divertor.
        """
        builder = ClosedFirstWallBuilder(params, build_config=build_config)
        wall = builder(params)

        wall_shape: BluemiraGeo = wall.get_component("first_wall").shape
        z_max = self.x_point[1]
        wall.get_component("first_wall").shape = self._cut_shape_in_z(wall_shape, z_max)
        return wall

    def _cut_shape_in_z(self, shape: BluemiraWire, z_max: float):
        """
        Remove the parts of the wire below the given value in the z-axis.
        """
        # Create a box that surrounds the wall below the given z coordinate,
        # then perform a boolean cut to remove that portion of the wall's shape.
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

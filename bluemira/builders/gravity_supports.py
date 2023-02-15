# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Gravity support builders
"""

from dataclasses import dataclass
from typing import Dict, Type, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.error import BuilderError
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, extrude_shape, make_polygon
from bluemira.geometry.wire import BluemiraWire


@dataclass
class ITERGravitySupportBuilderParams(ParameterFrame):
    """
    ITER-like gravity support parameters
    """

    x_g_support: Parameter[float]
    z_gs: Parameter[float]
    tf_wp_width: Parameter[float]
    tf_wp_depth: Parameter[float]
    tk_tf_side: Parameter[float]
    tf_gs_tk_plate: Parameter[float]
    tf_gs_g_plate: Parameter[float]
    tf_gs_base_depth: Parameter[float]


class ITERGravitySupportBuilder(Builder):
    """
    ITER-like gravity support builder
    """

    param_cls: Type[ITERGravitySupportBuilderParams] = ITERGravitySupportBuilderParams

    def __init__(
        self,
        params: Union[ITERGravitySupportBuilderParams, Dict],
        build_config: Dict,
        tf_xz_keep_out_zone: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.tf_xz_keep_out_zone = tf_xz_keep_out_zone

    def build(self) -> Component:
        """
        Build the ITER-like gravity support component.
        """
        return self.build_xyz()

    def build_xyz(
        self,
    ) -> PhysicalComponent:
        """
        Build the x-y-z components of the ITER-like gravity support.

        """
        shape_list = []
        # First, project upwards at the radius of the GS into the keep-out-zone
        # and get a x-z face of the boolean difference.

        # Get the square width
        width = self.params.tf_wp_depth + 2 * self.params.tk_tf_side
        x_inner_line = self.params.x_g_support - 0.5 * width
        x_outer_line = self.params.x_g_support + 0.5 * width
        z_min = self.tf_xz_keep_out_zone.bounding_box.z_min
        z_max = self.tf_xz_keep_out_zone.bounding_box.z_max
        z_max = z_min + 0.5 * (z_max - z_min)
        z_min -= 10.0  # just some offset downwards
        x_min = self.tf_xz_keep_out_zone.bounding_box.x_min - 0.5 * width
        x_max = self.tf_xz_keep_out_zone.bounding_box.x_max + 0.5 * width

        if (self.params.x_g_support < x_min) | (self.params.x_g_support > x_max):
            raise BuilderError(
                "The gravity support footprint is not contained within the provided TF coil geometry!"
            )

        cut_box = make_polygon(
            {
                "x": [x_inner_line, x_inner_line, x_outer_line, x_outer_line],
                "y": 0,
                "z": [z_min, z_max, z_max, z_min],
            },
            closed=True,
        )

        cut_result = boolean_cut(self.tf_xz_keep_out_zone, cut_box)

        if cut_result is None:
            raise BuilderError(
                "Boolean cutting returned nothing... check your geometry please."
            )

        intersection_wire = sorted(cut_result, key=lambda wire: wire.length)[0]
        v1 = intersection_wire.start_point()
        v4 = intersection_wire.end_point()
        if v1.x > v4.x:
            v1, v4 = v4, v1

        z_block_lower = v1.z - 5 * self.params.tf_gs_tk_plate
        v2 = Coordinates(np.array([v1.x, 0, z_block_lower]))
        v4 = intersection_wire.end_point()
        v3 = Coordinates(np.array([v4.x, 0, z_block_lower]))

        points = np.concatenate([v1.xyz, v2.xyz, v3.xyz, v4.xyz], axis=1)
        closing_wire = make_polygon(points, closed=False)
        face = BluemiraFace(BluemiraWire([intersection_wire, closing_wire]))

        # Then extrude that face in both directions to get the connection block
        face.translate(vector=(0, -0.5 * width, 0))
        block = extrude_shape(face, vec=(0, width, 0))
        shape_list.append(block)

        # Next, make the plates in a linear pattern, in y-z, along x
        yz_profile = Coordinates(
            {
                "x": 4 * [x_inner_line],
                "y": [
                    -0.5 * width,
                    0.5 * width,
                    0.5 * self.params.tf_gs_base_depth,
                    -0.5 * self.params.tf_gs_base_depth,
                ],
                "z": [z_block_lower, z_block_lower, self.params.z_gs, self.params.z_gs],
            },
        )
        yz_profile = make_polygon(yz_profile, closed=True)

        plating_width = x_outer_line - x_inner_line
        plate_and_gap = self.params.tf_gs_g_plate + self.params.tf_gs_tk_plate
        n_plates = (plating_width + self.params.tf_gs_g_plate) / plate_and_gap
        total_width = (
            int(n_plates) * self.params.tf_gs_tk_plate
            + (int(n_plates) - 1) * self.params.tf_gs_g_plate
        )
        delta_width = plating_width - total_width
        yz_profile.translate(vector=(0.5 * delta_width, 0, 0))

        plate = extrude_shape(
            BluemiraFace(yz_profile), vec=(self.params.tf_gs_tk_plate, 0, 0)
        )
        shape_list.append(plate)
        for _ in range(int(n_plates) - 1):
            plate = plate.deepcopy()
            plate.translate(
                vector=(self.params.tf_gs_g_plate + self.params.tf_gs_tk_plate, 0, 0)
            )
            shape_list.append(plate)

        # Finally, make the floor block
        xz_profile = Coordinates(
            {
                "x": [x_inner_line, x_inner_line, x_outer_line, x_outer_line],
                "y": [
                    -0.5 * self.params.tf_gs_base_depth,
                    0.5 * self.params.tf_gs_base_depth,
                    0.5 * self.params.tf_gs_base_depth,
                    -0.5 * self.params.tf_gs_base_depth,
                ],
                "z": 4 * [self.params.z_gs],
            },
        )
        xz_profile = make_polygon(xz_profile, closed=True)
        floor_block = extrude_shape(
            xz_profile, vec=(0, 0, -5 * self.params.tf_gs_tk_plate)
        )
        shape_list.append(floor_block)
        shape = BluemiraCompound(shape_list)
        component = PhysicalComponent("ITER-like gravity support", shape)
        component.display_cad_options.color = BLUE_PALETTE["TF"][2]
        return component


if __name__ == "__main__":
    from bluemira.display import show_cad
    from bluemira.geometry.parameterisations import PrincetonD

    my_test_params = ITERGravitySupportBuilderParams(
        x_g_support=10,
        z_gs=-15,
        tf_wp_depth=1.4,
        tf_wp_width=0.8,
        tk_tf_side=0.05,
        tf_gs_tk_plate=0.025,
        tf_gs_g_plate=0.025,
        tf_gs_base_depth=2.4,
    )

    my_dummy_tf = PrincetonD()
    my_dummy_tf.adjust_variable("x1", value=3, lower_bound=2, upper_bound=4)
    my_dummy_tf.adjust_variable("x2", value=15, lower_bound=2, upper_bound=24)
    my_dummy_tf_xz_koz = my_dummy_tf.create_shape()

    my_builder = ITERGravitySupportBuilder(my_test_params, {}, my_dummy_tf_xz_koz)

    component = my_builder.build()
    component.show_cad()

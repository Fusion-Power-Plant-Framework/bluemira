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
PF coil support builders
"""

from dataclasses import dataclass
from typing import Dict, Type, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_cut,
    distance_to,
    extrude_shape,
    make_polygon,
    offset_wire,
)
from bluemira.geometry.wire import BluemiraWire


@dataclass
class PFCoilSupportBuilderParams(ParameterFrame):
    """
    PF coil support parameters
    """

    tf_wp_width: Parameter[float]
    tf_wp_depth: Parameter[float]
    tk_tf_side: Parameter[float]
    pf_s_tk_plate: Parameter[float]
    pf_s_n_plate: Parameter[int]
    pf_s_g: Parameter[float]


class PFCoilSupportBuilder(Builder):
    """
    PF coil support builder
    """

    param_cls: Type[PFCoilSupportBuilderParams] = PFCoilSupportBuilderParams

    def __init__(
        self,
        params: Union[PFCoilSupportBuilderParams, Dict],
        build_config: Dict,
        tf_xz_keep_out_zone: BluemiraWire,
        pf_coil_xz: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.tf_xz_keep_out_zone = tf_xz_keep_out_zone
        self.pf_coil_xz = pf_coil_xz

    def build(self) -> Component:
        """
        Build the PF coil support component.
        """
        return self.build_xyz()

    def _get_intersection_wire(self, width):
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
        return intersection_wire

    def _build_support_block(self):
        bb = self.pf_coil_xz.bounding_box
        width = self.params.tf_wp_depth + 2 * self.params.tk_tf_side
        half_width = 0.5 * width
        alpha = np.arcsin(half_width / bb.x_min)
        inner_dr = half_width * np.tan(alpha)

        beta = np.arcin(half_width / bb.x_max)
        outer_dr = half_width * np.tan(beta)

        x_min = bb.x_min - self.params.pf_s_g - inner_dr
        x_max = bb.x_max + self.params.pf_s_g + outer_dr
        z_min = bb.z_min
        z_max = bb.z_max
        box_inner = make_polygon(
            {
                "x": [x_min, x_max, x_max, x_min],
                "y": 0,
                "z": [z_min, z_min, z_max, z_max],
            },
            closed=True,
        )
        box_outer = offset_wire(box_inner, self.params.pf_s_tk_plate)
        face = BluemiraFace([box_outer, box_inner])
        return face

    def _get_support_point_angle(self, support_face: BluemiraFace):
        bb = support_face.bounding_box
        x_2 = 0.5 * (bb.x_max - bb.x_min)
        z_2 = bb.z_min
        z_3 = bb.z_max

        distance = np.inf
        best_angle = None
        start_point = None
        end_point = None
        for point in [[x_2, z_2], [x_2, z_3]]:
            for angle in [np.pi, -np.pi, -0.75 * np.pi, 0.75 * np.pi]:
                x_out = point[0] + np.cos(angle) * 100
                z_out = point[1] + np.sin(angle) * 100
                line = make_polygon(
                    {"x": [point[0], x_out], "y": 0, "z": [point[1], z_out]}
                )
                d_intersection, info = distance_to(self.tf_xz_keep_out_zone, line)
                if d_intersection == 0.0:
                    d = np.hypot(info[0][0], info[0][2])
                    if d < distance:
                        distance = d
                        best_angle = angle
                        start_point = point
                        end_point = info[0]

        if distance == np.inf:
            raise BuilderError("No intersections found!")
        return start_point, end_point, best_angle

    def build_xyz(
        self,
    ) -> PhysicalComponent:
        """
        Build the x-y-z components of the ITER-like gravity support.

        """
        shape_list = []
        # First build the support block around the PF coil
        support_face = self._build_support_xs()
        width = self.params.tf_wp_depth + 2 * self.params.tk_tf_size
        support_block = extrude_shape(support_face, vec=(0, width, 0))
        shape_list.append(support_block)

        # Then, project sideways to find the minimum distance from a support point
        # to the TF coil
        start_point, end_point, angle = self._get_support_point_angle(support_face)

        bb = support_face.bounding_box
        if start_point[1] < end_point[1]:
            # Then we're connecting the coil upwards from the top of the support block
            x_out1 = bb.x_min + np.cos(angle) * 10
            z_out1 = bb.z_min + np.sin(angle) * 10
            x_out2 = bb.x_max + np.cos(angle) * 10
            z_out2 = bb.z_min + np.sin(angle) * 10
            v1 = [bb.x_min, 0, bb.z_min]
            v2 = [bb.x_max, 0, bb.z_min]
            v3 = [x_out2, 0, z_out2]
            v4 = [x_out1, 0, z_out1]

        else:
            # Then we're connecting the coil downwards from the bottom of the support block
            x_out1 = bb.x_min + np.cos(angle) * 10
            z_out1 = bb.z_max + np.sin(angle) * 10
            x_out2 = bb.x_max + np.cos(angle) * 10
            z_out2 = bb.z_max + np.sin(angle) * 10
            v1 = [bb.x_min, 0, bb.z_max]
            v2 = [bb.x_max, 0, bb.z_max]
            v3 = [x_out2, 0, z_out2]
            v4 = [x_out1, 0, z_out1]

        cut_box = make_polygon([v1, v2, v3, v4])
        intersection_wire = sorted(
            boolean_cut(self.tf_xz_keep_out_zone, cut_box), key=lambda wire: -wire.length
        )[0]
        v3 = intersection_wire.start_point().xyz
        v4 = intersection_wire.end_point().xyz

        xz_profile = BluemiraFace(
            make_polygon(
                {
                    "x": [v1[0], v2[0], v3[0], v4[0]],
                    "y": 0,
                    "z": [v1[2], v2[2], v3[2], v4[2]],
                },
                closed=True,
            )
        )

        # Calculate the rib gap width
        total_rib_tk = self.params.pf_s_n_plates * self.params.pf_s_tk_plate
        if total_rib_tk >= width:
            bluemira_warn(
                "PF coil support rib thickness and number exceed available thickness! You're getting a solid block instead"
            )
            total_rib_tk = width

        # Then make the connection ribs

        shape = BluemiraCompound(shape_list)
        component = PhysicalComponent("PF coil support", shape)
        component.display_cad_options.color = BLUE_PALETTE["TF"][3]
        return component


if __name__ == "__main__":
    from bluemira.geometry.parameterisations import PictureFrame, PrincetonD

    my_test_params = PFCoilSupportBuilderParams(
        tf_wp_depth=1.4,
        tf_wp_width=0.8,
        tk_tf_side=0.05,
        pf_s_tk_plate=0.3,
        pf_s_n_plate=3,
        pf_s_g=0.05,
    )

    my_dummy_tf = PrincetonD()
    my_dummy_tf.adjust_variable("x1", value=3, lower_bound=2, upper_bound=4)
    my_dummy_tf.adjust_variable("x2", value=15, lower_bound=2, upper_bound=24)
    my_dummy_tf_xz_koz = my_dummy_tf.create_shape()

    my_dummy_pf = PictureFrame()
    my_dummy_pf.adjust_variable("ri", value=0.1, lower_bound=0, upper_bound=1.0)
    my_dummy_pf.adjust_variable("ro", value=0.1, lower_bound=0, upper_bound=1.0)
    my_dummy_pf.adjust_variable("x1", value=10, lower_bound=0, upper_bound=11.0)
    my_dummy_pf.adjust_variable("x2", value=11, lower_bound=0, upper_bound=13.0)
    my_dummy_pf.adjust_variable("z1", value=10, lower_bound=0, upper_bound=11.0)
    my_dummy_pf.adjust_variable("z2", value=9, lower_bound=0, upper_bound=11.0)
    my_dummy_pf_xz = my_dummy_pf.create_shape()

    my_builder = PFCoilSupportBuilder(
        my_test_params, {}, my_dummy_tf_xz_koz, my_dummy_pf_xz
    )

    component = my_builder.build()
    component.show_cad()

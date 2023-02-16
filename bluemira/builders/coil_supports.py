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

    def _build_support_xs(self):
        bb = self.pf_coil_xz.bounding_box
        width = self.params.tf_wp_depth + 2 * self.params.tk_tf_side
        half_width = 0.5 * width
        alpha = np.arcsin(half_width / bb.x_min)
        inner_dr = half_width * np.tan(alpha)

        beta = np.arcsin(half_width / bb.x_max)
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
        bb = support_face.boundary[0].bounding_box
        x_2 = bb.x_min + 0.5 * (bb.x_max - bb.x_min)
        z_2 = bb.z_min
        z_3 = bb.z_max

        distance = np.inf
        best_angle = None
        start_point = None
        end_point = None
        for point, sign in zip([[x_2, z_2], [x_2, z_3]], [-1, 1]):
            for angle in [0.5 * np.pi, 2 / 3 * np.pi]:
                x_out = point[0] + np.cos(sign * angle) * 100
                z_out = point[1] + np.sin(sign * angle) * 100
                line = make_polygon(
                    {"x": [point[0], x_out], "y": 0, "z": [point[1], z_out]}
                )
                d_intersection, info = distance_to(self.tf_xz_keep_out_zone, line)

                distances = []
                for inter_point_pair in info:
                    dist = np.hypot(
                        inter_point_pair[0][0] - point[0],
                        inter_point_pair[0][2] - point[1],
                    )
                    print(f"{dist=}")
                    distances.append(dist)
                i_min = np.argmin(distances)
                p_inter = info[i_min][0]

                # p_inter = info[0][0]
                print(f"{info=}")
                print(f"{d_intersection=}")
                if np.isclose(d_intersection, 0.0):
                    d = np.hypot(point[0] - p_inter[0], point[1] - p_inter[2])
                    print(f"{d=}")
                    if d < distance:
                        distance = d
                        best_angle = sign * angle
                        start_point = point
                        end_point = p_inter

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
        width = self.params.tf_wp_depth + 2 * self.params.tk_tf_side
        support_block = extrude_shape(support_face, vec=(0, width, 0))
        shape_list.append(support_block)

        # Then, project sideways to find the minimum distance from a support point
        # to the TF coil
        start_point, end_point, angle = self._get_support_point_angle(support_face)

        bb = support_face.bounding_box
        if start_point[1] > end_point[1]:
            # Then we're connecting the coil upwards from the top of the support
            # block
            x_out1 = bb.x_min + np.cos(angle) * 2
            z_out1 = bb.z_min + np.sin(angle) * 2
            x_out2 = bb.x_max + np.cos(angle) * 2
            z_out2 = bb.z_min + np.sin(angle) * 2
            v1 = np.array([bb.x_min, 0, bb.z_min])
            v2 = np.array([bb.x_max, 0, bb.z_min])
            v3 = np.array([x_out2, 0, z_out2])
            v4 = np.array([x_out1, 0, z_out1])

        else:
            # Then we're connecting the coil downwards from the bottom of the support
            # block
            x_out1 = bb.x_min + np.cos(angle) * 2
            z_out1 = bb.z_max + np.sin(angle) * 2
            x_out2 = bb.x_max + np.cos(angle) * 2
            z_out2 = bb.z_max + np.sin(angle) * 2
            v1 = np.array([bb.x_min, 0, bb.z_max])
            v2 = np.array([bb.x_max, 0, bb.z_max])
            v3 = np.array([x_out2, 0, z_out2])
            v4 = np.array([x_out1, 0, z_out1])

        cut_box = make_polygon([v1, v2, v3, v4], closed=True)
        intersection_wire = sorted(
            boolean_cut(self.tf_xz_keep_out_zone, cut_box), key=lambda wire: wire.length
        )[0]
        v3 = intersection_wire.start_point().xyz.T[0]
        v4 = intersection_wire.end_point().xyz.T[0]
        d1 = np.sqrt(np.sum((v3 - v1) ** 2))
        d2 = np.sqrt(np.sum((v4 - v1) ** 2))
        if d1 > d2:
            v3, v4 = v4, v3

        closing_wire = make_polygon(
            {
                "x": [v3[0], v1[0], v2[0], v4[0]],
                "y": 0,
                "z": [v3[2], v1[2], v2[2], v4[2]],
            },
            closed=False,
        )
        xz_profile = BluemiraFace(BluemiraWire([intersection_wire, closing_wire]))
        # show_cad([intersection_wire, closing_wire])

        # Calculate the rib gap width and make the ribs
        total_rib_tk = self.params.pf_s_n_plate * self.params.pf_s_tk_plate
        if total_rib_tk >= width:
            bluemira_warn(
                "PF coil support rib thickness and number exceed available thickness! You're getting a solid block instead"
            )
            gap_size = 0
            rib_block = extrude_shape(xz_profile, vec=(0, width, 0))
            shape_list.append(rib_block)
        else:
            gap_size = (width - total_rib_tk) / (self.params.pf_s_n_plate - 1)
            rib = extrude_shape(xz_profile, vec=(0, self.params.pf_s_tk_plate, 0))
            shape_list.append(rib)
            for _ in range(self.params.pf_s_n_plate - 1):
                rib = rib.deepcopy()
                rib.translate(vector=(0, self.params.pf_s_tk_plate + gap_size, 0))
                shape_list.append(rib)

        shape = BluemiraCompound(shape_list)
        shape.translate(vector=(0, -0.5 * width, 0))
        component = PhysicalComponent("PF coil support", shape)
        component.display_cad_options.color = BLUE_PALETTE["TF"][2]
        return component


if __name__ == "__main__":
    from bluemira.display import show_cad
    from bluemira.geometry.parameterisations import PictureFrame, PrincetonD
    from bluemira.geometry.tools import revolve_shape

    my_test_params = PFCoilSupportBuilderParams(
        tf_wp_depth=1.4,
        tf_wp_width=0.8,
        tk_tf_side=0.05,
        pf_s_tk_plate=0.15,
        pf_s_n_plate=4,
        pf_s_g=0.05,
    )

    my_dummy_tf = PrincetonD()
    my_dummy_tf.adjust_variable("x1", value=3, lower_bound=2, upper_bound=4)
    my_dummy_tf.adjust_variable("x2", value=15, lower_bound=2, upper_bound=24)
    my_dummy_tf.adjust_variable("dz", value=0, lower_bound=-10, upper_bound=24)
    my_dummy_tf_xz_koz = my_dummy_tf.create_shape()
    inner_wire = offset_wire(my_dummy_tf_xz_koz, -1.0)
    tf_face = BluemiraFace([my_dummy_tf_xz_koz, inner_wire])
    tf = extrude_shape(
        tf_face, (0, my_test_params.tf_wp_depth + 2 * my_test_params.tk_tf_side, 0)
    )
    tf.translate((0, -0.5 * my_test_params.tf_wp_depth, 0))
    tf = PhysicalComponent("TF coil", tf)
    tf.display_cad_options.color = BLUE_PALETTE["TF"][0]

    def make_dummy_pf(xc, zc, dxc, dzc):
        """
        Flake8
        """
        my_dummy_pf = PictureFrame()
        my_dummy_pf.adjust_variable("ri", value=0.1, lower_bound=0, upper_bound=np.inf)
        my_dummy_pf.adjust_variable("ro", value=0.1, lower_bound=0, upper_bound=np.inf)
        my_dummy_pf.adjust_variable(
            "x1", value=xc - dxc, lower_bound=0, upper_bound=np.inf
        )
        my_dummy_pf.adjust_variable(
            "x2", value=xc + dxc, lower_bound=0, upper_bound=np.inf
        )
        my_dummy_pf.adjust_variable(
            "z1", value=zc + dzc, lower_bound=-np.inf, upper_bound=np.inf
        )
        my_dummy_pf.adjust_variable(
            "z2", value=zc - dzc, lower_bound=-np.inf, upper_bound=np.inf
        )
        return my_dummy_pf.create_shape()

    pf_shapes = []
    pf_shapes.append(make_dummy_pf(6, 12.5, 0.5, 0.5))
    pf_shapes.append(make_dummy_pf(10, 11, 0.5, 0.5))
    pf_shapes.append(make_dummy_pf(14.5, 6, 0.5, 0.5))
    pf_shapes.append(make_dummy_pf(14.5, -6, 0.5, 0.5))
    pf_shapes.append(make_dummy_pf(10, -11, 0.5, 0.5))
    pf_shapes.append(make_dummy_pf(6, -12.5, 0.5, 0.5))

    guff = Component("wtf")
    guff.add_child(tf)
    for i, pf in enumerate(pf_shapes):
        my_builder = PFCoilSupportBuilder(my_test_params, {}, my_dummy_tf_xz_koz, pf)
        component = my_builder.build()
        component.name = component.name + f" {i}"
        guff.add_child(component)
        pf_coil_shape = revolve_shape(BluemiraFace(pf), degree=360)
        pf_coil = PhysicalComponent(f"PF coil {i}", pf_coil_shape)
        pf_coil.display_cad_options.color = BLUE_PALETTE["PF"][1]
        guff.add_child(pf_coil)

    guff.show_cad()

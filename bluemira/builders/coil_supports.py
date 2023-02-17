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
Coil support builders
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
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    distance_to,
    extrude_shape,
    make_polygon,
    offset_wire,
    slice_shape,
)
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

    Parameters
    ----------
    params:
        Parameters to use
    build_config:
        Build config to use
    tf_kz_keep_out_zone:
        TF coil wire keep-out-zone for the outer edge of the TF coil (including casing)
        Note that this should be on the y=0 plane.
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
        xyz = self.build_xyz()
        return self.component_tree([self.build_xz(xyz)], self.build_xy(), [xyz])

    def build_xz(self, xyz_component):
        """
        Build the x-z component of the ITER-like gravity support.
        """
        xz_plane = BluemiraPlane((0, 0, 0), (0, 1, 0))
        slice_result = slice_shape(xyz_component.shape, xz_plane)

        # Process UGLY SLICE
        wires = sorted(slice_result, key=lambda wire: wire.length)
        wire_list = [wires.pop()]
        wire_list.extend(wires)
        shape = BluemiraFace(wire_list)

        component = PhysicalComponent("ITER-like gravity support", shape)
        component.display_cad_options.color = BLUE_PALETTE["TF"][2]
        component.plot_options.face_options["color"] = BLUE_PALETTE["TF"][2]
        return component

    def build_xy(self):
        """
        Build the x-y component of the ITER-like gravity support.
        """
        pass

    def _get_intersection_wire(self, width):
        x_g_support = self.params.x_g_support.value
        x_inner_line = x_g_support - 0.5 * width
        x_outer_line = x_g_support + 0.5 * width
        z_min = self.tf_xz_keep_out_zone.bounding_box.z_min
        z_max = self.tf_xz_keep_out_zone.bounding_box.z_max
        z_max = z_min + 0.5 * (z_max - z_min)
        x_min = self.tf_xz_keep_out_zone.bounding_box.x_min + 0.5 * width
        x_max = self.tf_xz_keep_out_zone.bounding_box.x_max - 0.5 * width
        if (x_g_support < x_min) | (x_g_support > x_max):
            raise BuilderError(
                "The gravity support footprint is not contained within the provided TF coil geometry!"
            )

        if (self.params.z_gs.value - 6 * self.params.tf_gs_tk_plate.value) > z_min:
            raise BuilderError(
                "The gravity support floor is not lower than where the TF coil is!"
            )
        z_min = self.params.z_gs.value - 6 * self.params.tf_gs_tk_plate.value

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

        return sorted(cut_result, key=lambda wire: wire.length)[0]

    def _make_connection_block(self, width, v1, v4, intersection_wire):
        """
        Make the connection block of the gravity support with the TF coil
        """
        z_block_lower = min(v1.z[0], v4.z[0]) - 5 * self.params.tf_gs_tk_plate.value
        v2 = Coordinates(np.array([v1.x[0], 0, z_block_lower]))
        v3 = Coordinates(np.array([v4.x[0], 0, z_block_lower]))

        points = np.concatenate([v1.xyz, v2.xyz, v3.xyz, v4.xyz], axis=1)
        closing_wire = make_polygon(points, closed=False)
        face = BluemiraFace(BluemiraWire([intersection_wire, closing_wire]))

        # Then extrude that face in both directions to get the connection block
        face.translate(vector=(0, -0.5 * width, 0))
        return extrude_shape(face, vec=(0, width, 0))

    def _make_plates(self, width, v1x, v4x, z_block_lower):
        """
        Make the gravity support vertical plates
        """
        plate_list = []
        yz_profile = Coordinates(
            {
                "x": 4 * [v1x],
                "y": [
                    -0.5 * width,
                    0.5 * width,
                    0.5 * self.params.tf_gs_base_depth.value,
                    -0.5 * self.params.tf_gs_base_depth.value,
                ],
                "z": [
                    z_block_lower,
                    z_block_lower,
                    self.params.z_gs.value,
                    self.params.z_gs.value,
                ],
            },
        )
        yz_profile = make_polygon(yz_profile, closed=True)

        plating_width = v4x - v1x
        plate_and_gap = (
            self.params.tf_gs_g_plate.value + self.params.tf_gs_tk_plate.value
        )
        n_plates = (plating_width + self.params.tf_gs_g_plate.value) / plate_and_gap
        total_width = (
            int(n_plates) * self.params.tf_gs_tk_plate.value
            + (int(n_plates) - 1) * self.params.tf_gs_g_plate.value
        )
        delta_width = plating_width - total_width
        yz_profile.translate(vector=(0.5 * delta_width, 0, 0))

        plate = extrude_shape(
            BluemiraFace(yz_profile), vec=(self.params.tf_gs_tk_plate.value, 0, 0)
        )
        plate_list.append(plate)
        for _ in range(int(n_plates) - 1):
            plate = plate.deepcopy()
            plate.translate(
                vector=(
                    plate_and_gap,
                    0,
                    0,
                )
            )
            plate_list.append(plate)
        return plate_list

    def _make_floor_block(self, v1x, v4x):
        xz_profile = Coordinates(
            {
                "x": [v1x, v1x, v4x, v4x],
                "y": [
                    -0.5 * self.params.tf_gs_base_depth.value,
                    0.5 * self.params.tf_gs_base_depth.value,
                    0.5 * self.params.tf_gs_base_depth.value,
                    -0.5 * self.params.tf_gs_base_depth.value,
                ],
                "z": 4 * [self.params.z_gs.value],
            },
        )
        xz_profile = BluemiraFace(make_polygon(xz_profile, closed=True))
        return extrude_shape(
            xz_profile, vec=(0, 0, -5 * self.params.tf_gs_tk_plate.value)
        )

    def build_xyz(
        self,
    ) -> PhysicalComponent:
        """
        Build the x-y-z component of the ITER-like gravity support.
        """
        shape_list = []
        # First, project upwards at the radius of the GS into the keep-out-zone
        # and get a x-z face of the boolean difference.

        # Get the square width
        width = self.params.tf_wp_depth.value + 2 * self.params.tk_tf_side.value

        intersection_wire = self._get_intersection_wire(width)
        v1 = intersection_wire.start_point()
        v4 = intersection_wire.end_point()
        if v1.x > v4.x:
            v1, v4 = v4, v1

        connection_block = self._make_connection_block(width, v1, v4, intersection_wire)
        shape_list.append(connection_block)

        # Next, make the plates in a linear pattern, in y-z, along x
        z_block_lower = connection_block.bounding_box.z_min
        shape_list.extend(
            self._make_plates(width, float(v1.x), float(v4.x), z_block_lower)
        )

        # Finally, make the floor block
        shape_list.append(self._make_floor_block(float(v1.x), float(v4.x)))
        shape = boolean_fuse(shape_list)
        component = PhysicalComponent("ITER-like gravity support", shape)
        component.display_cad_options.color = BLUE_PALETTE["TF"][2]
        return component


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

                # Get the first intersection point, what ever happened to my
                # get_first_intersection?!
                distances = []
                for inter_point_pair in info:
                    dist = np.hypot(
                        inter_point_pair[0][0] - point[0],
                        inter_point_pair[0][2] - point[1],
                    )
                    distances.append(dist)

                i_min = np.argmin(distances)
                p_inter = info[i_min][0]

                if np.isclose(d_intersection, 0.0):
                    d = np.hypot(point[0] - p_inter[0], point[1] - p_inter[2])
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

        # Get the intersection with the TF edge wire and use this for the rib profile
        bb = support_face.bounding_box
        if start_point[1] > end_point[1]:
            z_s = bb.z_min
        else:
            z_s = bb.z_max

        # some offset to get one small wire when cutting
        x_out1 = bb.x_min + np.cos(angle) * 2
        z_out1 = z_s + np.sin(angle) * 2
        x_out2 = bb.x_max + np.cos(angle) * 2
        z_out2 = z_s + np.sin(angle) * 2
        v1 = np.array([bb.x_min, 0, z_s])
        v2 = np.array([bb.x_max, 0, z_s])
        v3 = np.array([x_out2, 0, z_out2])
        v4 = np.array([x_out1, 0, z_out1])

        cut_box = make_polygon([v1, v2, v3, v4], closed=True)
        intersection_wire = sorted(
            boolean_cut(self.tf_xz_keep_out_zone, cut_box), key=lambda wire: wire.length
        )[0]

        # Make the closing wire, and make sure the polygon doesn't self-intersect
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

        # Make the rib x-z profile
        xz_profile = BluemiraFace(BluemiraWire([intersection_wire, closing_wire]))

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

        shape = boolean_fuse(shape_list)
        shape.translate(vector=(0, -0.5 * width, 0))
        component = PhysicalComponent("PF coil support", shape)
        component.display_cad_options.color = BLUE_PALETTE["TF"][2]
        return component


if __name__ == "__main__":
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

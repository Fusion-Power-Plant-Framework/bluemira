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
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.coordinates import Coordinates, get_intersect
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    extrude_shape,
    make_polygon,
    mirror_shape,
    offset_wire,
    slice_shape,
    sweep_shape,
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
        z_max = z_min + 0.5 * (self.tf_xz_keep_out_zone.bounding_box.z_max - z_min)
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
        n_plates = int((plating_width + self.params.tf_gs_g_plate.value) / plate_and_gap)
        total_width = (
            n_plates * self.params.tf_gs_tk_plate.value
            + (n_plates - 1) * self.params.tf_gs_g_plate.value
        )
        delta_width = plating_width - total_width
        yz_profile.translate(vector=(0.5 * delta_width, 0, 0))

        plate = extrude_shape(
            BluemiraFace(yz_profile), vec=(self.params.tf_gs_tk_plate.value, 0, 0)
        )
        plate_list.append(plate)
        for _ in range(n_plates - 1):
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

    SUPPORT = "PF coil support"
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
        xyz = self.build_xyz()
        return self.component_tree(self.build_xz(xyz), self.build_xy(), [xyz])

    def build_xy(self):
        """
        Build the x-y components of the PF coil support.
        """
        pass

    def build_xz(self, xyz):
        """
        Build the x-z components of the PF coil support.
        """
        pass

    def _build_support_xs(self):
        bb = self.pf_coil_xz.bounding_box
        width = self.params.tf_wp_depth.value + 2 * self.params.tk_tf_side.value
        half_width = 0.5 * width

        if bb.x_min < half_width:
            raise BuilderError("PF coil has too small a minimum radius!")

        alpha = np.arcsin(half_width / bb.x_min)
        inner_dr = half_width * np.tan(alpha)

        beta = np.arcsin(half_width / bb.x_max)
        outer_dr = half_width * np.tan(beta)

        x_min = bb.x_min - self.params.pf_s_g.value - inner_dr
        x_max = bb.x_max + self.params.pf_s_g.value + outer_dr
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
        box_outer = offset_wire(box_inner, self.params.pf_s_tk_plate.value)
        face = BluemiraFace([box_outer, box_inner])
        return face

    @staticmethod
    def _get_first_intersection(point, angle, wire):
        """
        Get the first intersection from a point along an angle with a wire.
        """
        point = np.array(point)
        x_out = point[0] + np.cos(angle) * VERY_BIG
        z_out = point[2] + np.sin(angle) * VERY_BIG
        dir_point = np.array([x_out, 0, z_out])

        correct_direction = dir_point - point
        correct_direction /= np.linalg.norm(correct_direction)

        plane = BluemiraPlane.from_3_points(point, dir_point, [x_out, 1, z_out])
        intersections = slice_shape(wire, plane)
        distances = []
        if intersections is None:
            return None

        directed_intersections = []
        for inter in intersections:
            direction = inter - point
            direction /= np.linalg.norm(direction)
            if not np.dot(correct_direction, direction) < 0:
                dx = inter[0] - point[0]
                dz = inter[2] - point[2]

                dist = np.hypot(dx, dz)
                distances.append(dist)
                directed_intersections.append(inter)

        if len(directed_intersections) > 0:
            i_min = np.argmin(distances)
            p_inter = directed_intersections[i_min]
            return p_inter

    def _get_support_point_angle(self, support_face: BluemiraFace):
        bb = support_face.boundary[0].bounding_box
        z_down = bb.z_min
        z_up = bb.z_max

        distance = np.inf
        best_angle = None
        v1, v2, v3, v4 = None, None, None, None
        for z, sign in zip([z_up, z_down], [1, -1]):
            for angle in [0.5 * np.pi, 2 / 3 * np.pi, 1 / 3 * np.pi]:
                p_inters = []
                distances = []
                for x in [bb.x_min, bb.x_max]:
                    point = [x, 0, z]
                    p_inter = self._get_first_intersection(
                        point, sign * angle, self.tf_xz_keep_out_zone
                    )

                    if p_inter is not None:
                        d = np.hypot(point[0] - p_inter[0], point[2] - p_inter[2])
                        p_inters.append(p_inter)
                        distances.append(d)

                if len(p_inters) == 2:
                    avg_distance = np.average(distances)
                    if avg_distance <= distance:
                        distance = avg_distance
                        v1 = np.array([bb.x_min, 0, z])
                        v2 = np.array([bb.x_max, 0, z])
                        v3 = p_inters[1]
                        v4 = p_inters[0]
                        best_angle = sign * angle

        if distance == np.inf:
            raise BuilderError("No intersections found!")

        return v1, v2, v3, v4, best_angle

    def _get_intersecting_wire(self, v1, v2, v3, v4, angle):
        # Add some offset to get one small wire when cutting
        v3 += 0.1 * np.array([np.cos(angle), 0, np.sin(angle)])
        v4 += 0.1 * np.array([np.cos(angle), 0, np.sin(angle)])

        cut_box = make_polygon([v1, v2, v3, v4], closed=True)

        intersection_wire = sorted(
            boolean_cut(self.tf_xz_keep_out_zone, cut_box), key=lambda wire: wire.length
        )[0]
        return intersection_wire

    def _make_rib_profile(self, support_face):
        # Then, project sideways to find the minimum distance from a support point
        # to the TF coil
        v1, v2, v3, v4, angle = self._get_support_point_angle(support_face)

        # Get the intersection with the TF edge wire and use this for the rib profile
        intersection_wire = self._get_intersecting_wire(v1, v2, v3, v4, angle)

        # Make the closing wire, and make sure the polygon doesn't self-intersect
        v3 = intersection_wire.start_point().xyz.T[0]
        v4 = intersection_wire.end_point().xyz.T[0]

        inter1 = get_intersect(
            np.array([[v1[0], v3[0]], [v1[2], v3[2]]]),
            np.array([[v2[0], v4[0]], [v2[2], v4[2]]]),
        )
        if len(inter1[0]) > 0:
            v3, v4 = v4, v3

        closing_wire = make_polygon(
            {
                "x": [v3[0], v1[0], v2[0], v4[0]],
                "y": 0,
                "z": [v3[2], v1[2], v2[2], v4[2]],
            },
            closed=False,
        )
        return BluemiraFace(BluemiraWire([intersection_wire, closing_wire]))

    def _make_ribs(self, width, support_face):
        xz_profile = self._make_rib_profile(support_face)
        # Calculate the rib gap width and make the ribs
        rib_list = []
        total_rib_tk = self.params.pf_s_n_plate.value * self.params.pf_s_tk_plate.value
        if total_rib_tk >= width:
            bluemira_warn(
                "PF coil support rib thickness and number exceed available thickness! You're getting a solid block instead"
            )
            gap_size = 0
            rib_block = extrude_shape(xz_profile, vec=(0, width, 0))
            rib_list.append(rib_block)
        else:
            gap_size = (width - total_rib_tk) / (self.params.pf_s_n_plate.value - 1)
            rib = extrude_shape(xz_profile, vec=(0, self.params.pf_s_tk_plate.value, 0))
            rib_list.append(rib)
            for _ in range(self.params.pf_s_n_plate.value - 1):
                rib = rib.deepcopy()
                rib.translate(vector=(0, self.params.pf_s_tk_plate.value + gap_size, 0))
                rib_list.append(rib)
        return rib_list

    def build_xyz(
        self,
    ) -> PhysicalComponent:
        """
        Build the x-y-z components of the PF coil support.
        """
        shape_list = []
        # First build the support block around the PF coil
        support_face = self._build_support_xs()
        width = self.params.tf_wp_depth.value + 2 * self.params.tk_tf_side.value
        support_block = extrude_shape(support_face, vec=(0, width, 0))
        shape_list.append(support_block)

        # Make the rib x-z profile and ribs
        shape_list.extend(self._make_ribs(width, support_face))

        shape = boolean_fuse(shape_list)
        shape.translate(vector=(0, -0.5 * width, 0))
        component = PhysicalComponent(self.SUPPORT, shape)
        component.display_cad_options.color = BLUE_PALETTE["TF"][2]
        return component


@dataclass
class OISBuilderParams(ParameterFrame):
    """
    Outer intercoil structure parameters
    """

    n_TF: Parameter[int]
    tf_wp_depth: Parameter[float]
    tk_tf_side: Parameter[float]


class OISBuilder(Builder):
    """
    Outer intercoil structure builder
    """

    RIGHT_OIS = "TF OIS right"
    LEFT_OIS = "TF OIS left"
    param_cls: Type[OISBuilderParams] = OISBuilderParams

    def __init__(
        self,
        params: Union[OISBuilderParams, Dict],
        build_config: Dict,
        ois_xz_profile: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.ois_xz_profile = ois_xz_profile

    def build(self) -> Component:
        """
        Build the PF coil support component.
        """
        return self.component_tree([self.build_xz()], self.build_xy(), self.build_xyz())

    def build_xy(self):
        """
        Build the x-y component of the OIS
        """
        pass

    def build_xz(self):
        """
        Build the x-z component of the OIS
        """
        face = BluemiraFace(self.ois_xz_profile)
        component = PhysicalComponent(self.RIGHT_OIS, face)
        component.display_cad_options.color = BLUE_PALETTE["TF"][2]
        component.plot_options.face_options["color"] = BLUE_PALETTE["TF"][2]
        return component

    def build_xyz(self):
        """
        Build the x-y-z component of the OIS
        """
        width = self.params.tf_wp_depth.value + 2 * self.params.tk_tf_side.value
        tf_angle = 2 * np.pi / self.params.n_TF.value
        ois_profile_1 = self.ois_xz_profile.deepcopy()
        ois_profile_1.translate(vector=(0, 0.5 * width, 0))
        ois_profile_2 = ois_profile_1.deepcopy()

        centre_radius = 0.5 * width / np.tan(0.5 * tf_angle)

        ois_profile_2.rotate(
            base=(centre_radius, 0.5 * width, 0), degree=np.rad2deg(tf_angle)
        )

        # First we make the full OIS
        path = make_polygon([ois_profile_1.center_of_mass, ois_profile_2.center_of_mass])
        ois_right = sweep_shape([ois_profile_1, ois_profile_2], path)

        # Then we "chop" it in half, but without the boolean_cut operation
        # This is because I cba to write a project_shape function...
        direction = (-np.sin(0.5 * tf_angle), np.cos(0.5 * tf_angle), 0)
        half_plane = BluemiraPlane(base=(0, 0, 0), axis=direction)
        ois_profile_mid = slice_shape(ois_right, half_plane)[0]

        path = make_polygon(
            [ois_profile_1.center_of_mass, ois_profile_mid.center_of_mass]
        )
        ois_right = sweep_shape([ois_profile_1, ois_profile_mid], path)
        ois_left = mirror_shape(ois_right, base=(0, 0, 0), direction=(0, 1, 0))

        right_component = PhysicalComponent(self.RIGHT_OIS, ois_right)
        right_component.display_cad_options.color = BLUE_PALETTE["TF"][2]
        left_component = PhysicalComponent(self.LEFT_OIS, ois_left)
        left_component.display_cad_options.color = BLUE_PALETTE["TF"][2]
        return [left_component, right_component]

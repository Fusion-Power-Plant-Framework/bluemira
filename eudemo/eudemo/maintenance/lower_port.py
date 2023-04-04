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
EU-DEMO Lower Port
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Type, Union

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    distance_to,
    make_polygon,
    offset_wire,
)
from bluemira.geometry.wire import BluemiraWire


@dataclass
class LowerPortDesignerParams(ParameterFrame):
    """LowerPort ParameterFrame"""

    lower_port_angle: Parameter[float]
    n_TF: Parameter[int]


class LowerPortDesigner(Designer):
    """
    Lower Port Designer

    Notes
    -----
    Retrictions on the lower_port_angle are between [-90, -0.5]

    """

    param_cls: Type[ParameterFrame] = LowerPortDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        divertor_xz: BluemiraFace,
        tf_coil_xz_boundary: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.divertor_face = divertor_xz
        self.tf_coil_xz_boundary = tf_coil_xz_boundary

        if not -90 <= self.params.lower_port_angle.value <= -0.5:
            raise ValueError(
                f"{self.params.lower_port_angle.name}"
                " must be within the range [-90, -0.5] degrees:"
                f" {self.params.lower_port_angle.value}"
            )

    def run(self) -> Tuple[BluemiraFace, BluemiraWire]:
        """Run method of Designer"""
        div_z_top = self.divertor_face.bounding_box.z_max
        div_z_bot = self.divertor_face.bounding_box.z_min

        div_x_inner = self.divertor_face.bounding_box.x_min
        div_x_outer = self.divertor_face.bounding_box.x_max

        r_search = 30
        x_duct_extent = 30

        # these are meant to be params, but their names may change,
        # so leaving them here for now
        lower_duct_angled_leg_extent_factor = 1.2
        lower_duct_angled_leg_padding = 1
        lower_duct_straight_leg_padding = 0.1
        lower_duct_wall_thickness = 0.1

        lower_duct_straight_leg_size = (
            abs(div_z_top - div_z_bot) + lower_duct_straight_leg_padding
        )

        # draw the angled leg from the padded diagonal divertor points

        # get padded divertor points
        div_gradient = (div_z_top - div_z_bot) / (div_x_outer - div_x_inner)
        div_diag_length, _ = distance_to(
            [div_x_inner, 0, div_z_bot],
            [div_x_outer, 0, div_z_top],
        )
        (
            _,
            div_top_outer_point,
        ) = LowerPortDesigner._xz_points_dist_away_from(
            (div_x_inner, div_z_bot),
            div_gradient,
            div_diag_length + lower_duct_angled_leg_padding / 2,
        )
        (div_bot_inner_point, _) = LowerPortDesigner._xz_points_dist_away_from(
            (div_x_outer, div_z_top),
            div_gradient,
            div_diag_length + lower_duct_angled_leg_padding / 2,
        )

        # get "search" points that will definitely intersect the tf_coil
        lower_duct_angled_leg_gradient = np.tan(
            np.deg2rad(self.params.lower_port_angle.value),
        )  # z/x

        (
            _,
            to_intc_search_point,
        ) = LowerPortDesigner._xz_points_dist_away_from(
            div_top_outer_point,
            lower_duct_angled_leg_gradient,
            r_search,
        )
        (
            _,
            bi_intc_search_point,
        ) = LowerPortDesigner._xz_points_dist_away_from(
            div_bot_inner_point,
            lower_duct_angled_leg_gradient,
            r_search,
        )

        lower_duct_angled_leg_boundary = make_polygon(
            {
                "x": np.array(
                    [
                        div_top_outer_point[0],
                        to_intc_search_point[0],
                        bi_intc_search_point[0],
                        div_bot_inner_point[0],
                    ]
                ),
                "y": 0,
                "z": np.array(
                    [
                        div_top_outer_point[1],
                        to_intc_search_point[1],
                        bi_intc_search_point[1],
                        div_bot_inner_point[1],
                    ]
                ),
            },
            closed=True,
        )
        lower_duct_angled_leg = BluemiraFace(
            lower_duct_angled_leg_boundary,
        )

        # Find the points where the angled duct
        # (is large due to the search points)
        # intersect the tf coil
        # and extent the length of that
        # by the lower_duct_angled_leg_extent_factor
        # and draw the straight leg from that point

        to_intc_points = LowerPortDesigner._intersection_points(
            lower_duct_angled_leg_boundary,
            self.tf_coil_xz_boundary,
        )
        # largest x
        to_intc_point = max(to_intc_points, key=lambda p: p[0])

        top_outer_dist_to_intc_point, _ = distance_to(
            [div_top_outer_point[0], 0, div_top_outer_point[1]],
            to_intc_point,
        )
        (
            _,
            duct_straight_leg_point,
        ) = LowerPortDesigner._xz_points_dist_away_from(
            div_top_outer_point,
            lower_duct_angled_leg_gradient,
            top_outer_dist_to_intc_point * lower_duct_angled_leg_extent_factor,
        )

        # draw straight duct leg from the lengthened intersection point
        lower_duct_straight_leg_boundary = make_polygon(
            {
                "x": np.array([0, x_duct_extent, x_duct_extent, 0]),
                "y": 0,
                "z": np.array(
                    [
                        duct_straight_leg_point[1],
                        duct_straight_leg_point[1],
                        duct_straight_leg_point[1] - lower_duct_straight_leg_size,
                        duct_straight_leg_point[1] - lower_duct_straight_leg_size,
                    ]
                ),
            },
            closed=True,
        )
        lower_duct_straight_leg = BluemiraFace(
            lower_duct_straight_leg_boundary,
        )

        # fuse and cut the unec. bits off
        straight_cutters = boolean_cut(
            lower_duct_angled_leg,
            [lower_duct_straight_leg],
        )
        angled_cutters = boolean_cut(
            lower_duct_straight_leg,
            [lower_duct_angled_leg],
        )

        lower_duct_vacant_space = boolean_fuse(
            [lower_duct_angled_leg, lower_duct_straight_leg]
        )
        lower_duct_vacant_space = boolean_cut(
            lower_duct_vacant_space,
            [straight_cutters[1], angled_cutters[0]],
        )[0]

        lower_duct_koz = BluemiraFace(
            offset_wire(
                lower_duct_vacant_space.wires[0],
                lower_duct_wall_thickness,
            )
        )

        return lower_duct_koz

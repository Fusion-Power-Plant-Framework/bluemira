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
from typing import Dict, List, Tuple, Type, Union

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.constants import D_TOLERANCE
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    distance_to,
    make_polygon,
    offset_wire,
)
from bluemira.geometry.wire import BluemiraWire
from eudemo.maintenance.lower_port.parameterisations import LowerPortDesignerParams

# todo: change make_polygon to use a list of points, param names, keep everything in m

# ib -> Inboard
# ob -> Outboard

# inner -> closer to the center
# outer -> further from the center


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

        self.tf_coil_thickness = self.params.tf_coil_thickness.value
        self.n_TF = self.params.n_TF.value
        self.n_div_cassettes = self.params.n_div_cassettes.value
        self.lower_duct_angle = self.params.lower_duct_angle.value
        self.lower_duct_tf_offset = self.params.lower_duct_tf_offset.value
        self.lower_duct_div_pad_dx_outer = self.params.lower_duct_div_pad_dx_outer.value
        self.lower_duct_div_pad_dx_inner = self.params.lower_duct_div_pad_dx_inner.value
        self.lower_duct_wall_tk = self.params.lower_duct_wall_tk.value
        self.lower_port_height = self.params.lower_port_height.value
        self.lower_port_width = self.params.lower_port_width.value

    def run(self) -> Tuple[BluemiraFace, BluemiraWire]:
        """Run method of Designer"""
        ib_div_pt, ob_div_pt = self._get_div_pts_at_angle()
        ib_div_pt_padded, ob_div_pt_padded = self._pad_points(ib_div_pt, ob_div_pt)

        (
            duct_inner_xz_boundary,
            duct_outer_xz_boundary,
            straight_top_inner_pt,
            straight_bot_inner_pt,
        ) = self._duct_xz_shapes(ob_div_pt_padded, ib_div_pt_padded)
        duct_w_wall_xz_koz = BluemiraFace(duct_outer_xz_boundary)

        angled_duct_inner_xy_boundary = self._angled_duct_inner_xy_boundary(
            ib_div_pt_padded, ob_div_pt_padded
        )

        straight_duct_inner_yz_boundary = self._straight_duct_inner_yz_boundary(
            straight_top_inner_pt, straight_bot_inner_pt
        )

        return (
            duct_inner_xz_boundary,
            duct_w_wall_xz_koz,
            angled_duct_inner_xy_boundary,
            straight_duct_inner_yz_boundary,
        )

    @property
    def _angled_duct_gradient(self) -> float:
        return np.tan(np.deg2rad(self.lower_duct_angle))

    @property
    def _half_beta(self) -> float:
        return np.pi / self.n_TF

    def _get_div_pts_at_angle(self) -> Tuple[Tuple, Tuple]:
        div_z_top = self.divertor_face.bounding_box.z_max
        div_z_bot = self.divertor_face.bounding_box.z_min

        div_x_inner = self.divertor_face.bounding_box.x_min
        div_x_outer = self.divertor_face.bounding_box.x_max

        div_diag_len, _ = distance_to(
            [div_x_outer, 0, div_z_top], [div_x_inner, 0, div_z_bot]
        )

        # construct a wire along _angled_duct_gradient, with the
        # start and end points div_diag_len away
        # from the bottomer inner point
        start_end_points = self._xz_points_dist_away_from(
            (div_x_inner, div_z_bot), self._angled_duct_gradient, div_diag_len
        )
        search_wire = self._make_xz_wire_from_points(
            start_end_points[0], start_end_points[1]
        )

        closets_pts = self._closest_points(search_wire, self.divertor_face)
        # just take the point with the highest z, if there's more than one
        z_highest_pt = max(closets_pts, key=lambda p: p[2])

        return (z_highest_pt[0], z_highest_pt[2]), (div_x_outer, div_z_top)

    def _pad_points(self, inner_point: Tuple, outer_point: Tuple):
        points_grad = (inner_point[1] - outer_point[1]) / (
            inner_point[0] - outer_point[0]
        )  # z/x
        points_len, _ = distance_to(
            [inner_point[0], 0, inner_point[1]], [outer_point[0], 0, outer_point[1]]
        )

        padded_bot_inner_point, _ = self._xz_points_dist_away_from(
            outer_point, points_grad, points_len + self.lower_duct_div_pad_dx_inner
        )
        _, padded_top_outer_point = self._xz_points_dist_away_from(
            inner_point, points_grad, points_len + self.lower_duct_div_pad_dx_outer
        )
        return padded_bot_inner_point, padded_top_outer_point

    def _straight_duct_inner_yz_boundary(
        self,
        straight_top_pt: Tuple,
        straight_bot_pt: Tuple,
    ) -> BluemiraFace:
        x_point = straight_top_pt[0]
        y_size = self.lower_port_width / 2

        return make_polygon(
            [
                [x_point] * 4,
                [y_size, y_size, -y_size, -y_size],
                [
                    straight_top_pt[1],
                    straight_bot_pt[1],
                    straight_bot_pt[1],
                    straight_top_pt[1],
                ],
            ],
            closed=True,
        )

    def _angled_duct_inner_xy_boundary(
        self, ib_div_pt_padded: Tuple, ob_div_pt_padded: Tuple
    ):
        def _calc_y_point(x_point):
            x_meet = self.tf_coil_thickness / np.sin(self._half_beta)
            x_len = x_point - x_meet

            if x_len < 0:
                raise GeometryError(
                    "LowerPortDesigner: tf_coil_thickness is too large for the"
                    f" space between TF coils at x={x_point}."
                )

            y_at_x_proj = x_len * np.tan(self._half_beta)
            return y_at_x_proj

        ib_inner_y = _calc_y_point(ib_div_pt_padded[0]) - self.lower_duct_wall_tk
        ob_inner_y = _calc_y_point(ob_div_pt_padded[0]) - self.lower_duct_wall_tk

        # check if the space between the y-points is large enough for the
        # divertor to fit through

        # Approximation of the divertor width at x-point (ib or ob).
        # It's valid because the angle is small and tf_coil's have straight edges.
        # half sector degree is used because you only comparse to the y pts
        # calc'd above, which are half the space avail.
        div_half_width_at_ib = ib_div_pt_padded[0] * np.tan(
            # half sector degree
            np.deg2rad((360 / self.n_TF) / self.n_div_cassettes)
            / 2
        )
        div_half_width_at_ob = ob_div_pt_padded[0] * np.tan(
            # half sector degree
            np.deg2rad((360 / self.n_TF) / self.n_div_cassettes)
            / 2
        )
        if div_half_width_at_ib > ib_inner_y or div_half_width_at_ob > ob_inner_y:
            raise GeometryError(
                "LowerPortDesigner: duct wall thickness is too large for the "
                "space between TF coils. "
                "Making the duct angle shallower or reducing the "
                "duct wall thickness would help."
            )

        return make_polygon(
            [
                [
                    ib_div_pt_padded[0],
                    ib_div_pt_padded[0],
                    ob_div_pt_padded[0],
                    ob_div_pt_padded[0],
                ],
                [
                    ib_inner_y,
                    -ib_inner_y,
                    -ob_inner_y,
                    ob_inner_y,
                ],
                [
                    ib_div_pt_padded[1],
                    ib_div_pt_padded[1],
                    ob_div_pt_padded[1],
                    ob_div_pt_padded[1],
                ],
            ],
            closed=True,
        )

    def _duct_xz_shapes(
        self,
        top_outer_point: Tuple,
        bottom_inner_point: Tuple,
    ):
        angled_duct_boundary = self._angled_duct_boundary(
            top_outer_point,
            bottom_inner_point,
        )

        offset_tf_boundary = offset_wire(
            self.tf_coil_xz_boundary, self.lower_duct_tf_offset
        )

        (
            duct_inner_boundary,
            straight_top_inner_pt,
            straight_bot_inner_edge,
        ) = self._duct_inner_boundary(angled_duct_boundary, offset_tf_boundary)

        return (
            duct_inner_boundary,
            offset_wire(duct_inner_boundary, self.lower_duct_wall_tk),
            straight_top_inner_pt,
            straight_bot_inner_edge,
        )

    def _angled_duct_boundary(self, top_outer_point: Tuple, bottom_inner_point: Tuple):
        """
        Returns a rectangular face at the duct angle,
        starting at the `top_outer_point` and `bottom_inner_point` points
        """
        r_search = 50  # must just be large

        angled_duct_grad = self._angled_duct_gradient  # z/x

        # get "search" points to construct a really long angle duct
        # that will intersect the straight duct
        _, to_intc_search_point = self._xz_points_dist_away_from(
            top_outer_point, angled_duct_grad, r_search
        )
        _, bi_intc_search_point = self._xz_points_dist_away_from(
            bottom_inner_point, angled_duct_grad, r_search
        )

        return make_polygon(
            [
                [
                    top_outer_point[0],
                    to_intc_search_point[0],
                    bi_intc_search_point[0],
                    bottom_inner_point[0],
                ],
                [0, 0, 0, 0],
                [
                    top_outer_point[1],
                    to_intc_search_point[1],
                    bi_intc_search_point[1],
                    bottom_inner_point[1],
                ],
            ],
            closed=True,
        )

    def _duct_inner_boundary(
        self, angled_duct_boundary: BluemiraWire, tf_boundary: BluemiraWire
    ):
        x_duct_extent = 30  # must extend past the outer rad. shield

        angled_inner_cut = boolean_cut(angled_duct_boundary, [tf_boundary])[0]
        itc_pts = self._intersection_points(angled_inner_cut, tf_boundary)

        if len(itc_pts) < 2:
            raise GeometryError(
                "LowerPortDesigner: angled duct does not intersect "
                "TF coil boundary sufficiently."
            )

        itc_top_pt = max(itc_pts, key=lambda p: p[2])
        itc_bot_pt = min(itc_pts, key=lambda p: p[2])
        # remap to 2D point
        itc_top_pt = (itc_top_pt[0], itc_top_pt[2])
        itc_bot_pt = (itc_bot_pt[0], itc_bot_pt[2])

        port_z_bot = itc_top_pt[1] - self.lower_port_height

        if itc_bot_pt[1] < port_z_bot:
            raise GeometryError(
                "LowerPortDesigner: port height is too small "
                "for the divertor at this angle"
            )

        # bottom corner point of the straight duct
        straight_top_inner_pt = (itc_bot_pt[0], itc_top_pt[1])
        straight_bot_inner_pt = (itc_bot_pt[0], port_z_bot)

        straight_boundary = make_polygon(
            [
                [
                    itc_bot_pt[0],
                    straight_bot_inner_pt[0],
                    x_duct_extent,
                    x_duct_extent,
                    itc_top_pt[0],
                ],
                [0] * 5,
                [
                    itc_bot_pt[1],
                    straight_bot_inner_pt[1],
                    straight_bot_inner_pt[1],
                    itc_top_pt[1],
                    itc_top_pt[1],
                ],
            ]
        )
        total_bndry = boolean_fuse([angled_inner_cut, straight_boundary])

        return total_bndry, straight_top_inner_pt, straight_bot_inner_pt

    @staticmethod
    def _xz_points_dist_away_from(
        starting_xz_point: Union[Tuple, List],
        gradient: float,
        distance: float,
    ) -> Tuple:
        """
        Returns two points, the first being in the negative x quadrant,
        the second in the positive x quadrant, at a distance away from
        from the starting point, along the line with the given gradient.
        """
        s_x = starting_xz_point[0]
        s_z = starting_xz_point[1]
        sqrt_value = np.sqrt(distance**2 / (1 + gradient**2))
        f_x_pve = sqrt_value + s_x
        f_z_pve = gradient * sqrt_value + s_z
        f_x_nve = -sqrt_value + s_x
        f_z_nve = gradient * -sqrt_value + s_z
        return [f_x_nve, f_z_nve], [f_x_pve, f_z_pve]

    @staticmethod
    def _make_xz_wire_from_points(
        a_xz_point: Tuple,
        b_xz_point: Tuple,
    ) -> BluemiraWire:
        return make_polygon(
            [[a_xz_point[0], b_xz_point[0]], [0] * 2, [a_xz_point[1], b_xz_point[1]]]
        )

    @staticmethod
    def _intersection_points(
        shape_a: BluemiraGeo,
        shape_b: BluemiraGeo,
    ) -> List[Tuple]:
        dist, vects = distance_to(shape_a, shape_b)
        if dist > D_TOLERANCE:  # not intersecting
            return []
        pois = []
        for vect_pair in vects:
            v = vect_pair[0]
            pois.append((v[0], v[1], v[2]))
        return pois

    @staticmethod
    def _closest_points(
        shape_a: BluemiraGeo,
        shape_b: BluemiraGeo,
    ) -> List[Tuple]:
        dist, vects = distance_to(shape_a, shape_b)
        if dist < D_TOLERANCE:  # intersecting, return intersection points
            return LowerPortDesigner._intersection_points(shape_a, shape_b)
        points = []
        vect_pairs = vects[0]
        for v in vect_pairs:
            points.append((v[0], v[1], v[2]))
        return points

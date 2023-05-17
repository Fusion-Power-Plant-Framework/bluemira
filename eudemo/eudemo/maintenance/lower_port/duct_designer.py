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
EU-DEMO Lower Port Duct Designer
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Union

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
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


@dataclass
class LowerPortDuctDesignerParams(ParameterFrame):
    """Lower Port Duct Designer ParameterFrame"""

    tf_coil_thickness: Parameter[float]
    n_TF: Parameter[int]
    n_div_cassettes: Parameter[int]

    lp_duct_angle: Parameter[float]
    lp_duct_tf_offset: Parameter[float]
    lp_duct_wall_tk: Parameter[float]
    lp_duct_div_pad_ob: Parameter[float]
    lp_duct_div_pad_ib: Parameter[float]

    lp_height: Parameter[float]
    lp_width: Parameter[float]


class LowerPortDuctDesigner(Designer):
    """
    Lower Port Duct Designer

    Notes
    -----
    Retractions on the lower_duct_angle are between [-90, 0] degrees.
    """

    param_cls: Type[ParameterFrame] = LowerPortDuctDesignerParams

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
        self.duct_angle = self.params.lp_duct_angle.value
        self.tf_offset = self.params.lp_duct_tf_offset.value
        self.div_pad_ob = self.params.lp_duct_div_pad_ob.value
        self.div_pad_ib = self.params.lp_duct_div_pad_ib.value
        self.wall_tk = self.params.lp_duct_wall_tk.value
        self.port_height = self.params.lp_height.value
        self.port_width = self.params.lp_width.value

    def run(self) -> Tuple[BluemiraFace, BluemiraWire]:
        """Run method of Designer"""
        # ib -> inboard
        # ob -> outboard
        # inner -> closer to the center (or without the duct wall)
        # outer -> further from the center
        # pt -> point

        ib_div_pt, ob_div_pt = self._get_div_pts_at_angle()
        ib_div_pt_padded, ob_div_pt_padded = self._pad_points(ib_div_pt, ob_div_pt)

        (
            duct_inner_xz_boundary,
            duct_outer_xz_boundary,
            straight_top_inner_pt,
            straight_bot_inner_pt,
        ) = self._duct_xz_shapes(ib_div_pt_padded, ob_div_pt_padded)
        duct_w_wall_xz_koz = BluemiraFace(duct_outer_xz_boundary)

        duct_angled_inner_extrude_boundary = self._angled_duct_inner_xy_boundary(
            ib_div_pt_padded, ob_div_pt_padded
        )

        duct_straight_inner_extrude_boundary = self._straight_duct_inner_yz_boundary(
            straight_top_inner_pt, straight_bot_inner_pt
        )

        return (
            duct_inner_xz_boundary,
            duct_w_wall_xz_koz,
            duct_angled_inner_extrude_boundary,
            duct_straight_inner_extrude_boundary,
        )

    @property
    def _duct_angle_gradient(self) -> float:
        return np.tan(np.deg2rad(self.duct_angle))

    @property
    def _half_beta(self) -> float:
        return np.pi / self.n_TF

    def _get_div_pts_at_angle(self) -> Tuple[Tuple, Tuple]:
        div_z_top = self.divertor_face.bounding_box.z_max
        div_z_bot = self.divertor_face.bounding_box.z_min

        div_x_ib = self.divertor_face.bounding_box.x_min
        div_x_ob = self.divertor_face.bounding_box.x_max

        div_diag_len, _ = distance_to([div_x_ob, 0, div_z_top], [div_x_ib, 0, div_z_bot])

        # construct a wire along the angled duct gradient, with the
        # start and end points div_diag_len away
        # from the bottom ib point
        start_end_points = self._xz_points_dist_away_from(
            (div_x_ib, div_z_bot), self._duct_angle_gradient, div_diag_len
        )
        search_wire = self._make_xz_wire_from_points(
            start_end_points[0], start_end_points[1]
        )

        closest_pts = self._closest_points(search_wire, self.divertor_face)
        # just take the point with the highest z, if there's more than one
        z_highest_pt = max(closest_pts, key=lambda p: p[2])

        return (z_highest_pt[0], z_highest_pt[2]), (div_x_ob, div_z_top)

    def _pad_points(self, ib_point: Tuple, ob_point: Tuple):
        points_grad = (ib_point[1] - ob_point[1]) / (ib_point[0] - ob_point[0])  # z/x
        points_len, _ = distance_to(
            [ib_point[0], 0, ib_point[1]], [ob_point[0], 0, ob_point[1]]
        )

        padded_ib_pt, _ = self._xz_points_dist_away_from(
            ob_point, points_grad, points_len + self.div_pad_ib
        )
        _, padded_ob_pt = self._xz_points_dist_away_from(
            ib_point, points_grad, points_len + self.div_pad_ob
        )
        return padded_ib_pt, padded_ob_pt

    def _straight_duct_inner_yz_boundary(
        self,
        straight_top_inner_pt: Tuple,
        straight_bot_inner_pt: Tuple,
    ) -> BluemiraFace:
        """
        Make the inner yz boundary of the straight duct.

        This takes the straight duct inner (no wall) top and bottom
        inboard points and uses the port width to make the boundary.
        """
        x_point = straight_top_inner_pt[0]
        y_size = self.port_width / 2

        return make_polygon(
            [
                [x_point] * 4,
                [y_size, y_size, -y_size, -y_size],
                [
                    straight_top_inner_pt[1],
                    straight_bot_inner_pt[1],
                    straight_bot_inner_pt[1],
                    straight_top_inner_pt[1],
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

        ib_inner_y = _calc_y_point(ib_div_pt_padded[0]) - self.wall_tk
        ob_inner_y = _calc_y_point(ob_div_pt_padded[0]) - self.wall_tk

        # check if the space between the y-points is large enough for the
        # divertor to fit through:
        # This uses an approx. of the divertor width at an x-point (ib or ob).
        # The approx. is valid because the angle is small and tf_coil's
        # have straight edges.
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
        # half sector degree is used because ib_inner_y, ob_inner_y are for
        # the upper half space available for the divertor.
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

    def _duct_xz_shapes(self, ib_div_pt_padded: Tuple, ob_div_pt_padded: Tuple):
        angled_duct_boundary = self._angled_duct_xz_boundary(
            ib_div_pt_padded, ob_div_pt_padded
        )

        offset_tf_boundary = offset_wire(self.tf_coil_xz_boundary, self.tf_offset)

        (
            duct_inner_boundary,
            straight_top_inner_pt,
            straight_bot_inner_pt,
        ) = self._duct_combined_inner_xz_boundary(
            angled_duct_boundary, offset_tf_boundary
        )

        duct_outer_boundary = offset_wire(duct_inner_boundary, self.wall_tk)

        return (
            duct_inner_boundary,
            duct_outer_boundary,
            straight_top_inner_pt,
            straight_bot_inner_pt,
        )

    def _angled_duct_xz_boundary(self, ib_pt: Tuple, ob_pt: Tuple):
        """
        Returns a rectangular face at the duct angle,
        starting at the inboard and outboard points
        of the padded points from the divertor.
        """
        r_search = 50  # must just be large

        # get "search" points to construct a really long angle duct
        # that will intersect the straight duct and be cut
        _, ib_intc_search_point = self._xz_points_dist_away_from(
            ib_pt, self._duct_angle_gradient, r_search
        )
        _, ob_intc_search_point = self._xz_points_dist_away_from(
            ob_pt, self._duct_angle_gradient, r_search
        )

        return make_polygon(
            [
                [
                    ob_pt[0],
                    ob_intc_search_point[0],
                    ib_intc_search_point[0],
                    ib_pt[0],
                ],
                [0, 0, 0, 0],
                [
                    ob_pt[1],
                    ob_intc_search_point[1],
                    ib_intc_search_point[1],
                    ib_pt[1],
                ],
            ],
            closed=True,
        )

    def _duct_combined_inner_xz_boundary(
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

        port_z_bot = itc_top_pt[1] - self.port_height

        if itc_bot_pt[1] < port_z_bot:
            raise GeometryError(
                "LowerPortDesigner: port height is too small "
                "for the divertor at this angle"
            )

        # these have x's of of the btm itc point,
        # the leftmost point of the straight duct,
        # which are needed when building the cad for straight duct
        straight_top_pt = (itc_bot_pt[0], itc_top_pt[1])
        straight_bot_pt = (itc_bot_pt[0], port_z_bot)

        straight_boundary = make_polygon(
            [
                [
                    itc_bot_pt[0],
                    straight_bot_pt[0],
                    x_duct_extent,
                    x_duct_extent,
                    itc_top_pt[0],
                ],
                [0] * 5,
                [
                    itc_bot_pt[1],
                    straight_bot_pt[1],
                    straight_bot_pt[1],
                    itc_top_pt[1],
                    itc_top_pt[1],
                ],
            ]
        )
        combined_boundary = boolean_fuse([angled_inner_cut, straight_boundary])

        return combined_boundary, straight_top_pt, straight_bot_pt

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
            return LowerPortDuctDesigner._intersection_points(shape_a, shape_b)
        points = []
        vect_pairs = vects[0]
        for v in vect_pairs:
            points.append((v[0], v[1], v[2]))
        return points

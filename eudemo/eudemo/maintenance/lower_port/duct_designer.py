# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
EU-DEMO Lower Port Duct KOZ Designer
"""

import operator
from dataclasses import dataclass

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
class LowerPortKOZDesignerParams(ParameterFrame):
    """Lower Port KOZ Designer ParameterFrame"""

    n_TF: Parameter[int]
    n_div_cassettes: Parameter[int]
    lower_port_angle: Parameter[float]
    g_ts_tf: Parameter[float]
    tk_ts: Parameter[float]
    g_vv_ts: Parameter[float]
    tk_vv_single_wall: Parameter[float]
    tf_wp_depth: Parameter[float]

    # Pseudo - local
    lp_height: Parameter[float]
    lp_width: Parameter[float]
    # Local (varying)

    lp_duct_div_pad_ob: Parameter[float]
    lp_duct_div_pad_ib: Parameter[float]


class LowerPortKOZDesigner(Designer):
    """
    Lower Port keep-out-zone designer

    Notes
    -----
    Retractions on the lower_duct_angle are between [-90, 0] degrees.
    """

    params: LowerPortKOZDesignerParams
    param_cls: type[ParameterFrame] = LowerPortKOZDesignerParams

    def __init__(
        self,
        params: dict | ParameterFrame,
        build_config: dict,
        divertor_xz: BluemiraFace,
        div_wall_join_pt: tuple[float, float],
        tf_coil_xz_boundary: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.divertor_face = divertor_xz
        self.div_wall_join_pt = div_wall_join_pt
        self.tf_coil_xz_boundary = tf_coil_xz_boundary

        # TODO: Cross-check with upper port handling and add casing
        # sidewall thickness + gaps?
        self.tf_coil_thickness = 0.5 * self.params.tf_wp_depth.value

        self.tf_offset = (
            self.params.g_ts_tf.value
            + self.params.tk_ts.value
            + self.params.g_vv_ts.value
        )

        self.div_pad_ob = self.params.lp_duct_div_pad_ob.value
        self.div_pad_ib = self.params.lp_duct_div_pad_ib.value
        self.wall_tk = self.params.tk_vv_single_wall.value
        self.port_height = self.params.lp_height.value
        self.port_width = self.params.lp_width.value

    def run(self) -> tuple[BluemiraFace, BluemiraFace, BluemiraWire, BluemiraWire]:
        """
        Run method of Designer

        Returns
        -------
        duct_inner_xz:
            Inner duct xz
        duct_outer_xz:
            Outer duct xz
        duct_angled_inner_extrude_boundary:
            Angled extruded boundary
        duct_straight_inner_extrude_boundary:
            Straight extruded boundary
        """
        # ib -> inboard
        # ob -> outboard
        # inner -> closer to the center (or without the duct wall)
        # outer -> further from the center
        # pt -> point

        ib_div_pt, ob_div_pt = self._get_div_pts_at_angle()
        ib_div_pt_padded, ob_div_pt_padded = self._pad_points(ib_div_pt, ob_div_pt)

        (duct_inner_xz, duct_outer_xz, straight_top_inner_pt, straight_bot_inner_pt) = (
            self._duct_xz_shapes(ib_div_pt_padded, ob_div_pt_padded)
        )

        duct_angled_inner_extrude_boundary = self._angled_duct_inner_xy_boundary(
            ib_div_pt_padded, ob_div_pt_padded
        )

        duct_straight_inner_extrude_boundary = self._straight_duct_inner_yz_boundary(
            straight_top_inner_pt, straight_bot_inner_pt
        )

        return (
            duct_inner_xz,
            duct_outer_xz,
            duct_angled_inner_extrude_boundary,
            duct_straight_inner_extrude_boundary,
        )

    @property
    def _duct_angle_gradient(self) -> float:
        return np.tan(np.deg2rad(self.params.lower_port_angle.value))

    @property
    def _half_beta(self) -> float:
        return np.pi / self.params.n_TF.value

    def _get_div_pts_at_angle(self) -> tuple[tuple[float, ...], ...]:
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
        z_highest_pt = max(closest_pts, key=operator.itemgetter(2))

        return (z_highest_pt[0], z_highest_pt[2]), self.div_wall_join_pt

    def _pad_points(self, ib_point: tuple, ob_point: tuple):
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
        self, straight_top_inner_pt: tuple, straight_bot_inner_pt: tuple
    ) -> BluemiraWire:
        """
        Make the inner yz boundary of the straight duct.

        This takes the straight duct inner (no wall) top and bottom
        inboard points and uses the port width to make the boundary.

        Returns
        -------
        :
            Inner yz wire of straight duct
        """
        x_point = straight_top_inner_pt[0]

        # restrict port size of angled duct to port_width
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
        self, ib_div_pt_padded: tuple, ob_div_pt_padded: tuple
    ):
        def _calc_y_point(x_point):
            # TODO(je-cook) This functions assumes the TF coil is circular
            # this is not a great approximation, in future could look at otho projection.
            x_meet = self.tf_coil_thickness / np.sin(self._half_beta)
            x_len = x_point - x_meet

            if x_len < 0:
                raise GeometryError(
                    "LowerPortDesigner: tf_coil_thickness is too large for the"
                    f" space between TF coils at x={x_point}."
                )

            return x_len * np.tan(self._half_beta)

        # TODO alternative limit
        flat_y_max = self.port_width / 2
        ib_inner_y = min(flat_y_max, _calc_y_point(ib_div_pt_padded[0]) - self.wall_tk)
        ob_inner_y = min(flat_y_max, _calc_y_point(ob_div_pt_padded[0]) - self.wall_tk)

        # check if the space between the y-points is large enough for the
        # divertor to fit through:
        # This uses an approx. of the divertor width at an x-point (ib or ob).
        # The approx. is valid because the angle is small and tf_coil's
        # have straight edges.
        # Half-sector degree
        angle = np.pi / self.params.n_TF.value / self.params.n_div_cassettes.value
        div_half_width_at_ib = ib_div_pt_padded[0] * np.tan(angle)
        div_half_width_at_ob = ob_div_pt_padded[0] * np.tan(angle)
        # half sector degree is used because ib_inner_y, ob_inner_y are for
        # the upper half space available for the divertor.
        if div_half_width_at_ib > ib_inner_y or div_half_width_at_ob > ob_inner_y:
            raise GeometryError(
                "LowerPortDesigner: duct wall thickness is too large for the "
                "space between TF coils. "
                "Making the duct angle shallower or reducing the "
                "duct wall thickness would help."
            )

        ib_div_pt_x, ib_div_pt_z = ib_div_pt_padded
        ob_div_pt_x, ob_div_pt_z = ob_div_pt_padded

        x = [ib_div_pt_x, ib_div_pt_x, ob_div_pt_x, ob_div_pt_x]
        y = [ib_inner_y, -ib_inner_y, -ob_inner_y, ob_inner_y]
        z = [ib_div_pt_z, ib_div_pt_z, ob_div_pt_z, ob_div_pt_z]
        duct_inner_xy = make_polygon({"x": x, "y": y, "z": z}, closed=True)

        # Translate a little inwards and upwards to ensure penetration to
        # main body
        angle = np.deg2rad(self.params.lower_port_angle.value)
        direction = np.array([np.cos(angle), 0, np.sin(angle)])
        duct_inner_xy.translate(-1 * direction)
        return duct_inner_xy

    def _duct_xz_shapes(self, ib_div_pt_padded: tuple, ob_div_pt_padded: tuple):
        angled_duct_boundary = self._angled_duct_xz_boundary(
            ib_div_pt_padded, ob_div_pt_padded
        )

        (straight_duct_boundary, straight_top_inner_pt, straight_bot_inner_pt) = (
            self._straight_duct_xz_boundary(angled_duct_boundary)
        )

        angled_cuts = boolean_cut(angled_duct_boundary, [straight_duct_boundary])

        angled_duct_top_xz = angled_cuts[0]
        angled_duct_top_xz.close()
        angled_duct_top_xz = BluemiraFace(angled_duct_top_xz)

        straight_duct_xz = BluemiraFace(straight_duct_boundary)

        duct_inner_xz: BluemiraFace = boolean_fuse([
            angled_duct_top_xz,
            straight_duct_xz,
        ])
        duct_inner_boundary = duct_inner_xz.boundary[0]

        duct_outer_boundary = offset_wire(duct_inner_boundary, self.wall_tk)
        duct_outer_xz = BluemiraFace(duct_outer_boundary)

        return (
            duct_inner_xz,
            duct_outer_xz,
            straight_top_inner_pt,
            straight_bot_inner_pt,
        )

    def _angled_duct_xz_boundary(self, ib_pt: tuple, ob_pt: tuple):
        """
        Returns
        -------
        :
            A rectangular face at the duct angle,
            starting at the inboard and outboard points
            of the padded points from the divertor.
        """
        r_search = 40  # must just be large

        # construct a really long angle duct
        _, ib_end_pt = self._xz_points_dist_away_from(
            ib_pt, self._duct_angle_gradient, r_search
        )
        _, ob_end_pt = self._xz_points_dist_away_from(
            ob_pt, self._duct_angle_gradient, r_search
        )

        return make_polygon(
            [
                [ob_pt[0], ob_end_pt[0], ib_end_pt[0], ib_pt[0]],
                [0, 0, 0, 0],
                [ob_pt[1], ob_end_pt[1], ib_end_pt[1], ib_pt[1]],
            ],
            closed=True,
        )

    def _straight_duct_xz_boundary(self, angled_duct_boundary: BluemiraWire):
        x_duct_extent = 30  # must extend past the outer rad. shield
        tf_offset_boundary = offset_wire(self.tf_coil_xz_boundary, self.tf_offset)

        itc_pts = self._intersection_points(angled_duct_boundary, tf_offset_boundary)

        if len(itc_pts) < 2:  # noqa: PLR2004
            raise GeometryError(
                "LowerPortDesigner: angled duct must be made larger (increase r_search)"
            )

        # find the top and bottom itc points
        itc_top_ob_pt = max(itc_pts, key=operator.itemgetter(2))
        itc_bot_ib_pt = min(itc_pts, key=operator.itemgetter(2))

        # remap to 2D point
        itc_top_ob_pt = (itc_top_ob_pt[0], itc_top_ob_pt[2])
        itc_bot_ib_pt = (itc_bot_ib_pt[0], itc_bot_ib_pt[2])

        if (
            self.params.lower_port_angle.value < -45  # noqa: PLR2004
            and itc_top_ob_pt[0] < itc_bot_ib_pt[0]
        ):
            # This is a weird edge case where the 'top' x point is more
            # inboard than the 'bottom' x point
            itc_top_ob_pt, itc_bot_ib_pt = itc_bot_ib_pt, itc_top_ob_pt

        # choose corner point
        topleft_corner_pt = itc_bot_ib_pt
        if self.params.lower_port_angle.value > -45:  # noqa: PLR2004
            topleft_corner_pt = itc_top_ob_pt

        topright_corner_pt = (x_duct_extent, topleft_corner_pt[1])

        botright_corner_pt = (x_duct_extent, topleft_corner_pt[1] - self.port_height)

        botleft_corner_pt = (
            topleft_corner_pt[0],
            topleft_corner_pt[1] - self.port_height,
        )

        # check if the left edge goes below the angled duct when
        # the corner point is the top itc point (i.e. angle > -45)
        if topleft_corner_pt == itc_top_ob_pt:
            left_e = self._make_xz_wire_from_points(topleft_corner_pt, botleft_corner_pt)
            l_e_itc_pts = self._intersection_points(left_e, angled_duct_boundary)
            if len(l_e_itc_pts) == 1:
                raise GeometryError(
                    "LowerPortDesigner: port height is too small "
                    "at this angle and will not meet the angled duct."
                )

        straight_boundary = make_polygon(
            [
                [
                    topleft_corner_pt[0],
                    topright_corner_pt[0],
                    botright_corner_pt[0],
                    botleft_corner_pt[0],
                ],
                [0] * 4,
                [
                    topleft_corner_pt[1],
                    topright_corner_pt[1],
                    botright_corner_pt[1],
                    botleft_corner_pt[1],
                ],
            ],
            closed=True,
        )

        return straight_boundary, topleft_corner_pt, botleft_corner_pt

    @staticmethod
    def _xz_points_dist_away_from(
        starting_xz_point: tuple | list, gradient: float, distance: float
    ) -> tuple[list[float], list[float]]:
        """
        Returns
        -------
        :
            Two points, the first being in the negative x quadrant,
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
    def _make_xz_wire_from_points(a_xz_point: tuple, b_xz_point: tuple) -> BluemiraWire:
        return make_polygon([
            [a_xz_point[0], b_xz_point[0]],
            [0] * 2,
            [a_xz_point[1], b_xz_point[1]],
        ])

    @staticmethod
    def _intersection_points(shape_a: BluemiraGeo, shape_b: BluemiraGeo) -> list[tuple]:
        dist, vects = distance_to(shape_a, shape_b)
        if dist > D_TOLERANCE:  # not intersecting
            return []
        pois = []
        for vect_pair in vects:
            v = vect_pair[0]
            pois.append((v[0], v[1], v[2]))
        return pois

    @staticmethod
    def _closest_points(shape_a: BluemiraGeo, shape_b: BluemiraGeo) -> list[tuple]:
        dist, vects = distance_to(shape_a, shape_b)
        if dist < D_TOLERANCE:  # intersecting, return intersection points
            return LowerPortKOZDesigner._intersection_points(shape_a, shape_b)
        return [(v[0], v[1], v[2]) for v in vects[0]]

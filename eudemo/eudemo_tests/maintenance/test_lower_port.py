# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tests for EU-DEMO Lower Port
"""

import numpy as np
import pytest

from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import (
    boolean_fuse,
    make_circle,
    make_polygon,
    offset_wire,
    sweep_shape,
)
from bluemira.geometry.wire import BluemiraWire
from eudemo.maintenance.lower_port.builder import TSLowerPortDuctBuilder
from eudemo.maintenance.lower_port.duct_designer import (
    LowerPortKOZDesigner,
    LowerPortKOZDesignerParams,
)


class TestLowerPort:
    """Test Lower Port Duct Designer"""

    @classmethod
    def setup_class(cls):
        cls.divertor_xz_silhouette = BluemiraWire([
            make_polygon([
                [2, 4],
                [0, 0],
                [0, 0],
            ]),
            make_circle(
                1, center=(3, 0, 0), start_angle=-180, end_angle=0, axis=(0, -1, 0)
            ),
        ])
        cls.tf_coils_centre_boundary = PrincetonD({
            "x1": {"value": 1},
            "x2": {"value": 8, "lower_bound": 8},
            "dz": {"value": 5, "upper_bound": 5},
        }).create_shape()

        cls.tf_coils_outer_boundary = offset_wire(cls.tf_coils_centre_boundary, 0.5)

        BluemiraWire([
            make_polygon([
                [3, 3],
                [0, 0],
                [8, -8],
            ]),
            make_circle(
                8, center=(3, 0, 0), start_angle=-90, end_angle=90, axis=(0, -1, 0)
            ),
        ])

    def setup_method(self):
        self.duct_des_params = make_parameter_frame(
            {
                "n_TF": {"value": 10, "unit": "dimensionless"},
                "n_div_cassettes": {"value": 3, "unit": "dimensionless"},
                "lower_port_angle": {"value": -30, "unit": "degrees"},
                "g_ts_tf": {"value": 0.05, "unit": "m"},
                "tk_ts": {"value": 0.05, "unit": "m"},
                "g_vv_ts": {"value": 0.05, "unit": "m"},
                "tk_vv_single_wall": {"value": 0.06, "unit": "m"},
                "tf_wp_depth": {"value": 0.5, "unit": "m"},
                "lp_duct_div_pad_ob": {"value": 0.3, "unit": "m"},
                "lp_duct_div_pad_ib": {"value": 0.1, "unit": "m"},
                "lp_height": {"value": 4.5, "unit": "m"},
                "lp_width": {"value": 3, "unit": "m"},
            },
            LowerPortKOZDesignerParams,
        )

    @pytest.mark.parametrize("duct_angle", [0, -30, -45, -60, -90])
    @pytest.mark.parametrize("tf_wp_depth", np.linspace(0, 0.5, 3))
    def test_duct_angle(self, duct_angle, tf_wp_depth):
        self.duct_des_params.lower_port_angle.value = duct_angle
        self.duct_des_params.tf_wp_depth.value = tf_wp_depth

        (
            lp_duct_xz_void_space,
            lp_duct_xz_koz,
            lp_duct_angled_nowall_extrude_boundary,
            lp_duct_straight_nowall_extrude_boundary,
        ) = LowerPortKOZDesigner(
            self.duct_des_params,
            {},
            self.divertor_xz_silhouette,
            (3.5, 0),  # connection point between divertor and blanket
            self.tf_coils_outer_boundary,
        ).execute()

        builder = TSLowerPortDuctBuilder(
            self.duct_des_params,
            {},
            lp_duct_angled_nowall_extrude_boundary,
            lp_duct_straight_nowall_extrude_boundary,
            15,  # [m] bigger than the radius of the machine
        )
        lp_duct = builder.build()

        # make angle plane
        x_angled_start = lp_duct_angled_nowall_extrude_boundary.vertexes.x[0]
        z_angled_start = lp_duct_angled_nowall_extrude_boundary.vertexes.z[0]
        pl = BluemiraFace(
            make_polygon(
                [
                    [
                        x_angled_start,
                        x_angled_start + 1,
                        x_angled_start + 1,
                        x_angled_start,
                    ],
                    [0.1, 0.1, -0.1, -0.1],
                    [z_angled_start] * 4,
                ],
                closed=True,
            )
        )
        pl.rotate(
            degree=duct_angle,
            base=(x_angled_start, 0, z_angled_start),
            direction=(0, -1, 0),
        )
        pl.rotate(degree=180 / self.duct_des_params.n_TF.value)

        duct_xyz_cad = lp_duct.get_component("xyz").get_component(builder.name).shape

        angled_face = duct_xyz_cad.faces[0]  # this was an angled face when I tested it

        np.testing.assert_allclose(angled_face.normal_at(), pl.normal_at(), atol=1e-8)

        # Does duct touch tf coils
        tf = self._make_tf()
        tf2 = tf.deepcopy()
        tf2.rotate(degree=360 / self.duct_des_params.n_TF.value)

        with pytest.raises(
            GeometryError,
            match=r".*\[(<Solid[\n ]*object at [0-9a-z]{14}>[, ]*){3}\]",
        ):
            boolean_fuse([tf, tf2, duct_xyz_cad])
        # import ipdb

        # ipdb.set_trace()
        # lp_duct_xz_koz
        from bluemira.display import show_cad

        # show_cad([lp_duct_xz_koz, duct_xyz_cad])

        show_cad([tf, tf2, duct_xyz_cad, self.divertor_xz_silhouette, lp_duct_xz_koz])

    @pytest.mark.parametrize("duct_angle", [0, -30, -45, -60, -90])
    @pytest.mark.parametrize("tf_wp_depth", np.linspace(0, 0.5, 3))
    def test_straight_duct_boundingbox_is_larger_than_angled_duct(
        self, duct_angle, tf_wp_depth
    ):
        self.duct_des_params.lower_port_angle.value = duct_angle
        self.duct_des_params.tf_wp_depth.value = tf_wp_depth

        (
            _,
            _,
            lp_duct_angled_nowall_extrude_boundary,
            lp_duct_straight_nowall_extrude_boundary,
        ) = LowerPortKOZDesigner(
            self.duct_des_params,
            {},
            self.divertor_xz_silhouette,
            (3.5, 0),  # connection point between divertor and blanket
            self.tf_coils_outer_boundary,
        ).execute()

        angle_bb = lp_duct_angled_nowall_extrude_boundary.bounding_box
        straight_bb = lp_duct_straight_nowall_extrude_boundary.bounding_box
        assert angle_bb.y_max <= straight_bb.y_max
        assert angle_bb.y_min >= straight_bb.y_min

    def test_too_thick_tf_raises_GeometryError(self):
        self.duct_des_params.lower_port_angle.value = -60
        self.duct_des_params.tf_wp_depth.value = 0.75

        with pytest.raises(GeometryError, match="duct wall thickness is too large"):
            LowerPortKOZDesigner(
                self.duct_des_params,
                {},
                self.divertor_xz_silhouette,
                (3.5, 0),  # connection point between divertor and blanket
                self.tf_coils_outer_boundary,
            ).execute()

    def _get_tf_y_max(self):
        return (
            (0.5 * self.duct_des_params.tf_wp_depth.value)
            + self.duct_des_params.g_ts_tf.value
            + self.duct_des_params.tk_ts.value
            + self.duct_des_params.g_vv_ts.value
        )

    def _make_tf(self):
        tf_ymax = self._get_tf_y_max()
        xy_cross = make_polygon(
            {"x": [1.5, 0.5, 0.5, 1.5], "y": [tf_ymax, tf_ymax, -tf_ymax, -tf_ymax]},
            closed=True,
        )

        return sweep_shape(
            xy_cross,
            self.tf_coils_centre_boundary,
        )

# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tests for EU-DEMO Maintenance
"""

import numpy as np
import pytest

from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire
from eudemo.maintenance.equatorial_port import (
    EquatorialPortDuctBuilder,
    EquatorialPortKOZDesigner,
)


class TestEquatorialPortKOZDesigner:
    """Tests the Equatorial Port KOZ Designer"""

    def setup_method(self) -> None:
        """Set-up Equatorial Port Designer"""

    @pytest.mark.parametrize(
        ("xi", "xo", "zh"),
        zip([9.0, 9.0, 6.0], [16.0, 15.0, 9.0], [5.0, 4.0, 2.0], strict=False),
    )
    def test_ep_designer(self, xi, xo, zh):
        """Test Equatorial Port KOZ Designer"""
        R_0 = xi
        self.params = {
            "R_0": {"value": R_0, "unit": "m"},
            "ep_height": {"value": zh, "unit": "m"},
            "ep_z_position": {"value": 0.0, "unit": "m"},
            "g_vv_ts": {"value": 0.0, "unit": "m"},
            "tk_ts": {"value": 0.0, "unit": "m"},
            "g_ts_tf": {"value": 0.0, "unit": "m"},
            "pf_s_g": {"value": 0.0, "unit": "m"},
            "pf_s_tk_plate": {"value": 0.0, "unit": "m"},
            "tk_vv_single_wall": {"value": 0.0, "unit": "m"},
        }
        x_len = xo - R_0
        self.designer = EquatorialPortKOZDesigner(self.params, None, xo)
        output = self.designer.execute()

        assert np.isclose(output.length, 2 * (x_len + zh))
        assert np.isclose(output.area, x_len * zh)


class TestEquatorialPortDuctBuilder:
    """Tests the Equatorial Port Duct Builder"""

    def setup_method(self) -> None:
        """Set-up to Equatorial Port Duct Builder"""

    @pytest.mark.parametrize(
        ("xi", "xo", "z", "y", "th"),
        zip(
            [9.0, 9.0, 6.0],  # x_inboard
            [16.0, 15.0, 9.0],  # x_outboard
            [5.0, 4.0, 2.0],  # z_height
            [3.0, 2.0, 1.0],  # y_widths
            [0.5, 0.5, 0.25],  # thickness
            strict=False,
            # expected volumes: [63, 42, 5.25]
        ),
    )
    def test_ep_builder(self, xi, xo, z, y, th):
        """Test Equatorial Port Duct Builder"""
        self.params = {
            "ep_height": {"value": z, "unit": "m"},
            "cst_r_corner": {"value": 0, "unit": "m"},
        }
        y_tup = (y / 2.0, -y / 2.0, -y / 2.0, y / 2.0)
        z_tup = (-z / 2.0, -z / 2.0, z / 2.0, z / 2.0)
        yz_profile = BluemiraWire(
            make_polygon({"x": xi, "y": y_tup, "z": z_tup}, closed=True)
        )
        length = xo - xi

        self.builder = EquatorialPortDuctBuilder(self.params, {}, yz_profile, length, th)
        output = self.builder.build()
        out_port = output.get_component("xyz").get_component("Equatorial Port Duct 1")
        if out_port is None:
            out_port = output.get_component("xyz").get_component("Equatorial Port Duct")
        expectation = length * (2 * (th * (y + z - (2 * th))))

        assert np.isclose(out_port.shape.volume, expectation)

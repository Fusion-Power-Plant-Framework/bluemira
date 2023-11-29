# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tests for cryostat builder.
"""

from unittest import mock

import pytest

from bluemira.builders.cryostat import CryostatBuilder, CryostatDesigner


class TestCryostatBuilder:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "g_cr_ts": {"value": 0.3, "unit": "m"},
            "n_TF": {"value": 16, "unit": ""},
            "tk_cr_vv": {"value": 0.3, "unit": "m"},
            "well_depth": {"value": 5, "unit": "m"},
            "x_g_support": {"value": 13, "unit": "m"},
            "x_gs_kink_diff": {"value": 4, "unit": "m"},
            "z_gs": {"value": -15, "unit": "m"},
        }

    def test_components_and_segments(self):
        builder = CryostatBuilder(self.params, {}, 10, 10)
        cryostat = builder.build()

        assert cryostat.get_component("xz")
        assert cryostat.get_component("xy")

        xyz = cryostat.get_component("xyz")
        assert xyz
        assert len(xyz.leaves) == 1

    def test_outward_kink_raises_ValueError(self):
        builder = CryostatBuilder(self.params, {}, 8, 10)
        with pytest.raises(ValueError):  # noqa: PT011
            builder.build()


class TestCryostatDesigner:
    @classmethod
    def setup_class(cls):
        cls.params = {"g_cr_ts": {"value": 0.3, "unit": "m"}}
        cls.z_max = 10
        cls.x_max = 5
        bb = mock.Mock(z_max=cls.z_max, x_max=cls.x_max)
        cls.cryostat_ts_xz = mock.Mock(bounding_box=bb)

    def test_designer(self):
        designer = CryostatDesigner(self.params, self.cryostat_ts_xz)
        x_out, z_out = designer.run()
        assert x_out == self.x_max + self.params["g_cr_ts"]["value"]
        assert z_out == self.z_max + self.params["g_cr_ts"]["value"]

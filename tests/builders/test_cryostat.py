# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Tests for cryostat builder.
"""
from unittest import mock

from bluemira.builders.cryostat import CryostatBuilder, CryostatDesigner


class TestCryostatBuilder:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "g_cr_ts": {"name": "g_cr_ts", "value": 0.3},
            "n_TF": {"name": "n_TF", "value": 16},
            "tk_cr_vv": {"name": "tk_cr_vv", "value": 0.3},
            "well_depth": {"name": "well_depth", "value": 5},
            "x_g_support": {"name": "x_g_support", "value": 13},
            "x_gs_kink_diff": {"name": "x_gs_kink_diff", "value": 2},
            "z_gs": {"name": "z_gs", "value": -15},
        }

    def test_components_and_segments(self):
        designer = mock.Mock(run=lambda: (10, 10))
        builder = CryostatBuilder(self.params, {}, designer)
        cryostat = builder.build()

        assert cryostat.component().get_component("xz")
        assert cryostat.component().get_component("xy")

        xyz = cryostat.component().get_component("xyz")
        assert xyz
        assert len(xyz.leaves) == self.params["n_TF"]["value"]


class TestCryostatDesigner:
    @classmethod
    def setup_class(cls):
        cls.params = {"g_cr_ts": {"name": "g_cr_ts", "value": 0.3}}
        cls.z_max = 10
        cls.x_max = 5
        bb = mock.Mock(z_max=cls.z_max, x_max=cls.x_max)
        cls.cryostat_ts_xz = mock.Mock(bounding_box=bb)

    def test_designer(self):
        designer = CryostatDesigner(self.params, self.cryostat_ts_xz)
        x_out, z_out = designer.run()
        assert x_out == self.x_max + self.params["g_cr_ts"]["value"]
        assert z_out == self.z_max + self.params["g_cr_ts"]["value"]

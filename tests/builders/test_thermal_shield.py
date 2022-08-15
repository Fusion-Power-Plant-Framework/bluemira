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
Tests for thermal shield builders.
"""
from unittest import mock

import numpy as np

from bluemira.builders.thermal_shield import (
    CryostatTSBuilder,
    CryostatTSDesigner,
    VVTSBuilder,
    VVTSDesigner,
)
from bluemira.display.displayer import show_cad
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import make_circle, make_polygon


class TestVVTSBuilder:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "g_vv_ts": {"name": "g_vv_ts", "value": 0.05},
            "n_TF": {"name": "n_TF", "value": 16},
            "tk_ts": {"name": "tk_ts", "value": 0.05},
        }
        cls.vv_koz = make_circle(10, center=(15, 0, 0), axis=(0.0, 1.0, 0.0))

    def test_components_and_segments(self):
        designer = mock.Mock(run=lambda: self.vv_koz)

        builder = VVTSBuilder(self.params, {}, designer)
        vvts = builder.build()

        assert vvts.component().get_component("xz")
        assert vvts.component().get_component("xy")

        xyz = vvts.component().get_component("xyz")
        assert xyz
        # not sectioned because of #1319 and related issues
        # assert len(xyz.leaves) == self.params["n_TF"]["value"]
        # xx = xyz.get_component_properties("shape", first=False)[0]
        show_cad(*xyz.get_component_properties("shape", first=False))


class TestCryostatTSBuilder:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "g_ts_pf": {"name": "g_ts_pf", "value": 0.3},
            "g_ts_tf": {"name": "g_ts_tf", "value": 0.3},
            "n_TF": {"name": "n_TF", "value": 16},
            "tk_ts": {"name": "tk_ts", "value": 0.3},
        }
        size_1_sq = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]).T
        pf_shifts = np.array([(5, 8), (13, 5), (15, 0)])
        squares = []

        for pf_s in pf_shifts:
            cp = size_1_sq.copy()
            cp[0] += pf_s[0]
            cp[2] += pf_s[1]
            squares.append(cp.T)

        # Only half of the pf coils to catch weird edge cases
        cls.pf_coil_koz = [make_polygon(sq, closed=True) for sq in squares]

        cls.tf_xz_koz = BluemiraFace(PrincetonD().create_shape())

    def test_components_and_segments(self):
        designer = mock.Mock(run=lambda: (self.pf_coil_koz, self.tf_xz_koz))
        builder = CryostatTSBuilder(self.params, {}, designer)
        cryostat_ts = builder.build()

        assert cryostat_ts.component().get_component("xz")
        assert cryostat_ts.component().get_component("xy")

        xyz = cryostat_ts.component().get_component("xyz")
        assert xyz
        assert len(xyz.leaves) == self.params["n_TF"]["value"]


class TestThermalShieldDesigners:
    """
    What goes in comes out
    """

    def test_vvtsdesigner(self):
        _in = 1
        designer = VVTSDesigner(_in)
        assert designer.run() == _in

    def test_cryotsdesigner(self):
        _in = 1, 2
        designer = CryostatTSDesigner(*_in)
        assert designer.run() == _in

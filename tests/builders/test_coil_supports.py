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

import pytest

from bluemira.base.error import BuilderError
from bluemira.builders.coil_supports import (
    ITERGravitySupportBuilder,
    ITERGravitySupportBuilderParams,
)
from bluemira.geometry.parameterisations import PictureFrame, PrincetonD, TripleArc


class TestITERGravitySupportBuilder:
    pd = PrincetonD()
    pd.adjust_variable("x1", value=3, lower_bound=2, upper_bound=4)
    pd.adjust_variable("x2", value=15, lower_bound=2, upper_bound=24)
    pd_xz_koz = pd.create_shape()
    pf = PictureFrame()
    pf.adjust_variable("x1", value=3, lower_bound=2, upper_bound=4)
    pf.adjust_variable("x2", value=15, lower_bound=2, upper_bound=24)
    pf_xz_koz = pf.create_shape()
    ta = TripleArc()
    ta.adjust_variable("x1", value=3, lower_bound=2, upper_bound=4)
    ta.adjust_variable("f2", value=2, lower_bound=2, upper_bound=4)
    ta_xz_koz = ta.create_shape()

    tf_kozs = [pd_xz_koz, pf_xz_koz, ta_xz_koz]

    @staticmethod
    def _make_builder(tf, **kwargs):
        defaults = {
            "x_g_support": {"value": 10, "unit": "m"},
            "z_gs": {"value": -20, "unit": "m"},
            "tf_wp_depth": {"value": 1.4, "unit": "m"},
            "tf_wp_width": {"value": 0.8, "unit": "m"},
            "tk_tf_side": {"value": 0.05, "unit": "m"},
            "tf_gs_tk_plate": {"value": 0.025, "unit": "m"},
            "tf_gs_g_plate": {"value": 0.025, "unit": "m"},
            "tf_gs_base_depth": {"value": 2.4, "unit": "m"},
        }
        params = ITERGravitySupportBuilderParams.from_dict(defaults)
        return ITERGravitySupportBuilder(params, {}, tf)

    @pytest.mark.parametrize("tf", tf_kozs)
    @pytest.mark.parametrize("x_gs", [0, 2, 3.44, 14.56, 100])
    def test_bad_support_radius(self, tf, x_gs):
        builder = self._make_builder(tf, x_g_support=x_gs)
        with pytest.raises(BuilderError):
            builder.build()

    @pytest.mark.parametrize("tf", tf_kozs)
    @pytest.mark.parametrize("z_gs", [0, -2, 100])
    def test_bad_support_height(self, tf, z_gs):
        builder = self._make_builder(tf, z_gs=z_gs)
        with pytest.raises(BuilderError):
            builder.build()

    @pytest.mark.parametrize("tf", tf_kozs)
    @pytest.mark.parametrize("x_gs", [3.45, 7, 10])
    def test_good_support_radius(self, tf, x_gs):
        builder = self._make_builder(tf, x_g_support=x_gs)
        component = builder.build()

        assert len(component.get_component("xyz").children) == 1
        assert len(component.get_component("xz").children) == 1
        assert len(component.get_component("xy").children) == 0

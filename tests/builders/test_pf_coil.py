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

from unittest import mock

from bluemira.builders.pf_coil import PFCoilBuilder
from bluemira.geometry.tools import make_polygon


class TestPFCoilBuilder:
    @classmethod
    def setup_class(cls):
        # Square as revolving a circle 360 causes an error
        # https://github.com/Fusion-Power-Plant-Framework/bluemira/issues/1090
        cls.square = make_polygon(
            [(1, 0, 1), (3, 0, 1), (3, 0, -1), (1, 0, -1)], closed=True
        )
        cls.params = {
            "tk_insulation": {"value": 0.1, "unit": "m", "name": "tk_insulation"},
            "tk_casing": {"value": 0.2, "unit": "m", "name": "tk_casing"},
            "ctype": {"value": "PF", "unit": "dimensionless", "name": "ctype"},
        }

    def test_component_dimensions_are_built(self):
        designer = mock.Mock(run=lambda: self.square)
        builder = PFCoilBuilder(self.params, {}, designer)

        coil = builder.build()

        assert coil.component().get_component("xz")
        assert coil.component().get_component("xy")
        assert coil.component().get_component("xyz")

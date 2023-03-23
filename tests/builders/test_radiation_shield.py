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
from bluemira.builders.radiation_shield import RadiationShieldBuilder
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon


class TestRadiationShieldBuilder:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "g_cr_rs": {"value": 3, "unit": "m"},
            "tk_rs": {"value": 0.02, "unit": "m"},
            "n_TF": {"value": 12, "unit": ""},
        }
        cls.cryo_koz = BluemiraFace(
            make_polygon([[0, 0, -5], [0, 0, -10], [5, 0, -10], [5, 0, -5]], closed=True)
        )

    def test_components_and_segments(self):
        builder = RadiationShieldBuilder(self.params, {}, self.cryo_koz)
        rad_shield = builder.build()

        assert rad_shield.get_component("xz")
        assert rad_shield.get_component("xy")

        xyz = rad_shield.get_component("xyz")
        assert xyz
        assert len(xyz.leaves) == 1

        xyz.show_cad()

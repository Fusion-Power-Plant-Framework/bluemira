# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
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

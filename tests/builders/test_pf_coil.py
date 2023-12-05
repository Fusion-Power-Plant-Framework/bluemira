# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

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
            "n_TF": {"value": 1, "unit": "dimensionless"},
            "tk_insulation": {"value": 0.1, "unit": "m"},
            "tk_casing": {"value": 0.2, "unit": "m"},
            "ctype": {"value": "PF", "unit": "dimensionless"},
        }

    def test_component_dimensions_are_built(self):
        builder = PFCoilBuilder(self.params, {}, self.square)

        coil = builder.build()

        assert coil.get_component("xz")
        assert coil.get_component("xy")
        assert coil.get_component("xyz")

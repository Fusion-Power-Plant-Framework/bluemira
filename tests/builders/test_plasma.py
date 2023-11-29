# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tests for plasma builder.
"""

from bluemira.base.parameter_frame._parameter import Parameter
from bluemira.builders.plasma import Plasma, PlasmaBuilder, PlasmaBuilderParams
from bluemira.geometry.tools import make_polygon


class TestPlasmaBuilder:
    @classmethod
    def setup_class(cls):
        # Square as revolving a circle 360 causes an error
        # https://github.com/Fusion-Power-Plant-Framework/bluemira/issues/1090
        cls.square = make_polygon(
            [(1, 0, 1), (3, 0, 1), (3, 0, -1), (1, 0, -1)], closed=True
        )

    def test_plasma_contains_components_in_3_dimensions(self):
        builder = PlasmaBuilder(
            PlasmaBuilderParams(n_TF=Parameter(name="n_TF", value=1)),
            {},
            self.square,
        )
        plasma = Plasma(builder.build())

        assert plasma.component().get_component("xz")
        assert plasma.component().get_component("xy")
        assert plasma.component().get_component("xyz")

    def test_lcfs_eq_to_designer_shape(self):
        builder = PlasmaBuilder(
            PlasmaBuilderParams(n_TF=Parameter(name="n_TF", value=1)),
            {},
            self.square,
        )

        plasma = Plasma(builder.build())

        assert plasma.lcfs() == self.square

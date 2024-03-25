# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest

from bluemira.base.constants import raw_uc
from tests.materials.materials_helpers import MATERIAL_CACHE


class TestMatDict:
    def test_wp(self):
        tf = MATERIAL_CACHE.get_material("Toroidal_Field_Coil_2015")

        assert isinstance(tf.E(294), float)
        assert isinstance(tf.CTE(294), float)
        assert isinstance(tf.rho(294), float)
        assert isinstance(tf.mu(294), float)
        assert isinstance(tf.Sy(294), float)

    def test_openmc(self):
        pytest.importorskip("openmc")
        tf = MATERIAL_CACHE.get_material("Toroidal_Field_Coil_2015")
        openmc_mat = tf.to_openmc_material()
        assert openmc_mat.density == pytest.approx(raw_uc(tf.rho(), "kg/m^3", "g/cm^3"))

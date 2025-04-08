# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

from bluemira.base.file import get_bluemira_path
from bluemira.materials import MaterialCache
from bluemira.structural.material import make_structural_material

MATERIAL_DATA_PATH = get_bluemira_path("materials", subfolder="data")
MATERIAL_CACHE = MaterialCache()
MATERIAL_CACHE.load_from_file(Path(MATERIAL_DATA_PATH, "materials.json"))
MATERIAL_CACHE.load_from_file(Path(MATERIAL_DATA_PATH, "mixtures.json"))


class TestMaterialInterface:
    def setup_method(self):
        self.SS316 = MATERIAL_CACHE.get_material("SS316-LN")
        self.SW = MATERIAL_CACHE.get_material("Steel Water 60/40")

    def test_simple(self):
        str_mat = make_structural_material(self.SS316, 300)
        self.check_material_valid(str_mat)

    def test_homogenised(self):
        str_mat = make_structural_material(self.SW, 400)
        self.check_material_valid(str_mat)

    @staticmethod
    def check_material_valid(material):
        for attribute in ["E", "nu", "rho", "alpha", "sigma_y", "G"]:
            value = getattr(material, attribute)
            assert isinstance(value, float)
        assert material.description

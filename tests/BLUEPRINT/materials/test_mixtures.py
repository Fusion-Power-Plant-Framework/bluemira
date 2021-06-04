# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
import pytest

# =============================================================================
# Material mixture utility classes
# =============================================================================
from BLUEPRINT.materials.constants import MATERIAL_BEAM_MAP
from BLUEPRINT.materials import materials_cache


class TestMatDict:
    def test_wp(self):
        tf = materials_cache.get_material("Toroidal_Field_Coil_2015")
        mat_dict = tf.make_mat_dict(294)

        for key in MATERIAL_BEAM_MAP.values():
            assert key in mat_dict


if __name__ == "__main__":
    pytest.main([__file__])

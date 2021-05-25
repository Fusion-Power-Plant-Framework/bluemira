# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.
import pytest

# =============================================================================
# Material mixture utility classes
# =============================================================================
from BLUEPRINT.materials.constants import MATERIAL_BEAM_MAP

from tests.materials.setup_methods import TEST_MATERIALS_CACHE


class TestMatDict:
    def test_wp(self):
        tf = TEST_MATERIALS_CACHE.get_material("Toroidal_Field_Coil_2015")
        mat_dict = tf.make_mat_dict(294)

        for key in MATERIAL_BEAM_MAP.values():
            assert key in mat_dict


if __name__ == "__main__":
    pytest.main([__file__])

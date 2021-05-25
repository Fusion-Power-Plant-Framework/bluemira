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
import os
import pytest
import numpy as np
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.cad.vesselCAD import VesselCAD
from BLUEPRINT.systems.vessel import VacuumVessel


class TestVVCAD:
    def setup_method(self):
        path = get_BP_path("test_data/reactors/SMOKE-TEST", subfolder="tests")
        file = os.sep.join([path, "SMOKE-TEST_VV.pkl"])
        self.vessel = VacuumVessel.load(file)

    def test_vacuumvessel(self):
        v = VesselCAD(self.vessel, neutronics=False)
        volume = v.get_properties()["vessel"]["Volume"]
        assert np.isclose(round(volume, 3), 100.609)


if __name__ == "__main__":
    pytest.main([__file__])

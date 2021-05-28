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

import os
from BLUEPRINT.base.file import get_BP_path, get_files_by_ext
from BLUEPRINT.utilities.tools import compare_dicts
from BLUEPRINT.equilibria.eqdsk import EQDSKInterface


class TestEQDSKInterface:
    path = get_BP_path("equilibria/test_data", subfolder="tests")
    # testfile = get_eqdsk('AR3d1_2015_04_v2_SOF_CSred_fine_final')

    @classmethod
    def setup_class(cls):
        cls.testfiles = get_files_by_ext(cls.path, "eqdsk")
        cls.testfiles += get_files_by_ext(cls.path, "eqdsk_out")

    def test_read(self):
        for f in self.testfiles:
            file = os.sep.join([self.path, f])
            eqdsk = EQDSKInterface()
            eqdsk.read(file)
            d1 = eqdsk.to_dict()
            name = f.split(".")[0] + "_temp"
            fname = os.sep.join([self.path, name])
            eqdsk.write(fname, d1, formatt="eqdsk")
            d2 = eqdsk.to_dict()
            jname = fname.split(".")[0] + ".json"
            eqdsk.write(jname, d1, formatt="json")
            neqdsk = EQDSKInterface()
            d3 = neqdsk.read(jname)
            # Clean up
            os.remove(fname + ".eqdsk")
            os.remove(fname + ".json")
            assert compare_dicts(d1, d2, verbose=True)
            assert compare_dicts(d1, d3, verbose=True)
            assert compare_dicts(d2, d3, verbose=True)


if __name__ == "__main__":
    pytest.main([__file__])

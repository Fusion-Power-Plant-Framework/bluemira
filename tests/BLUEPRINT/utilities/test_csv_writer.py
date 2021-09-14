# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
import numpy as np
from BLUEPRINT.utilities.csv_writer import write_csv
import filecmp
from BLUEPRINT.base.file import get_BP_path


def test_csv_writer():
    x_vals = [0, 1, 2]
    z_vals = [-1, 0, 1]
    flux_vals = [10, 15, 20]
    my_data = np.array([x_vals, z_vals, flux_vals]).T
    my_header = "This is a test"
    col_names = ["x", "z", "heat_flux"]
    read_path = get_BP_path("BLUEPRINT/utilities/test_data", subfolder="tests")

    write_csv(my_data, os.sep.join([read_path, "lets_test"]), col_names, my_header)

    tester = os.sep.join([read_path, "test_csv_writer.csv"])
    tested = os.sep.join([read_path, "lets_test.csv"])
    assert filecmp.cmp(tester, tested)


if __name__ == "__main__":
    pytest.main([__file__])

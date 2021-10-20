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
from BLUEPRINT.utilities.csv_writer import write_csv, write_geometry_to_csv
from BLUEPRINT.geometry.geomtools import make_box_xz
import filecmp
from BLUEPRINT.base.file import get_BP_path


def test_csv_writer():
    # Some dummy data to write to file
    x_vals = [0, 1, 2]
    z_vals = [-1, 0, 1]
    flux_vals = [10, 15, 20]
    data = np.array([x_vals, z_vals, flux_vals]).T
    header = "This is a test\nThis is a second line"
    col_names = ["x", "z", "heat_flux"]

    # Write the data to csv
    test_file_base = "csv_write_dummy_data"
    write_csv(data, test_file_base, col_names, header)

    # Retrieve data file to compare
    data_file = "test_csv_writer.csv"
    data_dir = "BLUEPRINT/utilities/test_data"
    data_path = get_BP_path(data_dir, subfolder="tests")
    compare_file = os.sep.join([data_path, data_file])

    # Compare
    test_file = test_file_base + ".csv"
    assert filecmp.cmp(test_file, compare_file)

    # Clean up
    os.remove(test_file)


def test_write_geometry_to_csv():
    # Define simple loop
    loop = make_box_xz(0.0, 1.0, 0.0, 2.0)

    # Write out the loop
    test_file_base = "loop_test_write"
    metadata = "Metadata string"
    write_geometry_to_csv(loop, test_file_base, metadata)

    # Fetch comparison data file
    data_file = "loop_test_data.csv"
    data_dir = "BLUEPRINT/utilities/test_data"
    compare_path = get_BP_path(data_dir, subfolder="tests")
    compare_file = os.sep.join([compare_path, data_file])

    # Compare generated data to data file
    test_file = test_file_base + ".csv"
    assert filecmp.cmp(test_file, compare_file)

    # Clean up
    os.remove(test_file)


if __name__ == "__main__":
    pytest.main([__file__])

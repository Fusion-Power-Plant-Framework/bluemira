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
from BLUEPRINT.utilities.csv_writer import (
    write_csv,
    write_geometry_to_csv,
    write_components_to_csv,
)
from BLUEPRINT.geometry.geomtools import make_box_xz
import filecmp
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.systems.baseclass import ReactorSystem


def test_csv_writer():
    # Some dummy data to write to file
    x_vals = [0, 1, 2]
    z_vals = [-1, 0, 1]
    flux_vals = [10, 15, 20]
    data = np.array([x_vals, z_vals, flux_vals]).T
    header = "This is a test\nThis is a second line"
    col_names = ["x", "z", "heat_flux"]

    # Write the data to csv, using default extension and comment style
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

    # Write the data to csv, using modified extension and comment style
    ext = ".txt"
    comment_char = "!"
    write_csv(data, test_file_base, col_names, header, ext, comment_char)

    # Retrieve data file to compare
    data_file = "test_csv_writer.txt"
    compare_file = os.sep.join([data_path, data_file])

    # Compare
    test_file = test_file_base + ext
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


class CSVWriterReactorSystem(ReactorSystem):
    """
    ReactorSystem to test csv write functionality.
    """

    def __init__(self):
        self.geom["Dummy Loop"] = make_box_xz(0.0, 1.0, 0.0, 1.0)
        self.geom["dUmmy looP 2"] = make_box_xz(0.0, 1.0, 0.0, 1.0)


def test_reactor_system_write_to_csv():
    # Create an instance of our dummy class
    csv_system = CSVWriterReactorSystem()

    # Get component keys
    component_names = csv_system.geom.keys()

    # Test write with all default arguments
    test_file_base = "reactor_system"
    write_components_to_csv(csv_system, component_names, test_file_base)

    # Fetch comparison data file
    data_file = "reactor_system_dummy_loop_no_metadata.csv"
    data_dir = "BLUEPRINT/utilities/test_data"
    compare_path = get_BP_path(data_dir, subfolder="tests")
    compare_file = os.sep.join([compare_path, data_file])

    # Compare against test file
    test_keys = ["dummy_loop", "dummy_loop_2"]
    for key in test_keys:
        test_file = test_file_base + "_" + key + ".csv"
        assert filecmp.cmp(test_file, compare_file)
        # Clean up
        os.remove(test_file)

    # Test write, specifying file_base, path and metadata
    metadata = "Metadata string"
    write_components_to_csv(
        csv_system, component_names, test_file_base, compare_path, metadata
    )

    # Fetch comparison data file
    data_file = "reactor_system_dummy_loop_with_metadata.csv"
    compare_file = os.sep.join([compare_path, data_file])

    # Compare against test file
    for key in test_keys:
        test_file = os.sep.join([compare_path, test_file_base + "_" + key + ".csv"])
        assert filecmp.cmp(test_file, compare_file)
        # Clean up
        os.remove(test_file)


if __name__ == "__main__":
    pytest.main([__file__])

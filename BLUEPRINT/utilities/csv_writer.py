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

"""
csv writer utilities
"""
import numpy as np
from bluemira.base.look_and_feel import bluemira_print


def write_csv(data, base_name, col_names, metadata="", ext=".csv"):
    """
    Write data in comma-separated value format.

    Parameters
    ----------
    data : numpy.ndarray
        Array of data to be written to csv file. Will raise an error if the
        dimensionality of the data is not two
    base_name : str
        Name of file to write to, minus the extension.
    col_names : list(str)
        List of strings for column headings for each data field provided.
    metadata: str
        Optional argument for metadata to be written as a header.
    ext : string
        Optional argument for file extension, defaults to ".csv".
    """
    # Fetch number of cols
    shape = data.shape
    n_cols = 1 if len(shape) < 2 else shape[1]

    # Write file name
    filename = base_name + ext

    # Write column names
    if not len(col_names) == n_cols:
        raise RuntimeError("Column names must be provided for all data fields")

    # Add a newline to existing metadata
    if metadata != "":
        metadata += "\n"

    # Add column headings
    metadata += ",".join(col_names)

    np.savetxt(
        filename,
        data,
        fmt="%.5e",
        delimiter=",",
        header=metadata,
        footer="",
        comments="",
    )
    bluemira_print("Wrote to " + filename)

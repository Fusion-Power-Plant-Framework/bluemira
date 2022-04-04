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

import os

import fortranformat as ff
import numpy as np

from bluemira.base.file import get_bluemira_path, get_files_by_ext
from bluemira.equilibria.file import EQDSKInterface
from bluemira.utilities.tools import compare_dicts


class TestEQDSKInterface:
    path = get_bluemira_path("bluemira/equilibria/test_data", subfolder="tests")

    @classmethod
    def setup_class(cls):
        cls.testfiles = get_files_by_ext(cls.path, "eqdsk")
        cls.testfiles += get_files_by_ext(cls.path, "eqdsk_out")

    def read_strict_geqdsk(self, fname):
        """
        Reads an input EQDSK file in, assuming strict adherence to the
        GEQDSK format. Used to check bluemira outputs can be read by
        external readers.

        Note: The main bluemira GEQDSK reader is more forgiving to
        format variations than this!

        Parameters
        ----------
        fname: str
            Full path string of the file, without the ".eqdsk" extension.

        """

        # Create FortranRecordReader objects with the Fortran format
        # edit descriptors to be used to parse the G-EQDSK input.
        f2000 = ff.FortranRecordReader("a48,3i4")
        f2020 = ff.FortranRecordReader("5e16.9")
        f2022 = ff.FortranRecordReader("2i5")
        fCSTM = ff.FortranRecordReader("i5")

        # Define helper function to read in flattened arrays.
        def read_flat_array(fortran_format, array_size):
            """
            Reads in a flat (1D) numpy array from a G-EQDSK file.

            Parameters
            ----------
            fortran_format: ff.FortranRecordReader
                FortranRecordReader object for Fortran format edit descriptor
                to be used to parse the format of each line of the output.
            array_size: int
                Number of elements in array to be read in.

            Returns
            -------
            array: np.array
                1D Numpy array of length array_size populated by elements from
                the GEQDSK file input.
            """

            # Initialise numpy array and read in first line.
            array = np.zeros((array_size,))
            line = fortran_format.read(file.readline())
            # Define a counter to track which column in a line
            # is currently being saved into the array.
            col = 0
            # Populate array. If the column index moves past the
            # end of a line, it is reset to zero and the next line is read.
            for i in range(array_size):
                if col == len(line):
                    line = fortran_format.read(file.readline())
                    col = 0
                array[i] = line[col]
                col += 1
            return array

        # Open file.
        file = open(fname + ".eqdsk", "r")

        # Read in data. Variable names are for readability;
        # strict format is as defined at
        # https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
        id, _, nx, nz = f2000.read(file.readline())
        xdim, zdim, xcentre, xgrid1, zmid = f2020.read(file.readline())
        xmag, zmag, psimag, psibdry, bcentre = f2020.read(file.readline())
        cplasma, psimag, _, xmag, _ = f2020.read(file.readline())
        zmag, _, psibdry, _, _ = f2020.read(file.readline())
        fpol = read_flat_array(f2020, nx)
        pressure = read_flat_array(f2020, nx)
        ffprime = read_flat_array(f2020, nx)
        pprime = read_flat_array(f2020, nx)
        psi = read_flat_array(f2020, nx * nz)
        qpsi = read_flat_array(f2020, nx)
        nbdry, nlim = f2022.read(file.readline())
        xbdry_zbdry = read_flat_array(f2020, 2 * nbdry)
        xlim_zlim = read_flat_array(f2020, 2 * nlim)

        # Read in coil information, as found in the GEQDSK extension
        # used by bluemira.
        (ncoil,) = fCSTM.read(file.readline())
        coil = read_flat_array(f2020, 5 * ncoil)

    def test_read(self):
        # Loop over all test files
        for f in self.testfiles:
            # Define absolute path of current test file
            file = os.sep.join([self.path, f])

            # Create EQDSK file interface and read data to a dict
            eqdsk = EQDSKInterface()
            eqdsk.read(file)
            d1 = eqdsk.to_dict()

            # Write data read in from test file into a new EQDSK
            # file, with the suffix "_temp"
            name = f.split(".")[0] + "_temp"
            fname = os.sep.join([self.path, name])
            eqdsk.write(fname, d1, formatt="eqdsk")
            d2 = eqdsk.to_dict()

            # Check eqdsk is readable by Fortran readers.
            # This demands stricter adherence to the G-EQDSK
            # format than bluemira's main reader.
            self.read_strict_geqdsk(fname)

            # Write data read in from test file into a new JSON
            # file, with the suffix "_temp"
            jname = fname.split(".")[0] + ".json"
            eqdsk.write(jname, d1, formatt="json")
            neqdsk = EQDSKInterface()
            d3 = neqdsk.read(jname)

            # Clean up temporary files
            os.remove(fname + ".eqdsk")
            os.remove(fname + ".json")

            # Compare dictionaries to check data hasn't
            # been changed.
            assert compare_dicts(d1, d2, verbose=True)
            assert compare_dicts(d1, d3, verbose=True)
            assert compare_dicts(d2, d3, verbose=True)

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
Input and output file interface. EQDSK and json. NOTE: jsons are better :)
"""
import json
import os
import time

import fortranformat as ff
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter import ParameterFrame
from bluemira.utilities.tools import is_num, json_writer

__all__ = ["EQDSKInterface"]


def read_array(tokens, n, name="Unknown"):
    data = np.zeros([n])
    try:
        for i in np.arange(n):
            data[i] = float(next(tokens))
    except StopIteration:
        raise ValueError(f"Failed reading array {name} of size {n}")
    return data


def read_2d_array(tokens, n_x, n_y, name="Unknown"):
    data = np.zeros([n_y, n_x])
    for i in np.arange(n_y):
        data[i, :] = read_array(tokens, n_x, name + "[" + str(i) + "]")
    data = np.transpose(data)
    return data


class EQDSKInterface:
    """
    Provides an interface to EQDSK files (G-EQDSK format)

    Inspired by an EQDSK reader originally developed by B. Dudson:
        https://github.com/bendudson/pyTokamak/blob/master/tokamak/formats/geqdsk.py

    The G-EQDSK file format is described here:
        https://fusion.gat.com/conferences/snowmass/working/mfe/physics/p3/equilibria/g_eqdsk_s.pdf

    Notes
    -----
    G-EQDSK is from the 1980's and EQDSK files should generally only be read and not written.
    New equilibria should really just be saved as JSON files.

    Poloidal magnetic flux units not enforced here!

    Plasma current direction is not enforced here!
    """  # noqa :W505

    # fmt: off
    p = [
        ["name", "Name of the equilibrium EQDSK", None, "dimensionless", None, "eqfile"],
        ["nx", "Number of grid points in the radial direction", None, "dimensionless", None, "eqfile"],
        ["nz", "Number of grid points in the vertical direction", None, "dimensionless", None, "eqfile"],
        ["xdim", "Horizontal dimension of the spatial grid", None, "m", None, "eqfile"],
        ["zdim", "Vertical dimension of the spatial grid", None, "m", None, "eqfile"],
        ["xcentre", "Radius of the reference toroidal magnetic field", None, "m", None, "eqfile"],
        ["xgrid1", "Minimum radius of the spatial grid", None, "m", None, "eqfile"],
        ["zmid", "Z coordinate of the middle of the spatial grid", None, "m", None, "eqfile"],
        ["xmag", "Radius of the magnetic axis", None, "m", None, "eqfile"],
        ["zmag", "Z coordinate of the magnetic axis", None, "m", None, "eqfile"],
        ["psimag", "Poloidal flux at the magnetic axis", None, "V.s/rad", None, "eqfile"],
        ["psibdry", "Poloidal flux at the magnetic axis", None, "V.s/rad", None, "eqfile"],
        ["bcentre", "Magnetic field at the reference radius", None, "T", None, "eqfile"],
        ["cplasma", "Plasma current", None, "A", None, "eqfile"],
        ["fpol", "Poloidal current function f = R*B on 1-D flux grid", None, "T.m", None, "eqfile"],
        ["pressure", "Plasma pressure function on 1-D flux grid", None, "N/m^2", None, "eqfile"],
        ["ffprime", "FF' function on 1-D flux grid", None, "m.T^2/V.s/rad", None, "eqfile"],
        ["pprime", "P' function on 1-D flux grid", None, "N/m^2/V.s/rad", None, "eqfile"],
        ["psi", "Poloidal magnetic flux on the 2-D grid", None, "V.s/rad", None, "eqfile"],
        ["qpsi", "Safety factor values on the 1-D flux grid", None, "dimensionless", None, "eqfile"],
        ["nbdry", "Number of boundary points", None, "dimensionless", None, "eqfile"],
        ["nlim", "Number of limiters", None, "dimensionless", None, "eqfile"],
        ["xbdry", "X coordinates of the plasma boundary", None, "m", None, "eqfile"],
        ["zbdry", "Z coordinates of the plasma boundary", None, "m", None, "eqfile"],
        ["xlim", "X coordinates of the limiters", None, "m", None, "eqfile"],
        ["zlim", "Z coordinates of the limiters", None, "m", None, "eqfile"],
        ["ncoil", "Number of coils", None, "dimensionless", None, "eqfile"],
        ["xc", "X coordinates of the coils", None, "m", None, "eqfile"],
        ["zc", "Z coordinates of the coils", None, "m", None, "eqfile"],
        ["dxc", "X half-thicknesses of the coils", None, "m", None, "eqfile"],
        ["dzc", "Z half-thicknesses of hte coils", None, "m", None, "eqfile"],
        ["Ic", "Coil currents", None, "A", None, "eqfile"],
        ["x", "X 1-D vector", None, "m", None, "eqfile"],
        ["z", "Z 1-D vector", None, "m", None, "eqfile"],
        ["psinorm", "Normalised psi vector", None, "A", None, "eqfile"],
    ]
    # fmt: on
    data = ParameterFrame(p)

    def __init__(self):
        self.filename = None
        self.header = None

    def read(self, file):
        """
        Reads an input EQDSK file in. Can be in EQDSK or JSON format.

        Parameters
        ----------
        file: str
            Full path string of the file

        Returns
        -------
        data: dict
            The dictionary of EQDSK data specified in the ParameterFrame
        """
        self.filename = self._get_filename(file)
        if (
            file.endswith(".eqdsk")
            or file.endswith(".eqdsk_out")
            or file.endswith(".geqdsk")
        ):
            return self._read_eqdsk(file)
        if file.endswith(".json"):
            return self._read_json(file)
        formatt = file.split(".")[-1]
        raise ValueError(f"Unrecognised file format {formatt}")

    def write(self, file, data, formatt="json", **kwargs):
        """
        Writes an output EQDSK file

        Parameters
        ----------
        file: str
            The full path string of the file to be created
        data: dict
            The dictionary of EQDSK data specified in the ParameterFrame
        formatt: str
            The format to save the file in
        kwargs: dict
            passed onto the writer
        """
        self.filename = self._get_filename(file)
        if formatt == "json":
            self._write_json(file, data, **kwargs)
        elif formatt in ["eqdsk", "geqdsk"]:
            bluemira_warn(
                "You are in the 21st century. Are you sure you want to be making an EDQSK in this day and age?"
            )
            self._write_eqdsk(file, data, **kwargs)
        else:
            raise ValueError(f"Unrecognised file format {formatt}")

    @staticmethod
    def _get_filename(file):
        return file.split(os.sep)[-1]

    def to_dict(self):
        """
        Produces a dictionary of the EQDSK information

        Returns
        -------
        eqdsk_dict: dict
            The dictionary of all the EQDSK information
        """
        return dict(self.data.items())

    def load_dict(self, eqdsk_dict):
        """
        Loads a dictionary of EQDSK information and updates the EQDSKInterface

        Parameters
        ----------
        eqdsk_dict: dict
            The dictionary of updated EQDSK information
        """
        self.data.update_kw_parameters(eqdsk_dict, f"Loaded from {self.filename}")

    def set_attr(self, name, value):
        """
        Sets an attribute of the underlying ParameterFrame

        Parameters
        ----------
        name: str
            A key of the ParameterFrame
        value: any type
            The value of the parameter to set
        """
        if name in self.data.keys():
            self.data.set_parameter(name, value, source=self.header)
        else:
            raise KeyError(f'No parameter "{name}" in ParameterFrame')

    def _read_json(self, file):
        if isinstance(file, str):
            with open(file, "r") as f_h:
                return self._read_json(f_h)

        data = json.load(file)
        for k, value in data.items():
            if isinstance(value, list):
                data[k] = np.asarray(value)

        self.load_dict(data)
        return self.to_dict()

    def _write_json(self, file, data, **kwargs):
        return json_writer(data, file, **kwargs)

    def _read_eqdsk(self, file):
        if isinstance(file, str):
            with open(file, "r") as f_handle:
                return self._read_eqdsk(f_handle)

        description = file.readline()
        if not description:
            raise IOError(f"Could not read the file {self.filename}.")
        description = description.split()
        self.header = description[0]

        ints = []
        for value in description:
            if is_num(value):
                ints.append(value)
        if len(ints) < 3:
            raise IOError(
                "Should be at least 3 numbers in the first line "
                f"of the EQDSK {self.filename}."
            )

        n_x = int(ints[-2])
        n_z = int(ints[-1])
        self.set_attr("name", description[0])
        self.set_attr("nx", n_x)
        self.set_attr("nz", n_z)

        tokens = self.eqdsk_generator(file)
        for name in [
            "xdim",
            "zdim",
            "xcentre",
            "xgrid1",
            "zmid",
            "xmag",
            "zmag",
            "psimag",
            "psibdry",
            "bcentre",
            "cplasma",
            "psimag",
            None,
            "xmag",
            None,
            "zmag",
            None,
            "psibdry",
            None,
            None,
        ]:
            if name is not None:  # Lots of dummies and duplication
                self.set_attr(name, float(next(tokens)))
            else:
                next(tokens)  # Dummy

        for name in ["fpol", "pressure", "ffprime", "pprime"]:
            self.set_attr(name, read_array(tokens, n_x, name))

        self.set_attr("psi", read_2d_array(tokens, n_x, n_z, "psi"))
        self.set_attr("qpsi", read_array(tokens, n_x, "qpsi"))
        nbdry = int(next(tokens))
        nlim = int(next(tokens))
        self.set_attr("nbdry", nbdry)
        self.set_attr("nlim", nlim)

        if nbdry > 0:
            xbdry = np.zeros([nbdry])
            zbdry = np.zeros([nbdry])
            for i in range(nbdry):
                xbdry[i] = float(next(tokens))
                zbdry[i] = float(next(tokens))
        else:
            xbdry = [0]
            zbdry = [0]
        self.set_attr("xbdry", xbdry)
        self.set_attr("zbdry", zbdry)

        if nlim > 0:
            xlim = np.zeros([nlim])
            zlim = np.zeros([nlim])
            for i in range(nlim):
                xlim[i] = float(next(tokens))
                zlim[i] = float(next(tokens))
            self.set_attr("xlim", xlim)
            self.set_attr("zlim", zlim)

        try:
            ncoil = int(next(tokens))
        except StopIteration:  # No coils in file
            ncoil, x_c, z_c, dxc, dzc, i_c = 0, 0, 0, 0, 0, 0
        if ncoil > 0:
            x_c = np.zeros(ncoil)
            z_c = np.zeros(ncoil)
            dxc = np.zeros(ncoil)
            dzc = np.zeros(ncoil)
            i_c = np.zeros(ncoil)
            for i in range(ncoil):
                x_c[i] = float(next(tokens))
                z_c[i] = float(next(tokens))
                dxc[i] = float(next(tokens))
                dzc[i] = float(next(tokens))
                i_c[i] = float(next(tokens))
        else:
            # No coils in file
            x_c, z_c, dxc, dzc, i_c = 0, 0, 0, 0, 0
        self.set_attr("ncoil", ncoil)
        self.set_attr("xc", x_c)
        self.set_attr("zc", z_c)
        self.set_attr("dxc", dxc)
        self.set_attr("dzc", dzc)
        self.set_attr("Ic", i_c)
        # Additional utility data
        x = np.linspace(
            self.data["xgrid1"],
            self.data["xgrid1"] + self.data["xdim"],
            self.data["nx"],
        )
        z = np.linspace(
            self.data["zmid"] - self.data["zdim"] / 2,
            self.data["zmid"] + self.data["zdim"] / 2,
            self.data["nz"],
        )
        psinorm = np.linspace(0, 1, len(self.data["fpol"]))
        self.set_attr("x", x)
        self.set_attr("z", z)
        self.set_attr("psinorm", psinorm)
        return self.to_dict()

    @staticmethod
    def eqdsk_generator(file):
        """
        Transforms a file object into a generator, following G-EQDSK number
        conventions

        Parameters
        ----------
        file: file object
            The file to read

        Returns
        -------
        generator: generator
            The generator of the file handle being read
        """
        while True:
            line = file.readline()
            if not line:
                break

            # Distinguish negative/positive numbers from negative/positive exponent
            if "E" in line or "e" in line:
                line = line.replace("E-", "*")
                line = line.replace("e-", "*")
                line = line.replace("-", " -")
                line = line.replace("*", "e-")
                line = line.replace("E+", "*")
                line = line.replace("e+", "*")
                line = line.replace("+", " ")
                line = line.replace("*", "e+")
            generator_list = line.split()
            for obj in generator_list:
                yield obj

    def _write_eqdsk(self, file, data, **kwargs):
        """
        Writes data out to a text file in G-EQDSK format.

        Parameters
        ----------
        file: str
            The full path string of the file to be created
        data: dict
            The dictionary of EQDSK data specified in the ParameterFrame
        kwargs: dict
            The kwargs do not provide any options here
        """
        if isinstance(file, str):
            if not file.endswith(".eqdsk") or not file.endswith(".geqdsk"):
                file = file.split(".")[0] + ".eqdsk"
            with open(file, "w") as f_handle:
                return self._write_eqdsk(f_handle, data)
        self.load_dict(data)

        def write_header(fortran_format, id_string, var_list):
            """
            Writes G-EQDSK header out to file.

            Parameters
            ----------
            fortran_format: ff.FortranRecordWriter
                FortranRecordWriter object for Fortran format edit descriptor
                to be used for header output.
            id_string: str
                String containing name of file to be used as identification
                string. Will be trimmed if length exceeds 39 characters,
                so it will fit within the permitted header length of the
                GEQDSK specification when a timestamp is added.
            var_list: list
                List of names of keys in EQDSKInterface.data identifying
                variables to add to the header following the id_string.
                Empty strings will be recorded as 0.
            """
            line = [id_string]
            line += [self.data[v] if v != "" else 0 for v in var_list]
            file.write(fortran_format.write(line))
            file.write("\n")

        def write_line(fortran_format, var_list):
            """
            Writes a line of variable values out to a G-EQDSK file.

            Parameters
            ----------
            fortran_format: ff.FortranRecordWriter
                FortranRecordWriter object for Fortran format edit descriptor
                to be used for the format of the line output.
            var_list: list
                List of names of keys in EQDSKInterface.data identifying
                variables to added to the current line.
                Empty strings will be recorded as 0.
            """
            line = [self.data[v] if v != "" else 0 for v in var_list]
            file.write(fortran_format.write(line))
            file.write("\n")

        def write_array(fortran_format, array):
            """
            Writes a numpy array out to a G-EQDSK file.

            Parameters
            ----------
            fortran_format: ff.FortranRecordWriter
                FortranRecordWriter object for Fortran format edit descriptor
                to be used for the format of the line output.
            array: np.array
                Numpy array of variables to be written to file.
                Array will be flattened in column-major (Fortran)
                order if is more than one-dimensional.
            """
            if array.ndim > 1:
                flat_array = array.flatten(order="F")
                file.write(fortran_format.write(flat_array))
            else:
                file.write(fortran_format.write(array))
            file.write("\n")

        # Create id string for file comprising of timestamp and trimmed filename
        # that fits the 48 character limit of strings in EQDSK headers.
        timestamp = time.strftime("%d%m%Y")
        trimmed_name = self.data["name"][0 : 48 - len(timestamp) - 1]
        file_id_string = "_".join([trimmed_name, timestamp])

        # Define dummy data for qpsi if it has not been previously defined.
        if self.data["qpsi"] is None:
            qpsi = np.zeros(self.data["nx"])
        else:
            qpsi = self.data["qpsi"]

        # Create array containing coilset information.
        coil = np.zeros(5 * self.data["ncoil"])
        for i, value in enumerate(["xc", "zc", "dxc", "dzc", "Ic"]):
            coil[i::5] = self.data[value]

        # Create FortranRecordWriter objects with the Fortran format
        # edit descriptors to be used in the G-EQDSK output.
        f2000 = ff.FortranRecordWriter("a48,3i4")
        f2020 = ff.FortranRecordWriter("5e16.9")
        f2022 = ff.FortranRecordWriter("2i5")
        fCSTM = ff.FortranRecordWriter("i5")

        # Write header in f2000 (6a8,3i4) format.
        write_header(f2000, file_id_string, ["", "nx", "nz"])
        # Write out lines containing floats in f2020 (5e16.9) format.
        write_line(f2020, ["xdim", "zdim", "xcentre", "xgrid1", "zmid"])
        write_line(f2020, ["xmag", "zmag", "psimag", "psibdry", "bcentre"])
        write_line(f2020, ["cplasma", "psimag", "", "xmag", ""])
        write_line(f2020, ["zmag", "", "psibdry", "", ""])
        # Write out arrays in in f2020 (5e16.9) format.
        write_array(f2020, self.data["fpol"])
        write_array(f2020, self.data["pressure"])
        write_array(f2020, self.data["ffprime"])
        write_array(f2020, self.data["pprime"])
        write_array(f2020, self.data["psi"])
        write_array(f2020, qpsi)
        # Write out number of boundary points and limiters f2022 (2i5) format.
        write_line(f2022, ["nbdry", "nlim"])
        # Write out boundary point and limiter data as array of ordered pairs.
        write_array(f2020, np.array([self.data["xbdry"], self.data["zbdry"]]))
        write_array(f2020, np.array([self.data["xlim"], self.data["zlim"]]))

        # Output of coilset information. This is an extension to the
        # regular eqdsk format.
        write_line(fCSTM, ["ncoil"])
        write_array(f2020, coil)

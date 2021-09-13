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
import os
import json
import time
from itertools import count
import numpy as np

from BLUEPRINT.base.parameter import ParameterFrame
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.tools import is_num, NumpyJSONEncoder

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
    """  # noqa (W505)

    # fmt: off
    p = [
        ["name", "Name of the equilibrium EQDSK", None, "N.A", None, None],
        ["nx", "Number of grid points in the radial direction", None, "N.A", None, None],
        ["nz", "Number of grid points in the vertical direction", None, "N.A", None, None],
        ["xdim", "Horizontal dimension of the spatial grid", None, "m", None, None],
        ["zdim", "Vertical dimension of the spatial grid", None, "m", None, None],
        ["xcentre", "Radius of the reference toroidal magnetic field", None, "m", None, None],
        ["xgrid1", "Minimum radius of the spatial grid", None, "m", None, None],
        ["zmid", "Z coordinate of the middle of the spatial grid", None, "m", None, None],
        ["xmag", "Radius of the magnetic axis", None, "m", None, None],
        ["zmag", "Z coordinate of the magnetic axis", None, "m", None, None],
        ["psimag", "Poloidal flux at the magnetic axis", None, "V.s/rad", None, None],
        ["psibdry", "Poloidal flux at the magnetic axis", None, "V.s/rad", None, None],
        ["bcentre", "Magnetic field at the reference radius", None, "T", None, None],
        ["cplasma", "Plasma current", None, "A", None, None],
        ["fpol", "Poloidal current function f = R*B on 1-D flux grid", None, "T.m", None, None],
        ["pressure", "Plasma pressure function on 1-D flux grid", None, "n.t/m^2", None, None],
        ["ffprime", "FF' function on 1-D flux grid", None, "m.T^2/V.s/rad", None, None],
        ["pprime", "P' function on 1-D flux grid", None, "n.t/m^2/V.s/rad", None, None],
        ["psi", "Poloidal magnetic flux on the 2-D grid", None, "V.s/rad", None, None],
        ["qpsi", "Safety factor values on the 1-D flux grid", None, "N.A", None, None],
        ["nbdry", "Number of boundary points", None, "N.A", None, None],
        ["nlim", "Number of limiters", None, "N.A", None, None],
        ["xbdry", "X coordinates of the plasma boundary", None, "m", None, None],
        ["zbdry", "Z coordinates of the plasma boundary", None, "m", None, None],
        ["xlim", "X coordinates of the limiters", None, "m", None, None],
        ["zlim", "Z coordinates of the limiters", None, "m", None, None],
        ["ncoil", "Number of coils", None, "N/A", None, None],
        ["xc", "X coordinates of the coils", None, "m", None, None],
        ["zc", "Z coordinates of the coils", None, "m", None, None],
        ["dxc", "X half-thicknesses of the coils", None, "m", None, None],
        ["dzc", "Z half-thicknesses of hte coils", None, "m", None, None],
        ["Ic", "Coil currents", None, "A", None, None],
        ["x", "X 1-D vector", None, "m", None, None],
        ["z", "Z 1-D vector", None, "m", None, None],
        ["psinorm", "Normalised psi vector", None, "A", None, None],
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

    def write(self, file, data, formatt="json"):
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
        """
        self.filename = self._get_filename(file)
        if formatt == "json":
            self._write_json(file, data)
        elif formatt in ["eqdsk", "geqdsk"]:
            bluemira_warn(
                "You are in the 21st century. Are you sure you want to be making an EDQSK in this day and age?"
            )
            self._write_eqdsk(file, data)
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
        self.data.update_kw_parameters(eqdsk_dict)

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

    def _write_json(self, file, data):
        if isinstance(file, str):
            with open(file, "w") as f_handle:
                return self._write_json(f_handle, data)
        json.dump(data, file, cls=NumpyJSONEncoder)

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

    def _write_eqdsk(self, file, data):
        if isinstance(file, str):
            if not file.endswith(".eqdsk") or not file.endswith(".geqdsk"):
                file = file.split(".")[0] + ".eqdsk"
            with open(file, "w") as f_handle:
                return self._write_eqdsk(f_handle, data)
        counter = count(0)
        self.load_dict(data)

        def carriage_return(i):
            if np.mod(i + 1, 5) == 0:
                file.write("\n")
            else:
                file.write(" ")

        def write_line(var):
            for i, value in enumerate(var):
                fmat = "{:16.9f}" if value != "cplasma" else "{:16.9e}"
                if not value:  # Empty
                    num = 0
                else:
                    num = self.data[value]
                file.write(fmat.format(num))
                carriage_return(i)

        def write_array(val, count_on):
            if np.size(val) == 1:
                file.write("{:16.9e}".format(val))
                carriage_return(next(count_on))
                return
            for value in val:
                file.write("{:16.9e}".format(value))
                carriage_return(next(count_on))

        def write_2d_array(val, count_on):
            for value in val.T:
                write_array(value, count_on)

        header = "_".join([self.data["name"], time.strftime("%d%m%Y")])
        file.write(f"{header:48s} ")
        file.write(f'{0:4d} {self.data["nx"]:4d} {self.data["nz"]:4d}\n')
        write_line(["xdim", "zdim", "xcentre", "xgrid1", "zmid"])
        write_line(["xmag", "zmag", "psimag", "psibdry", "bcentre"])
        write_line(["cplasma", "psimag", "", "xmag", ""])
        write_line(["zmag", "", "psibdry", "", ""])

        write_array(self.data["fpol"], counter)
        write_array(self.data["pressure"], counter)
        write_array(self.data["ffprime"], counter)
        write_array(self.data["pprime"], counter)
        write_2d_array(self.data["psi"], counter)
        if self.data["qpsi"] is None:
            qpsi = np.zeros(self.data["nx"])
        else:
            qpsi = self.data["qpsi"]
        write_array(qpsi, counter)

        file.write(f'{self.data["nbdry"]:5d} {self.data["nlim"]:5d}\n')
        bdry = np.zeros(2 * self.data["nbdry"])
        bdry[::2], bdry[1::2] = self.data["xbdry"], self.data["zbdry"]
        write_array(bdry, counter)
        lim = np.zeros(2 * self.data["nlim"])
        lim[::2], lim[1::2] = self.data["xlim"], self.data["zlim"]
        write_array(lim, counter)

        file.write(f'{self.data["ncoil"]:5d}\n')
        coil = np.zeros(5 * self.data["ncoil"])
        for i, value in enumerate(["xc", "zc", "dxc", "dzc", "Ic"]):
            coil[i::5] = self.data[value]
        write_array(coil, counter)

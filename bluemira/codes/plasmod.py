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
The bluemira Plasmod wrapper.
"""
import os
from .wrapper import RunMode


class PlasmodInputWriter:
    """
    Bluemira writer for Plasmod input.

    Parameters
    ----------
    template_indat: str
        Path to the IN.DAT file to use as the template for Plasmod parameters.
    """

    def __init__(self, template_indat=DEFAULT_INDAT):
        if os.path.isfile(template_indat):
            # InDat autoloads IN.DAT without checking for existence
            super().__init__(filename=template_indat)
        else:
            super().__init__(filename=None)
            self.filename = template_indat


class PlasmodRunMode(RunMode):
    def _prominence(self):
        pass

    def _batch(self):
        pass

    def _mock(self):
        pass


class Setup:
    """
    This class set up the Plasmod environment and parameters.

    Parameters
    ----------
    runmode: str
        The running method for plasmod. See bluemira.codes.plasmod.RunMode
        for possible values.
    save_path: str
        path to save plasmod input and output data
    """

    def __init__(self, runmode, save_path="data/plasmod"):
        self.set_runmode(runmode)
        self.save_path = save_path
        self.runmode(self)

    def set_runmode(self, runmode):
        self.runmode = RunMode[runmode]


class Run:
    def __init__(self, setup):
        self.result = setup.runmode(self, *setup.args)


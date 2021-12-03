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
The bluemira external code wrapper
"""
from __future__ import annotations

import subprocess
import string
from enum import Enum

import bluemira.base as bm_base
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.codes.error import CodesError
from bluemira.codes.utilities import LogPipe, get_recv_mapping, get_send_mapping

__all__ = ["FileProgramInterface", "ApplicationProgramInterface"]


class RunMode(Enum):
    """Defines the possible runmode"""

    def __call__(self, obj, *args, **kwargs):
        """
        Call function of object with lowercase name of
        enum
        Parameters
        ----------
        obj: instance
            instance of class the function will come from. If object is not specified,
            self will be used.
        *args
           args of function
        **kwargs
           kwargs of function
        Returns
        -------
        function result
        """
        if obj is not None:
            func = getattr(obj, f"_{self.name.lower()}")
            return func(*args, **kwargs)


class Task:
    """
    A class for any task integration
    """

    # todo: ensure a correspondence between the specified runmode and the implemented
    #  functions (if possible).
    def __init__(self, parent):
        self.parent = parent
        self._run_dir = parent._run_dir

    def _run_subprocess(self, command, **kwargs):
        stdout = LogPipe("print")
        stderr = LogPipe("error")

        kwargs["cwd"] = kwargs.get("cwd", self._run_dir)
        kwargs.pop("shell", None)  # Protect against user input

        with subprocess.Popen(  # noqa (S603)
            command, stdout=stdout, stderr=stderr, **kwargs
        ) as s:
            stdout.close()
            stderr.close()

        if s.returncode:
            raise CodesError(f"{self.parent.NAME} exited with a non zero exit code")


class Setup(Task):
    """A class that specified the code setup"""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)

    def set_parameters(self):
        pass


class Run(Task):
    """A class that specified the code run process"""

    _binary = None

    def __init__(self, parent, binary=None, *args, **kwargs):
        super().__init__(parent)
        if binary is not None:
            self._binary = binary
        if self._binary is None:
            raise CodesError(f"Binary for {self.parent.NAME} not defined")


class Teardown(Task):
    """A class that for the teardown"""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)

    def get_parameters(self):
        pass


class FileProgramInterface:
    """An external code wrapper"""

    _setup = Setup
    _run = Run
    _teardown = Teardown
    _runmode = RunMode

    def __init__(
        self, NAME, params, runmode, *args, run_dir=None, default_mappings=None, **kwargs
    ):
        self.NAME = NAME

        if default_mappings is not None:
            find_mappings(NAME, params, default_mappings)

        if NAME != "PLASMOD":  # TODO FIX
            self._parameter_mapping = get_recv_mapping(params, NAME, recv_all=True)
            self._params = type(params).from_template(self._parameter_mapping.values())
            self._params.update_kw_parameters(params.to_dict(verbose=True))
            self._recv_mapping = get_recv_mapping(params, NAME)
            self._send_mapping = get_send_mapping(params, NAME)

        if not hasattr(self, "_run_dir") and run_dir is None:
            self._run_dir = "./"

        if self._runmode is not RunMode:
            self._set_runmode(runmode)
        else:
            raise CodesError("Please define a RunMode child lass")

        if self._setup is None:
            self._setup = Setup

        if self._run is None:
            self._run = Run

        if self._teardown is None:
            self._teardown = Teardown

        self.setup_obj = self._setup(self, *args, **kwargs)
        self.run_obj = self._run(self, *args, **kwargs)
        self.teardown_obj = self._teardown(self, *args, **kwargs)

    @property
    def binary(self):
        return self.run_obj._binary

    @binary.setter
    def binary(self, _binary):
        self.run_obj._binary = _binary

    def _set_runmode(self, runmode):
        """Set the runmode"""
        mode = runmode.upper().translate(str.maketrans("", "", string.whitespace))
        self._runner = self._runmode[mode]

    @property
    def params(self) -> bm_base.ParameterFrame:
        """
        The ParameterFrame corresponding to this run.
        """
        return self._params

    def run(self):
        self._runner(self.setup_obj)
        self._runner(self.run_obj)
        self._runner(self.teardown_obj)

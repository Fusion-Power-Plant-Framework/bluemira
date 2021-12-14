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

import string
import subprocess  # noqa (S404)
from enum import Enum
from typing import Dict

import bluemira.base as bm_base
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes.error import CodesError
from bluemira.codes.utilities import (
    LogPipe,
    get_recv_mapping,
    get_send_mapping,
    add_mapping,
)

__all__ = [
    "FileProgramInterface",
    "RunMode",
    "Setup",
    "Run",
    "Teardown",
]  # "ApplicationProgramInterface"]


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
        name = f"_{self.name.lower()}"
        if hasattr(obj, name):
            func = getattr(obj, name)
            return func(*args, **kwargs)


class Task:
    """
    Task integration

    Parameters
    ----------
    parent: obj
        parent file interface object

    """

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
    """
    Generic Setup Task

    Parameters
    ----------
    parent: obj
        parent file interface object
    params: ParameterFrame
        ParameterFrame for interface

    """

    def __init__(self, parent, *args, params=None, **kwargs):
        super().__init__(parent)
        self.set_parameters(params)

    def set_parameters(self, params):
        """
        Set parameter mappings and add parameters to interface

        Parameters
        ----------
        params: ParameterFrame

        """
        NAME = self.parent.NAME
        self._parameter_mapping = get_recv_mapping(params, NAME, recv_all=True)
        self._params = type(params).from_template(self._parameter_mapping.values())
        self._params.update_kw_parameters(params.to_dict(verbose=True))
        self._recv_mapping = get_recv_mapping(params, NAME)
        self._send_mapping = get_send_mapping(params, NAME)

    @property
    def params(self) -> bm_base.ParameterFrame:
        """
        The ParameterFrame corresponding to this run.
        """
        return self._params


class Run(Task):
    """
    Generic Run Task

    Parameters
    ----------
    parent: obj
        parent file interface object
    binary: str
        binary location
    """

    _binary = None

    def __init__(self, parent, binary=None, *args, **kwargs):
        super().__init__(parent)
        if binary is not None:
            self._binary = binary
        if self._binary is None:
            raise CodesError(f"Binary for {self.parent.NAME} not defined")


class Teardown(Task):
    """
    Generic Teardown Task

    Parameters
    ----------
    parent: obj
        parent file interface object
    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)

    def get_parameters(self):
        """
        TODO?
        """
        pass


class FileProgramInterface:
    """An external code wrapper"""

    _setup = Setup
    _run = Run
    _teardown = Teardown
    _runmode = RunMode

    def __init__(
        self, NAME, params, runmode, *args, run_dir=None, mappings=None, **kwargs
    ):
        self.NAME = NAME

        add_mapping(NAME, params, mappings)

        if not hasattr(self, "__run_dir"):
            self.__run_dir = "./" if run_dir is None else run_dir

        if self._runmode is not RunMode and issubclass(self._runmode, RunMode):
            self._set_runmode(runmode)
        else:
            raise CodesError("Please define a RunMode child lass")

        self._protect_tasks()

        self.setup_obj = self._setup(self, *args, params=params, **kwargs)
        self.run_obj = self._run(self, *args, **kwargs)
        self.teardown_obj = self._teardown(self, *args, **kwargs)

    def _protect_tasks(self):
        """
        If tasks are not the a child class use the defaults.
        """
        for sub_name, parent in [
            ["_setup", Setup],
            ["_run", Run],
            ["_teardown", Teardown],
        ]:
            if not issubclass(getattr(self, sub_name), parent):
                bluemira_warn("Using default {parent.__name__} task")
                setattr(self, sub_name, parent)

    @property
    def binary(self):
        """
        Program binary name
        """
        return self.run_obj._binary

    @binary.setter
    def binary(self, _binary: str):
        """
        Set program binary name

        Parameters
        ----------
        _binary: str
            binary name

        """
        self.run_obj._binary = _binary

    def _set_runmode(self, runmode: str):
        """
        Set the runmode

        Parameters
        ----------
        runmode: str
            runmode to be set
        """
        mode = runmode.upper().translate(str.maketrans("", "", string.whitespace))
        self._runner = self._runmode[mode]

    @property
    def _run_dir(self):
        """
        Run directory
        """
        return self.__run_dir

    @_run_dir.setter
    def _run_dir(self, directory: str):
        """
        Set run directory

        Parameters
        ----------
        directory: str
            new directory
        """
        self.__run_dir = directory
        self._set_property("_run_dir", directory)

    def _set_property(self, prop, val):
        for obj in ["setup", "run", "teardown"]:
            setattr(getattr(self, f"{obj}_obj"), prop, val)

    @property
    def params(self) -> bm_base.ParameterFrame:
        """
        The ParameterFrame corresponding to this run.
        """
        return self.setup_obj.params

    @property
    def _parameter_mapping(self) -> Dict[str, str]:
        """
        The ParameterFrame corresponding to this run.
        """
        return self.setup_obj._parameter_mapping

    @property
    def _recv_mapping(self) -> Dict[str, str]:
        """
        The ParameterFrame corresponding to this run.
        """
        return self.setup_obj._recv_mapping

    @property
    def _send_mapping(self) -> Dict[str, str]:
        """
        The ParameterFrame corresponding to this run.
        """
        return self.setup_obj._send_mapping

    def run(self, *args, **kwargs):
        """
        Run the full program interface
        """
        self._runner(self.setup_obj, *args, **kwargs)
        self._runner(self.run_obj, *args, **kwargs)
        self._runner(self.teardown_obj, *args, **kwargs)

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
Utility functions for interacting with external codes
"""


import os
import threading
from enum import Enum
from typing import Dict, Literal

from bluemira.base.look_and_feel import (
    _bluemira_clean_flush,
    bluemira_error_clean,
    bluemira_print,
    bluemira_print_clean,
)
from bluemira.base.parameter import ParameterFrame, ParameterMapping
from bluemira.codes.error import CodesError
from bluemira.utilities.tools import get_module


class Model(Enum):
    """
    Base Model Enum
    """

    @classmethod
    def info(cls):
        """
        Show Model options
        """
        infostr = f"{cls.__doc__}\n" + "\n".join(repr(l_) for l_ in list(cls))
        bluemira_print(infostr)


def get_code_interface(module):
    """
    Dynamically import code interface

    Parameters
    ----------
    module: str
        module to import

    Returns
    -------
    code module

    """
    try:
        return get_module(f"bluemira.codes.{module.lower()}")
    except ImportError:
        return get_module(module)


def _get_mapping(
    params, code_name: str, send_recv: Literal["send", "recv"], override: bool = False
) -> Dict[str, str]:
    """
    Create a dictionary to get the send or recieve mappings for a given code.

    Parameters
    ----------
    params: ParameterFrame
        The parameters with mappings that define what is going to be sent or recieved.
    code_name: str
        The identifying name of the code that data being send to or recieved from.
    send_recv: Literal["send", "recv"]
        Whether to generate a mapping for sending or reciving.
    override: bool, optional
        If True then map variables with a mapping defined, even if recv or send=False.
        By default, False.

    Yields
    ------
    mapping: Dict[str, str]
        The mapping between external code parameter names (key) and bluemira parameter
        names (value).
    """
    if send_recv not in ["send", "recv"]:
        raise CodesError("Mapping must be obtained for either send or recv.")

    mapping = {}
    for key in params.keys():
        param = params.get_param(key)
        if code_name in param.mapping and (
            override or getattr(param.mapping[code_name], send_recv)
        ):
            mapping[param.mapping[code_name].name] = key
    return mapping


def get_recv_mapping(params, code_name, recv_all=False):
    """
    Get the recieve mapping for variables mapped from the external code to the provided
    input ParameterFrame.

    Parameters
    ----------
    params: ParameterFrame
        The parameters with mappings that define what is going to be recieved.
    code_name: str
        The identifying name of the code that is being recieved from.
    recv_all: bool, optional
        If True then recieve all variables with a mapping defined, even if recv=False. By
        default, False.

    Returns
    -------
    mapping: Dict[str, str]
        The mapping between external code parameter names (key) and bluemira parameter
        names (value) to use for recieving.
    """
    return _get_mapping(params, code_name, "recv", recv_all)


def get_send_mapping(params, code_name, send_all=False):
    """
    Get the send mapping for variables mapped from the external code to the provided
    input ParameterFrame.

    Parameters
    ----------
    params: ParameterFrame
        The parameters with mappings that define what is going to be sent.
    code_name: str
        The identifying name of the code that is being sent to.
    send_all: bool, optional
        If True then send all variables with a mapping defined, even if send=False. By
        default, False.

    Returns
    -------
    mapping: Dict[str, str]
        The mapping between external code parameter names (key) and bluemira parameter
        names (value) to use for sending.
    """
    return _get_mapping(params, code_name, "send", send_all)


def add_mapping(
    code_name: str,
    params: ParameterFrame,
    mapping: Dict[str, ParameterMapping],
):
    """
    Adds mappings for a given code to a ParameterFrame.
    Modifies directly params but only if no mapping for that code exists

    Parameters
    ----------
    code_name: str
        Name of code
    params: ParameterFrame
        ParameterFrame to modify
    mapping: Dict[str, ParameterMapping]
        mapping between bluemira and the code

    """
    for key in params.keys():
        param = params.get_param(key)
        if param.var in mapping and code_name not in param.mapping:
            param.mapping[code_name] = mapping[param.var]


def create_mapping(
    in_mappings=None, out_mappings=None, io_mappings=None, none_mappings=None
):
    """
    Creates mappings for external codes

    Returns
    -------
    mappings: Dict
        A mapping from bluemira names to an external code ParameterMapping

    """
    mappings = {}
    ins = {"send": True, "recv": False}
    outs = {"send": False, "recv": True}
    inouts = {"send": True, "recv": True}
    nones = {"send": False, "recv": False}

    for puts, sr in [
        [in_mappings, ins],
        [out_mappings, outs],
        [io_mappings, inouts],
        [none_mappings, nones],
    ]:
        if puts is not None:
            for (
                bm_key,
                (ec_key, unit),
            ) in puts.items():
                mappings[bm_key] = ParameterMapping(
                    ec_key, send=sr["send"], recv=sr["recv"], unit=unit
                )

    return mappings


class LogPipe(threading.Thread):
    """
    Capture logs for subprocesses

    https://codereview.stackexchange.com/questions/6567/redirecting-subprocesses-output-stdout-and-stderr-to-the-logging-module

    Parameters
    ----------
    loglevel: str
        print or error flush printing

    """

    def __init__(self, loglevel):
        super().__init__(daemon=True)

        self.logfunc = {"print": bluemira_print_clean, "error": bluemira_error_clean}[
            loglevel
        ]
        self.logfunc_flush = _bluemira_clean_flush
        self.fd_read, self.fd_write = os.pipe()
        self.pipe = os.fdopen(self.fd_read, encoding="utf-8", errors="ignore")
        self.start()

    def fileno(self):
        """
        Return the write file descriptor of the pipe
        """
        return self.fd_write

    def run(self):
        """
        Run the thread and pipe it all into the logger.
        """
        for line in iter(self.pipe.readline, ""):
            if line.startswith("==>"):
                self.logfunc_flush(line.strip("\n"))
            else:
                self.logfunc(line)

        self.pipe.close()

    def close(self):
        """
        Close the write end of the pipe.
        """
        os.close(self.fd_write)

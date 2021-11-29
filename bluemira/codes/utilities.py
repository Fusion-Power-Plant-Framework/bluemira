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
import subprocess
import threading
from typing import Dict, Literal

from bluemira.base.look_and_feel import bluemira_error_clean, bluemira_print_clean
from bluemira.codes.error import CodesError


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
        has_mapping = param.mapping is not None and code_name in param.mapping
        map_param = has_mapping and (
            override or getattr(param.mapping[code_name], send_recv)
        )
        if map_param:
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


class LogPipe(threading.Thread):
    def __init__(self, loglevel):
        """
        Setup the object with a loglevel and start a thread for logging

        https://codereview.stackexchange.com/questions/6567/redirecting-subprocesses-output-stdout-and-stderr-to-the-logging-module

        Parameters
        ----------
        loglevel: str
            print or error flush printing

        """
        super().__init__(daemon=True)

        self.logfunc = {"print": bluemira_print_clean, "error": bluemira_error_clean}[
            loglevel
        ]
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
            self.logfunc(line)

        self.pipe.close()

    def close(self):
        """
        Close the write end of the pipe.
        """
        os.close(self.fd_write)

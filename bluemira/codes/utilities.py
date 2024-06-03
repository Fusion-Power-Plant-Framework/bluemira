# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Utility functions for interacting with external codes
"""

import json
import os
import subprocess  # noqa: S404
import threading
from collections.abc import Callable
from enum import Enum
from types import ModuleType
from typing import Any

from bluemira.base.look_and_feel import (
    _bluemira_clean_flush,
    bluemira_error_clean,
    bluemira_print,
    bluemira_print_clean,
)
from bluemira.codes.error import CodesError
from bluemira.codes.params import ParameterMapping
from bluemira.utilities.tools import get_module


class Model(Enum):
    """
    Base Model Enum
    """

    def __new__(cls, *args, **_kwds):
        """Overwrite default new to ignore extra arguments"""
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    @classmethod
    def info(cls):
        """
        Show Model options
        """
        infostr = f"{cls.__doc__}\n" + "\n".join(repr(l_) for l_ in list(cls))
        bluemira_print(infostr)

    @classmethod
    def _missing_(cls, value):
        try:
            return cls[value]
        except KeyError:
            raise ValueError(
                f"{cls.__name__} has no type {value}."
                f" Select from {(*cls._member_names_,)}"
            ) from None


def read_mock_json_or_raise(file_path: str, name: str) -> dict[str, float]:
    """
    Read json file or raise CodesError
    """
    try:
        with open(file_path) as f:
            return json.load(f)
    except OSError as os_error:
        raise CodesError(
            f"Cannot open mock {name} results file '{file_path}'."
        ) from os_error


def get_code_interface(module: str) -> ModuleType:
    """
    Dynamically import code interface

    Parameters
    ----------
    module:
        module to import

    Returns
    -------
    code module
    """
    try:
        return get_module(f"bluemira.codes.{module.lower()}")
    except ImportError:
        return get_module(module)


def create_mapping(
    in_mappings=None, out_mappings=None, io_mappings=None, none_mappings=None
) -> dict[str, Any]:
    """
    Creates mappings for external codes

    Returns
    -------
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
            for bm_key, (ec_key, unit) in puts.items():
                if isinstance(ec_key, tuple):
                    ec_in = ec_key[0]
                    ec_out = ec_key[1]
                else:
                    ec_in = ec_out = ec_key

                mappings[bm_key] = ParameterMapping(
                    ec_in, ec_out, send=sr["send"], recv=sr["recv"], unit=unit
                )

    return mappings


class LogPipe(threading.Thread):
    """
    Capture logs for subprocesses

    https://codereview.stackexchange.com/questions/6567/redirecting-subprocesses-output-stdout-and-stderr-to-the-logging-module

    Parameters
    ----------
    loglevel:
        print or error flush printing

    """

    def __init__(
        self,
        loglevel: str,
        flush_callable: Callable[[str], bool] = lambda line: False,  # noqa: ARG005
        flush_printer: Callable[[str], None] | None = None,
    ):
        super().__init__(daemon=True)

        self.logfunc = {"print": bluemira_print_clean, "error": bluemira_error_clean}[
            loglevel
        ]
        self.logfunc_flush = flush_printer or _bluemira_clean_flush
        self.flush_callable = flush_callable
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
            if self.flush_callable(line):
                self.logfunc_flush(line.strip("\n"))
            else:
                self.logfunc(line)

        self.pipe.close()

    def close(self):
        """
        Close the write end of the pipe.
        """
        os.close(self.fd_write)


def run_subprocess(
    command: list[str],
    run_directory: str = ".",
    flush_callable: Callable[[str], bool] = lambda line: False,  # noqa: ARG005
    flush_printer: Callable[[str], None] | None = None,
    **kwargs,
) -> int:
    """
    Run a subprocess terminal command piping the output into bluemira's
    logs.

    Parameters
    ----------
    command:
        The arguments of the command to run.
    run_directory:
        The directory to run the command in. Default is current working
        directory.
    **kwargs:
        Arguments passed directly to subprocess.Popen.

    Returns
    -------
    return_code: int
        The return code of the subprocess.
    """
    stdout = LogPipe("print", flush_callable, flush_printer=flush_printer)
    stderr = LogPipe("error", flush_callable)

    kwargs["cwd"] = run_directory
    kwargs.pop("shell", None)  # Protect against user input

    with subprocess.Popen(
        command,
        stdout=stdout,
        stderr=stderr,
        shell=False,  # noqa: S603
        **kwargs,
    ) as s:
        stdout.close()
        stderr.close()

    return s.returncode

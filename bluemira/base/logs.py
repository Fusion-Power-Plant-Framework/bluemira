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


"""Logging system setup and control."""

import logging
from enum import Enum
import sys

from bluemira.base.error import LogsError


class LogLevel(Enum):
    """Linking level names and corresponding numbers."""

    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0


def logger_setup(logfilename="bluemira_logging.log"):
    """
    Create logger with two handlers.

    Parameters
    ----------
    logfilename: str (default = bluemira_logging.log)
        Name of file to write logs to

    Returns
    -------
    logger: logging.RootLogger
        The logger to be used

    Notes
    -----
    set to debug initially
    """
    logger = logging.getLogger("")

    # what will be shown on screen

    on_screen_handler = logging.StreamHandler(stream=sys.stderr)
    on_screen_handler.setLevel(logging.DEBUG)

    # what will be written to a file

    recorded_handler = logging.FileHandler(logfilename)
    recorded_handler.setLevel(logging.DEBUG)

    logger.addHandler(on_screen_handler)
    logger.addHandler(recorded_handler)
    return logger


def set_log_level(verbose=1, increase=False):
    """
    Get new log level and check if it is possible.

    Parameters
    ----------
    verbose: str or int (default = 1)
        Amount the severity level of the logger should be changed by or to
    increase: bool (default = False)
        Whether level should be increased by specified amount or changed to it
    """
    # change loggers level
    for logger_name in [""]:
        logger = logging.getLogger(logger_name)

        current_level = logger.getEffectiveLevel() if increase else 0
        try:
            if isinstance(verbose, str):
                new_level = LogLevel[verbose]
            else:
                value = int(current_level + (verbose * 10))
                new_level = LogLevel(value)
        except ValueError:
            raise LogsError(f"Unknown severity level - {value}")
        except KeyError:
            raise LogsError(f"Unknown severity level - {verbose}")

        _modify_handler(new_level, logger)


def _modify_handler(new_level, logger):
    """
    Change level of the logger from user's input.

    Parameters
    ----------
    new_level: LogLevel
        Severity level for handler to be changed to, from set_log_level
    logger: logging.RootLogger
        Logger to be used
    """
    for handler in logger.handlers or logger.parent.handlers:
        if not isinstance(handler, logging.FileHandler):
            handler.setLevel(new_level.value)

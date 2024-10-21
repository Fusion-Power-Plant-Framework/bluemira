# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


"""Logging system setup and control."""

from __future__ import annotations

import logging
import sys
from enum import Enum
from textwrap import dedent, wrap
from types import DynamicClassAttribute
from typing import TYPE_CHECKING

from bluemira.base.constants import ANSI_COLOR, EXIT_COLOR
from bluemira.base.error import LogsError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class LogLevel(Enum):
    """Linking level names and corresponding numbers."""

    CRITICAL = (5, "darkred")
    ERROR = (4, "red")
    WARNING = (3, "orange")
    INFO = (2, "blue")
    DEBUG = (1, "green")
    NOTSET = (0, None)

    def __new__(cls, *args, **kwds):  # noqa: ARG003
        """Create Enum from first half of tuple"""
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _, colour: str | None = ""):
        self.colour = colour

    @classmethod
    def _missing_(cls, value: int | str) -> LogLevel:
        if isinstance(value, int):
            if cls.CRITICAL.value < value < 10:  # noqa: PLR2004
                return cls.CRITICAL
            value = max(value // 10 + value % 10, 0)
            if value <= cls.CRITICAL.value:
                return cls(value)
            return cls.CRITICAL
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            raise LogsError(
                f"Unknown severity level: {value}. Choose from: {(*cls._member_names_,)}"
            ) from None

    @DynamicClassAttribute
    def _value_for_logging(self) -> int:
        """Return builtin logging level value"""
        return int(self.value * 10)


class LoggerAdapter(logging.Logger):
    """Adapt the base logging class for our uses"""

    def _base(
        self,
        func: Callable[[str]],
        msg: str,
        *args,
        flush: bool = False,
        fmt: bool = True,
        _clean: bool = False,
        **kwargs,
    ):
        loglevel = LogLevel(func.__name__)
        return self._terminator_handler(
            func,
            colourise(
                msg,
                colour=None if _clean and loglevel is LogLevel.INFO else loglevel.colour,
                flush=flush,
                fmt=fmt,
            ),
            *args,
            fhterm=logging.StreamHandler.terminator if flush or not _clean else "",
            shterm="" if flush or _clean else logging.StreamHandler.terminator,
            **kwargs,
        )

    def debug(self, msg: str, *args, flush: bool = False, fmt: bool = True, **kwargs):
        """Debug"""
        return self._base(super().debug, msg, *args, flush=flush, fmt=fmt, **kwargs)

    def info(self, msg: str, *args, flush: bool = False, fmt: bool = True, **kwargs):
        """Info"""
        return self._base(super().info, msg, *args, flush=flush, fmt=fmt, **kwargs)

    def warning(self, msg: str, *args, flush: bool = False, fmt: bool = True, **kwargs):
        """Warning"""
        return self._base(
            super().warning, f"WARNING: {msg}", *args, flush=flush, fmt=fmt, **kwargs
        )

    def error(self, msg: str, *args, flush: bool = False, fmt: bool = True, **kwargs):
        """Error"""
        return self._base(
            super().error, f"ERROR: {msg}", *args, flush=flush, fmt=fmt, **kwargs
        )

    def critical(self, msg: str, *args, flush: bool = False, fmt: bool = True, **kwargs):
        """Critical"""
        return self._base(
            super().critical, f"CRITICAL: {msg}", *args, flush=flush, fmt=fmt, **kwargs
        )

    def clean(
        self,
        msg: str,
        loglevel: str | LogLevel = LogLevel.INFO,
        *args,
        flush: bool = False,
        **kwargs,
    ):
        """Unmodified logging"""
        func = getattr(super(), LogLevel(loglevel).name.lower())
        return self._base(
            func, msg, *args, flush=flush, fmt=False, _clean=True, **kwargs
        )

    @staticmethod
    def _terminator_handler(
        func: Callable[[str], None],
        string: str,
        *args,
        fhterm: str = "",
        shterm: str = "",
        **kwargs,
    ):
        """
        Log string allowing modification to handler terminator

        Parameters
        ----------
        func:
            The function to use for logging (e.g LOGGER.info)
        string:
            The string to colour flush print
        fhterm:
            FileHandler Terminator
        shterm:
            StreamHandler Terminator

        Notes
        -----
        This deals with some formatting issues when flushing or using external programs.
        Extra new line characters are added by default (this removes that behaviour):

            - When trying to flush text
            - When wrapped external printing

        For the file handler newlines are desired in all cases apart from when wrapping
        external programs
        For the stream handler newlines are only desired for normal logging

        """
        original_terminator = logging.StreamHandler.terminator
        logging.StreamHandler.terminator = shterm
        logging.FileHandler.terminator = fhterm
        try:
            func(string, *args, **kwargs)
        finally:
            logging.StreamHandler.terminator = original_terminator
            logging.FileHandler.terminator = original_terminator


def _bm_print(string: str, width: int = 73, *, single_flush: bool = False) -> str:
    """
    Create the text string for boxed text to print to the console.

    Parameters
    ----------
    string:
        The string of text to colour and box
    width:
        The width of the box, default = 73 (leave this alone for best results)

    Returns
    -------
    :
        The text string of the boxed text
    """
    if single_flush:
        return _bm_print_singleflush(string, width)

    strings = [
        " " if s == "\n" and i != 0 else s.removesuffix("\n")
        for i, s in enumerate(string.splitlines(keepends=True))
    ]
    t = [
        wrap(s, width=width - 4, replace_whitespace=False, drop_whitespace=False)
        for s in strings
    ]

    s = [dedent(item) for sublist in t for item in sublist]
    h = "".join(["+", "-" * width, "+"])
    lines = "\n".join([_bm_print_singleflush(i, width) for i in s])
    return f"{h}\n{lines}\n{h}"


def _bm_print_singleflush(string: str, width: int = 73) -> str:
    r"""
    Wrap the string in \| \|.

    Parameters
    ----------
    string:
        The string of text to colour and box
    width:
        The width of the box, default = 73 (leave this alone for best results)

    Returns
    -------
    :
        The wrapped text string
    """
    a = width - len(string) - 2
    return "| " + string + a * " " + " |"


def colourise(
    string: str,
    width: int = 73,
    colour: str | None = "blue",
    *,
    flush: bool = False,
    fmt: bool = True,
) -> str:
    """
    Print coloured, boxed text to the console. Default template for bluemira
    information.

    Parameters
    ----------
    string:
        The string of text to colour and box
    width:
        The width of the box, default = 73 (leave this alone for best results)
    colour:
        The colour to print the text in from `bluemira.base.constants.ANSI_COLOR`
    """
    text = _bm_print(string, width=width, single_flush=flush) if fmt else string

    return ("\r" if flush else "") + (
        text if colour is None else _print_colour(text, colour)
    )


def _print_colour(string: str, colour: str) -> str:
    """
    Create text to print. NOTE: Does not call print command

    Parameters
    ----------
    string:
        The text to colour
    colour:
        The colour to make the colour-string for

    Returns
    -------
    :
        The string with ANSI colour decoration
    """
    return f"{ANSI_COLOR[colour]}{string}{EXIT_COLOR}"


def logger_setup(
    logfilename: str = "bluemira.log", *, level: str | int = "INFO"
) -> logging.Logger:
    """
    Create logger with two handlers.

    Parameters
    ----------
    logfilename:
        Name of file to write logs to, default = bluemira.log
    level:
        The initial logging level to be printed to the console, default = INFO.

    Notes
    -----
    set to debug initially
    """
    logging.setLoggerClass(LoggerAdapter)
    root_logger = logging.getLogger("")
    bm_logger = logging.getLogger("bluemira")

    # what will be shown on screen
    on_screen_handler_out = logging.StreamHandler(stream=sys.stdout)
    on_screen_handler_out.setLevel(LogLevel(level)._value_for_logging)
    on_screen_handler_out.addFilter(lambda record: record.levelno < logging.WARNING)
    on_screen_handler_out.name = "BM stream stdout"

    on_screen_handler_err = logging.StreamHandler(stream=sys.stderr)
    on_screen_handler_err.setLevel(LogLevel(level)._value_for_logging)
    on_screen_handler_err.addFilter(lambda record: record.levelno >= logging.WARNING)
    on_screen_handler_err.name = "BM stream stderr"

    # what will be written to a file
    recorded_handler = logging.FileHandler(logfilename)
    recorded_formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    recorded_handler.setLevel(logging.DEBUG)
    recorded_handler.setFormatter(recorded_formatter)
    recorded_handler.name = "BM file out"

    bm_logger.setLevel(logging.DEBUG)

    root_logger.addHandler(on_screen_handler_out)
    root_logger.addHandler(on_screen_handler_err)
    root_logger.addHandler(recorded_handler)

    return bm_logger


def set_log_level(
    verbose: int | str = 1,
    *,
    increase: bool = False,
    logger_names: Iterable[str] = ("bluemira",),
):
    """
    Get new log level and check if it is possible.

    Parameters
    ----------
    verbose:
        Amount the severity level of the logger should be changed by or to
    increase:
        Whether level should be increased by specified amount or changed to it
    logger_names:
        The loggers for which to set the level, default = ("bluemira")
    """
    # change loggers level
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)

        current_level = logger.getEffectiveLevel() if increase else 0
        _modify_handler(
            LogLevel(verbose)
            if isinstance(verbose, str)
            else LogLevel(int(current_level + verbose)),
            logger,
        )


def get_log_level(logger_name: str = "bluemira", *, as_str: bool = True) -> str | int:
    """
    Return the current logging level.

    Parameters
    ----------
    logger_name
        The named logger to get the level for.
    as_str
        If True then return the logging level as a string, else as an int.
    """
    logger = logging.getLogger(logger_name)

    max_level = 0
    for handler in logger.handlers or logger.parent.handlers:
        if not isinstance(handler, logging.FileHandler) and handler.level > max_level:
            max_level = LogLevel(handler.level).value
    if as_str:
        return LogLevel(max_level).name
    return max_level


def _modify_handler(new_level: LogLevel, logger: logging.Logger):
    """
    Change level of the logger from user's input.

    Parameters
    ----------
    new_level:
        Severity level for handler to be changed to, from set_log_level
    logger:
        Logger to be used
    """
    for handler in logger.handlers or logger.parent.handlers:
        if not isinstance(handler, logging.FileHandler):
            handler.setLevel(new_level._value_for_logging)


class LoggingContext:
    """
    A context manager for temporarily adjusting the logging level

    Parameters
    ----------
    level:
        The bluemira logging level to set within the context.
    """

    def __init__(self, level: str | int):
        self.level = level
        self.original_level = get_log_level()

    def __enter__(self):
        """
        Set the logging level to the new level when we enter the context.
        """
        set_log_level(self.level)

    def __exit__(self, type, value, traceback):  # noqa: A002
        """
        Set the logging level to the original level when we exit the context.
        """
        set_log_level(self.original_level)

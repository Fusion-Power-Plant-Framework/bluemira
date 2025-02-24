# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


"""Logging system setup and control."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from threading import Event
from time import sleep
from types import DynamicClassAttribute
from typing import TYPE_CHECKING

import rich.jupyter as jp
from rich import default_styles
from rich.console import Console, ConsoleRenderable
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from bluemira.base.error import LogsError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from rich.traceback import Traceback

# TODO @je-cook: Remove on culmination of rich fix
# 3802
jp.JUPYTER_HTML_FORMAT = (
    '<pre style="white-space:pre;overflow-x:auto;line-height:normal;'
    "margin:0;"  # this is the change
    "font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New'"
    ',monospace">{code}</pre>'
)


class LogLevel(Enum):
    """Linking level names and corresponding numbers."""

    CRITICAL = (5, "darkred")
    ERROR = (4, "red")
    WARNING = (3, "orange")
    INFO = (2, "blue")
    DEBUG = (1, "green")
    NOTSET = (0, None)

    def __new__(cls, *args, **kwds):
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
    def value_for_logging(self) -> int:
        """Return builtin logging level value"""
        return int(self.value * 10)


def stop_progress(logger: LoggerAdapter, stop: Event, wait: float = 4):
    """
    Kill progress bar as soon as possible

    Parameters
    ----------
    logger:
        Logger instance
    stop:
        Kill signal event
    wait:
        time (s) to wait before killing progress bar
    """
    wait /= 0.1
    for _ in range(int(wait)):
        if stop.is_set():
            return
        sleep(0.1)
    logger.progress.stop()
    logger.progress = None


class LoggerAdapter(logging.Logger):
    """Adapt the base logging class for our uses"""

    progress: Progress | None = None
    _stop_p: Event = Event()

    def _base(
        self,
        func: Callable[[str], None],
        msg: str,
        *args,
        flush: bool = False,
        progress_timeout: float = 4,
        _clean: bool = False,
        **kwargs,
    ):
        self._flushing = flush
        self._clean = _clean
        msg = msg.strip()
        if (
            flush
            and get_log_level("bluemira", as_str=False) <= LogLevel(func.__name__).value
        ):
            self._stop_p.set()
            if self.progress is None:
                self.progress = Progress(
                    SpinnerColumn("simpleDots"), TextColumn("{task.description}")
                )
                self.progress.start()
                self.t1 = self.progress.add_task(description=msg)
            self.progress.update(self.t1, description=msg, visible=True)
            self._stop_p = Event()
            self.executor = ThreadPoolExecutor()
            self.executor.submit(stop_progress, self, self._stop_p, progress_timeout)
        elif self.progress is not None:
            self._stop_p.set()
            self.executor.shutdown(wait=False)
            self.progress.stop()
            self.progress = None

        func(msg, *args, stacklevel=kwargs.pop("stacklevel", 3), **kwargs)

    def makeRecord(self, *args, **kwargs) -> logging.LogRecord:  # noqa: N802
        """Overridden makeRecord to pass variables to handler"""  # noqa: DOC201
        record = super().makeRecord(*args, **kwargs)
        record._flushing = self._flushing
        record._clean = self._clean
        return record

    def debug(self, msg: str, *args, flush: bool = False, **kwargs):
        """Debug"""
        self._base(super().debug, msg, *args, flush=flush, **kwargs)

    def info(self, msg: str, *args, flush: bool = False, **kwargs):
        """Info"""
        self._base(super().info, msg, *args, flush=flush, **kwargs)

    def warning(self, msg: str, *args, flush: bool = False, **kwargs):
        """Warning"""
        self._base(super().warning, msg, *args, flush=flush, **kwargs)

    def error(self, msg: str, *args, flush: bool = False, **kwargs):
        """Error"""
        self._base(super().error, msg, *args, flush=flush, **kwargs)

    def critical(self, msg: str, *args, flush: bool = False, **kwargs):
        """Critical"""
        self._base(super().critical, msg, *args, flush=flush, **kwargs)

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
        self._base(func, msg, *args, flush=flush, _clean=True, **kwargs)


class BluemiraRichHandler(RichHandler):
    """
    Rich handler modified for different output types
    """

    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Traceback | None,
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        """Rich handler rendering in a panel under as requested

        Returns
        -------
        :
            The text to be rendered by the logger
        """
        log_renderable = super().render(
            record=record,
            traceback=traceback,
            message_renderable=message_renderable,
        )
        if getattr(record, "_flushing", False):
            self.console._flushing = True
            return log_renderable
        self.console._flushing = False
        if getattr(record, "_clean", True):
            return log_renderable
        return Panel(
            log_renderable,
            border_style=default_styles.DEFAULT_STYLES[
                f"logging.level.{record.levelname.lower()}"
            ],
        )


class BluemiraRichFileHandler(BluemiraRichHandler):
    """Allow some filtering on file log handlers"""


class ConsoleFlush(Console):
    """Rich console modified for progress bar use"""

    _flushing = False

    def print(self, *args, **kwargs):
        """Console output function customised for progress bar."""
        if self._flushing:
            return
        super().print(*args, **kwargs)


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

    Returns
    -------
    :
        The logger object.

    Notes
    -----
    set to debug initially.
    """
    logging.setLoggerClass(LoggerAdapter)
    root_logger = logging.getLogger("")
    bm_logger = logging.getLogger("bluemira")

    # what will be shown on screen
    on_screen_handler_out = BluemiraRichHandler(
        console=ConsoleFlush(), show_time=False, markup=True
    )
    on_screen_handler_out.setLevel(LogLevel(level).value_for_logging)
    on_screen_handler_out.addFilter(lambda record: record.levelno < logging.WARNING)
    on_screen_handler_out.name = "BM stream stdout"

    on_screen_handler_err = BluemiraRichHandler(
        console=ConsoleFlush(stderr=True), show_time=False, markup=True
    )
    on_screen_handler_err.setLevel(LogLevel(level).value_for_logging)
    on_screen_handler_err.addFilter(lambda record: record.levelno >= logging.WARNING)
    on_screen_handler_err.name = "BM stream stderr"

    # what will be written to a file
    # TODO @je-cook: force_jupyter and force_terminal shouldnt be needed
    #                but there is a bug in rich
    # 3803
    recorded_handler = BluemiraRichFileHandler(
        console=Console(
            file=open(logfilename, "a"),  # noqa: SIM115
            width=100,
            force_terminal=False,
            force_jupyter=False,
        )
    )
    recorded_handler.setLevel(logging.DEBUG)
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

    Returns
    -------
    :
        The logging level.
    """
    logger = logging.getLogger(logger_name)

    max_level = 0
    for handler in logger.handlers or logger.parent.handlers:
        if (
            not isinstance(handler, BluemiraRichFileHandler)
            and handler.level > max_level
        ):
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
        if not isinstance(handler, BluemiraRichFileHandler):
            handler.setLevel(new_level.value_for_logging)


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

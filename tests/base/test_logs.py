# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Testing for logging system."""

import logging

import pytest

from bluemira.base.error import LogsError
from bluemira.base.logs import (
    LogLevel,
    LoggingContext,
    get_log_level,
    logger_setup,
    set_log_level,
)


def _get_logger():
    """Avoid adding extra handlers"""
    handler_names = {"BM stream stdout", "BM stream stderr", "BM file out"}

    if (
        handler_names.intersection([h.name for h in logging.getLogger("").handlers])
        == handler_names
    ):
        return logging.getLogger("bluemira")
    return logger_setup()


class TestLoggingLevel:
    @classmethod
    def setup_class(cls):
        cls.orig_log = logging.getLogger("")
        cls.original_handlers = list(
            cls.orig_log.handlers or cls.orig_log.parent.handlers
        )
        cls.original_level = LogLevel(
            max(handler.level for handler in cls.original_handlers)
        )
        cls.LOGGER = _get_logger()
        set_log_level("NOTSET")

    @classmethod
    def teardown_class(cls):
        set_log_level(cls.original_level.name)

    @pytest.mark.parametrize("input_level", ["INF"])
    def test_raise_error(self, input_level):
        """Testing if errors for invalid log levels are caught."""
        with pytest.raises(LogsError) as exc_info:
            set_log_level(input_level)
        assert exc_info.type is LogsError

    @pytest.mark.parametrize(
        "input_level",
        [1, "DEBUG", 2, "INFO", 3, "WARNING", 4, "ERROR", 5, "CRITICAL"],
    )
    def test_not_error(self, input_level):
        """Testing if errors for invalid log levels are caught."""
        try:
            set_log_level(input_level)
        except LogsError:
            pytest.fail("Error raised")

    @pytest.mark.parametrize(
        ("input_level", "expected"),
        [
            ("DEBUG", 1),
            (1, 1),
            ("INFO", 2),
            (2, 2),
            ("WARNING", 3),
            (3, 3),
            ("ERROR", 4),
            (4, 4),
            ("CRITICAL", 5),
            (5, 5),
            (6, 5),
            (10, 1),
            (60, 5),
        ],
    )
    def test_level_change(self, input_level, expected):
        """Testing if the handlers level is actually changed."""
        set_log_level(input_level)
        for handler in self.LOGGER.handlers or self.LOGGER.parent.handlers:
            if not isinstance(handler, logging.FileHandler):
                assert handler.level // 10 == expected
        if not isinstance(input_level, str):
            if input_level >= 10:
                input_level //= 10
            if 5 < input_level < 10:
                input_level = 5
        assert get_log_level(as_str=isinstance(input_level, str)) == input_level


class TestLoggingContext:
    def test_logging_context(self):
        original_log_level = get_log_level(as_str=False)
        with LoggingContext(original_log_level + 1):
            assert original_log_level != get_log_level(as_str=False)
        assert get_log_level(as_str=False) == original_log_level


class TestLoggerClass:
    LOGGER = _get_logger()

    @classmethod
    def setup_class(cls):
        cls.orig_log = logging.getLogger("")
        cls.original_handlers = list(
            cls.orig_log.handlers or cls.orig_log.parent.handlers
        )
        cls.original_level = LogLevel(
            max(handler.level for handler in cls.original_handlers)
        )
        cls.LOGGER = logging.getLogger("bluemira")
        set_log_level("NOTSET")

    def teardown_method(self):
        set_log_level(self.original_level.name)

    @pytest.mark.parametrize(
        "logfunc",
        [LOGGER.debug, LOGGER.info, LOGGER.warning, LOGGER.error, LOGGER.critical],
    )
    @pytest.mark.parametrize("flush", [False, True])
    def test_basics(self, logfunc, flush, caplog):
        set_log_level("DEBUG")
        logfunc("string1", flush=flush)
        logfunc("string2", flush=flush)

        assert all("string" in c for c in caplog.messages)

    @pytest.mark.parametrize("loglevel", ["info", "error"])
    @pytest.mark.parametrize("flush", [False, True])
    def test_clean(self, flush, loglevel, caplog):
        self.LOGGER.clean("string1", flush=flush, loglevel=loglevel)
        self.LOGGER.clean("string2", flush=flush, loglevel=loglevel)
        assert any("string" in c for c in caplog.messages)

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


class TestLoggingLevel:
    @classmethod
    def setup_class(cls):
        # The logger needs to be reset after testing otherwise
        # extra handlers are added to the logger leading to double printing
        # if pytest is run with capturing switched off
        cls.orig_log = logging.getLogger("")
        cls.original_handlers = list(
            cls.orig_log.handlers or cls.orig_log.parent.handlers
        )
        cls.original_level = LogLevel(
            max(handler.level for handler in cls.original_handlers)
        )

    @classmethod
    def teardown_class(cls):
        for handler in cls.orig_log.handlers or cls.orig_log.parent.handlers:
            if handler not in cls.original_handlers:
                cls.orig_log.removeHandler(handler)
        set_log_level(cls.original_level.name)

    def setup_method(self):
        self.LOGGER = logger_setup()

    @pytest.mark.parametrize("input_level", [6, "INF"])
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
            ("DEBUG", 10),
            (1, 10),
            ("INFO", 20),
            (2, 20),
            ("WARNING", 30),
            (3, 30),
            ("ERROR", 40),
            (4, 40),
            ("CRITICAL", 50),
            (5, 50),
        ],
    )
    def test_level_change(self, input_level, expected):
        """Testing if the handlers level is actually changed."""
        set_log_level(input_level)
        for handler in self.LOGGER.handlers or self.LOGGER.parent.handlers:
            if not isinstance(handler, logging.FileHandler):
                assert handler.level == expected
        assert get_log_level(as_str=isinstance(input_level, str)) == input_level


class TestLoggingContext:
    def test_logging_context(self):
        original_log_level = get_log_level(as_str=False)
        with LoggingContext(original_log_level + 1):
            assert original_log_level != get_log_level(as_str=False)
        assert get_log_level(as_str=False) == original_log_level

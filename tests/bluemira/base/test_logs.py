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

"""Testing for logging system."""

import pytest
import logging

from bluemira.base.logs import (
    get_log_level,
    logger_setup,
    set_log_level,
    LogLevel,
    LoggingContext,
)
from bluemira.base.error import LogsError

LOGGER = logger_setup()


class TestLoggingLevel:
    def setup_method(self):
        self.original_level = LogLevel(
            max([handler.level for handler in LOGGER.handlers or LOGGER.parent.handlers])
        )

    def teardown_method(self):
        set_log_level(self.original_level.name)

    @pytest.mark.parametrize("input_level", [(6), ("INF")])
    def test_raise_error(self, input_level):
        """Testing if errors for invalid log levels are caught."""
        with pytest.raises(LogsError) as exc_info:
            set_log_level(input_level)
        assert exc_info.type is LogsError

    @pytest.mark.parametrize(
        "input_level",
        [
            (1),
            ("DEBUG"),
            (2),
            ("INFO"),
            (3),
            ("WARNING"),
            (4),
            ("ERROR"),
            (5),
            ("CRITICAL"),
        ],
    )
    def test_not_error(self, input_level):
        """Testing if errors for invalid log levels are caught."""
        try:
            set_log_level(input_level)
        except LogsError:
            pytest.fail("Error raised")

    @pytest.mark.parametrize(
        "input_level,expected",
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
        for handler in LOGGER.handlers or LOGGER.parent.handlers:
            if not isinstance(handler, logging.FileHandler):
                assert handler.level == expected
        assert get_log_level(as_str=isinstance(input_level, str)) == input_level


class TestLoggingContext:
    def test_logging_context(self):
        original_log_level = get_log_level(as_str=False)
        with LoggingContext(original_log_level + 1):
            assert original_log_level != get_log_level(as_str=False)
        assert get_log_level(as_str=False) == original_log_level

# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tests for the display auto_config module.

"""

import time
from functools import partial
from unittest.mock import patch

from bluemira.display.auto_config import get_primary_screen_size


def timeout(t=0.05):
    """
    The timeout in get_primary_screen_size is 3s
    """
    time.sleep(t)
    return True


class TestGetScreenSize:
    def setup_method(self):
        # clear lru_cache
        get_primary_screen_size.cache_clear()

    @patch(
        "bluemira.display.auto_config._get_primary_screen_size",
        new=partial(timeout, 0.1),
    )
    def test_timeout(self, caplog):
        out = get_primary_screen_size(timeout=0.01)
        assert out == (None, None)
        assert len(caplog.messages) == 1

    @patch(
        "bluemira.display.auto_config._get_primary_screen_size",
        new=partial(timeout, 0.01),
    )
    def test_no_timeout(self, caplog):
        out = get_primary_screen_size(1)
        assert out
        assert len(caplog.messages) == 0

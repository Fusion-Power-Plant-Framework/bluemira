# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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

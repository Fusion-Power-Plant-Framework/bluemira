# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import logging

import pytest

from bluemira.base.tools import _timing


def dummy(a, *, b=4):
    return (a, b)


@pytest.mark.parametrize(
    ("debug", "records"), [(False, ["INFO", "DEBUG"]), (True, ["DEBUG", "DEBUG"])]
)
def test_timing(debug, records, caplog):
    caplog.set_level(logging.DEBUG)
    assert _timing(dummy, "debug", "print", debug_info_str=debug)(1, b=2) == (1, 2)
    assert len(caplog.records) == 2
    assert [r.levelname for r in caplog.records] == records
    for msg, exp in zip(caplog.messages, ("print", "debug"), strict=False):
        assert exp in msg

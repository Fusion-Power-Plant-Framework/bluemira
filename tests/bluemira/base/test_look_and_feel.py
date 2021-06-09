# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

import pytest
import io
import sys
from bluemira.base.file import get_bluemira_root
from bluemira.base.constants import EXIT_COLOR, ANSI_COLOR
from bluemira.base.look_and_feel import (
    get_git_version,
    get_git_branch,
    count_slocs,
    user_banner,
    version_banner,
    bluemira_warn,
    bluemira_print,
)


ROOT = get_bluemira_root()


def test_get_git_version():
    assert isinstance(get_git_version(ROOT), bytes)


def test_get_git_branch():
    assert isinstance(get_git_branch(ROOT), str)


def test_count_slocs():
    branch = get_git_branch(ROOT)
    slocs = count_slocs(ROOT, branch)
    assert slocs[".py"] > 0
    assert slocs["total"] > 0
    total = sum(slocs.values()) - slocs["total"]
    assert total == slocs["total"]


def test_user_banner():
    assert len(user_banner()) == 2


def test_version_banner():
    assert len(version_banner()) == 3


def test_bluemira_warn():
    capture = io.StringIO()
    sys.stdout = capture
    bluemira_warn("bad")
    sys.stdout = sys.__stdout__

    result = capture.getvalue().splitlines()
    assert len(result) == 3
    assert ANSI_COLOR["red"] in result[0]
    assert "WARNING:" in result[1]
    assert "bad" in result[1]
    assert EXIT_COLOR in result[-1]

    capture = io.StringIO()
    sys.stdout = capture
    bluemira_warn(
        "test a very long and verbacious warning message that is bound to be boxed in over two lines."
    )
    sys.stdout = sys.__stdout__
    result = capture.getvalue().splitlines()

    assert len(result) == 4
    assert "WARNING:" in result[1]
    assert "test" in result[1]
    assert "WARNING:" not in result[2]
    assert "boxed" in result[2]
    assert EXIT_COLOR in result[-1]


def test_bluemira_print():
    capture = io.StringIO()
    sys.stdout = capture
    bluemira_print("good")
    sys.stdout = sys.__stdout__

    result = capture.getvalue().splitlines()
    assert len(result) == 3
    assert ANSI_COLOR["blue"] in result[0]
    assert "good" in result[1]
    assert EXIT_COLOR in result[-1]

    capture = io.StringIO()
    sys.stdout = capture
    bluemira_print(
        "test a very long and verbacious warning message that is bound to be boxed in over two lines."
    )
    sys.stdout = sys.__stdout__
    result = capture.getvalue().splitlines()

    assert len(result) == 4
    assert ANSI_COLOR["blue"] in result[0]
    assert "test" in result[1]
    assert "boxed" in result[2]
    assert EXIT_COLOR in result[-1]


if __name__ == "__main__":
    pytest.main([__file__])

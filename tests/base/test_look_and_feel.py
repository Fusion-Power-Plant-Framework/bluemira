# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
import subprocess  # noqa: S404
from unittest import mock

import pytest

from bluemira.base.constants import ANSI_COLOR, EXIT_COLOR
from bluemira.base.file import get_bluemira_root
from bluemira.base.logs import LoggingContext
from bluemira.base.look_and_feel import (
    bluemira_critical,
    bluemira_debug,
    bluemira_error,
    bluemira_print,
    bluemira_print_flush,
    bluemira_warn,
    count_slocs,
    get_git_branch,
    get_git_version,
    print_banner,
    user_banner,
    version_banner,
)

ROOT = get_bluemira_root()


GIT_WORKTREE = subprocess.run(
    ["git", "rev-parse", "--is-inside-work-tree"],  # noqa: S607
    shell=False,  # noqa: S603
    check=True,
)


@pytest.mark.skipif(
    GIT_WORKTREE.returncode != 0, reason="Not inside functioning git repository"
)
def test_get_git_version():
    assert isinstance(get_git_version(ROOT), bytes)


@pytest.mark.skipif(
    GIT_WORKTREE.returncode != 0, reason="Not inside functioning git repository"
)
def test_get_git_branch():
    assert isinstance(get_git_branch(ROOT), str)


@pytest.mark.skipif(
    GIT_WORKTREE.returncode != 0, reason="Not inside functioning git repository"
)
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


def capture_output(caplog, func, *inputs):
    """
    Print testing utility function.
    """
    if len(inputs) == 0:
        func()
    else:
        func(*inputs)

    # Flatten in the case where we have multiple messages and split by line
    result = [line for message in caplog.messages for line in message.split(os.linesep)]
    caplog.clear()
    return result


@pytest.mark.parametrize(
    ("method", "text", "colour", "default_text"),
    [
        (bluemira_critical, "boom", "darkred", "CRITICAL:"),
        (bluemira_error, "oops", "red", "ERROR:"),
        (bluemira_warn, "bad", "orange", "WARNING:"),
        (bluemira_print, "good", "blue", ""),
        (bluemira_debug, "check", "green", ""),
    ],
)
def test_bluemira_log(caplog, method, text, colour, default_text):
    # Make sure we capture in DEBUG regardless of default logging level
    # Otherwise we may miss values being recorded.
    with LoggingContext("DEBUG"):
        result = capture_output(caplog, method, text)

    assert len(result) == 3
    assert ANSI_COLOR[colour] in result[0]
    if len(default_text) > 0:
        assert default_text in result[1]
    assert text in result[1]
    assert EXIT_COLOR in result[-1]

    with LoggingContext("DEBUG"):
        result = capture_output(
            caplog,
            method,
            "test a very long and verbacious warning message that is bound to be boxed"
            " in over two lines.",
        )

    assert len(result) == 4
    if len(default_text) > 0:
        assert default_text in result[1]
        assert default_text not in result[2]
    assert "test" in result[1]
    assert "boxed" in result[2]
    assert EXIT_COLOR in result[-1]


def test_bluemira_print_flush(caplog):
    text = "First pass"
    result = capture_output(caplog, bluemira_print_flush, text)
    assert text in result[0]
    assert "\r" in result[0]
    assert os.linesep not in result[0]

    text = "Second pass"
    result = capture_output(caplog, bluemira_print_flush, text)
    assert text in result[0]
    assert "\r" in result[0]
    assert os.linesep not in result[0]


# Mock out the git branch name as long names can cause new lines which
# mess up this test's 'len' assertion
@mock.patch(  # noqa: PT008
    "bluemira.base.look_and_feel.get_git_branch", lambda _: "develop"
)
@mock.patch(  # noqa: PT008
    "bluemira.base.look_and_feel.count_slocs", lambda _, __: {"total": 1}
)
def test_print_banner(caplog):
    result = capture_output(caplog, print_banner)

    assert len(result) == 15
    assert ANSI_COLOR["blue"] in result[0]
    assert EXIT_COLOR in result[-1]

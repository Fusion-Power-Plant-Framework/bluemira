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
    _bluemira_clean_flush,
    bluemira_critical,
    bluemira_debug,
    bluemira_debug_flush,
    bluemira_error,
    bluemira_error_clean,
    bluemira_print,
    bluemira_print_clean,
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
    shell=False,
    check=False,
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
    result = [
        line
        for message in caplog.messages
        for line in message.split(os.linesep)
        if line != ""
    ]
    caplog.clear()
    return result


@pytest.mark.parametrize(
    ("method", "text", "colour", "default_text"),
    [
        (bluemira_critical, "boom", "darkred", "CRITICAL"),
        (bluemira_error, "oops", "red", "ERROR"),
        (bluemira_warn, "bad", "orange", "WARNING"),
        (bluemira_print, "good", "blue", "INFO"),
        (bluemira_debug, "check", "green", "DEBUG"),
    ],
)
def test_bluemira_log(caplog, capsys, method, text, colour, default_text):  # noqa: ARG001
    # Make sure we capture in DEBUG regardless of default logging level
    # Otherwise we may miss values being recorded.
    with LoggingContext("DEBUG"):
        result = capture_output(caplog, method, text)

    output = (
        capsys.readouterr().out
        if default_text in {"INFO", "DEBUG"}
        else capsys.readouterr().err
    )

    assert len(result) == 1
    assert default_text in output
    assert text in result[0]


@pytest.mark.parametrize(
    "func", [bluemira_print_flush, bluemira_debug_flush, _bluemira_clean_flush]
)
def test_bluemira_flush(func, caplog):
    caplog.set_level("DEBUG")
    text = "First pass"
    result = capture_output(caplog, func, text)
    assert text in result[0]
    assert os.linesep not in result[0]

    text = "Second pass"
    result = capture_output(caplog, func, text)
    assert text in result[0]
    assert os.linesep not in result[0]


@pytest.mark.parametrize("func", [bluemira_print_clean, bluemira_error_clean])
def test_bluemira_clean(func, caplog):
    text = "First pass"
    result = capture_output(caplog, func, text)
    assert text == result[0]
    assert os.linesep not in result[0]

    text = "Second pass"
    result = capture_output(caplog, func, text)
    assert text == result[0]
    assert os.linesep not in result[0]


# Mock out the git branch name as long names can cause new lines which
# mess up this test's 'len' assertion
@mock.patch(  # noqa: PT008
    "bluemira.base.look_and_feel.get_git_branch", lambda _: "develop"
)
@mock.patch(  # noqa: PT008
    "bluemira.base.look_and_feel.count_slocs", lambda _, __: {"total": 1}
)
def test_print_banner(caplog, capsys):
    _reset = capsys.readouterr().out
    result = capture_output(caplog, print_banner)
    output = capsys.readouterr().out
    print(output.split("\n"), flush=True)
    assert len(list(filter(bool, output.split("\n")))) in {16, 17}
    assert len(result) == 5
    assert ANSI_COLOR["blue"][:3] in output
    assert EXIT_COLOR in output

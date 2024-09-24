# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Aesthetic and ambiance functions.
"""

import os
import platform
import shutil
import subprocess  # noqa: S404
from getpass import getuser
from pathlib import Path

from bluemira import __version__
from bluemira.base.file import get_bluemira_path, get_bluemira_root
from bluemira.base.logs import logger_setup

LOGGER = logger_setup()

# Calculate the number of lines in this file
try:
    LOCAL_LINES = len(
        Path(get_bluemira_path("base"), "look_and_feel.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )
except FileNotFoundError:
    # Approximately
    LOCAL_LINES = 550

# =============================================================================
# Getters for miscellaneous information
# =============================================================================


def get_git_version(directory: str) -> str:
    """
    Get the version string of the current git branch, e.g.: '0.0.3-74-g70d48be'.

    Parameters
    ----------
    directory:
        The full path directory of the folder to get git information from

    Returns
    -------
    str
        The git version bytestring
    """
    return subprocess.check_output(  # noqa: S603
        ["git", "describe", "--tags", "--always"],  # noqa: S607
        cwd=directory,
    ).strip()


def get_git_branch(directory: str) -> str:
    """
    Get the name of the current git branch, e.g. 'develop'.

    Parameters
    ----------
    directory:
        The full path directory of the folder to get git information from

    Returns
    -------
    str
        The git branch string
    """
    return (
        subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
            cwd=directory,
        )
        .strip()
        .decode("utf-8")
    )


def get_git_files(directory: str, branch: str) -> list[str]:
    """
    Get the names of the files in the directory of the specified branch name.

    Parameters
    ----------
    directory:
        The full path directory of the folder to get git information from
    branch:
        The name of the git branch to retrieve the filenames from

    Returns
    -------
    list[str]
        The list of git-controlled path strings
    """
    return (
        subprocess.check_output(  # noqa: S603
            ["git", "ls-tree", "-r", branch, "--name-only"],  # noqa: S607
            cwd=directory,
        )
        .decode("utf-8")
        .splitlines()
    )


def get_platform() -> str:
    """
    Get the OS platform.

    Returns
    -------
    str
        The generic name of the platform (e.g. Linux, Windows)
    """
    return platform.uname()[0]


def count_slocs(
    directory: str,
    branch: str,
    exts: list[str] | None = None,
    ignore: list[str] | None = None,
) -> dict[str, int | list[int]]:
    """
    Counts lines of code within a given directory for a given git branch

    Parameters
    ----------
    directory:
        The full path directory of the folder to get git information from
    branch:
        The git branch string
    exts:
        The list of file extensions to search the directory for
    ignore:
        The list of extensions and filenames to ignore

    Returns
    -------
    dict[str, int | list[int]]
        The dictionary of number of lines of code per file extension, and the
        total linecount
    """
    if ignore is None:
        ignore = [".git", ".txt", "look_and_feel.py"]

    if exts is None:
        exts = [".py"]

    lines = {}
    for k in exts:
        lines[k] = 0
    files = get_git_files(directory, branch)
    for name in files:
        if Path(name).parts[-1] not in ignore and name not in ignore:
            for e in exts:
                if name.endswith(e):
                    path = Path(directory, name)
                    try:
                        lines[e] += len(
                            Path(path).read_text(encoding="utf-8").splitlines()
                        )

                    except FileNotFoundError:
                        bluemira_warn(
                            "count_slocs: Probably not on the right git branch"
                        )
                        continue

    lines[".py"] += LOCAL_LINES
    lines["total"] = sum(lines[k] for k in lines)
    return lines


# =============================================================================
# Printing functions
# =============================================================================


def bluemira_critical(string: str):
    """
    Standard template for bluemira critical errors.
    """
    return LOGGER.critical(string)


def bluemira_error(string: str):
    """
    Standard template for bluemira errors.
    """
    return LOGGER.error(string)


def bluemira_warn(string: str):
    """
    Standard template for bluemira warnings.
    """
    return LOGGER.warning(string)


def bluemira_print(string: str):
    """
    Standard template for bluemira information messages.
    """
    return LOGGER.info(string)


def bluemira_debug(string: str):
    """
    Standard template for bluemira debugging.
    """
    return LOGGER.debug(string)


def _bluemira_clean_flush(
    string,
):
    """
    Print and flush string. Useful for updating information.

    Parameters
    ----------
    string:
        The string to colour flush print
    """
    LOGGER.clean_flush(string)


def bluemira_print_flush(string: str):
    """
    Print a coloured, boxed line to the console and flushes it. Useful for
    updating information.

    Parameters
    ----------
    string:
        The string to colour flush print
    """
    LOGGER.info_flush(string)


def bluemira_debug_flush(string: str):
    """
    Print a coloured, boxed line to the console and flushes it. Useful for
    updating information when running at the debug logging level.

    Parameters
    ----------
    string:
        The string to colour flush print for debug messages.
    """
    LOGGER.debug_flush(string)


def bluemira_print_clean(string: str):
    """
    Print to the logging info console with no modification.
    Useful for external programs

    Parameters
    ----------
    string:
        The string to print
    """
    LOGGER.info_clean(string)


def bluemira_error_clean(string: str):
    """
    Print to the logging error console, colouring the output red.
    No other modification is made. Useful for external programs

    Parameters
    ----------
    string:
        The string to colour print
    """
    LOGGER.error_clean(string)


# =============================================================================
# Banner printing
# =============================================================================


BLUEMIRA_ASCII = r"""+-------------------------------------------------------------------------+
|  _     _                      _                                         |
| | |   | |                    (_)                                        |
| | |__ | |_   _  ___ _ __ ___  _ _ __ __ _ __                            |
| | '_ \| | | | |/ _ \ '_ ` _ \| | '__/ _| |_ \                           |
| | |_) | | |_| |  __/ | | | | | | | | (_| |_) |                          |
| |_.__/|_|\__,_|\___|_| |_| |_|_|_|  \__|_|__/                           |
+-------------------------------------------------------------------------+"""  # noqa: E501


def print_banner():
    """
    Print the initial banner to the console upon running the bluemira code.
    """
    LOGGER.info(BLUEMIRA_ASCII, fmt=False)
    v = version_banner()
    v.extend(user_banner())
    bluemira_print("\n".join(v))


def version_banner() -> list[str]:
    """
    Get the string for the version banner.

    Returns
    -------
    list[str]
        The list of strings of text describing the version and code information
    """
    mapping = {
        "SLOC": "total",
    }
    root = get_bluemira_root()
    if not Path(f"{root}/.git").is_dir() or shutil.which("git") is None:
        return [
            f"Version    : {__version__}",
            "git branch : docker",
            "SLOC      : N/A",
        ]
    branch = get_git_branch(root)
    sloc = count_slocs(get_bluemira_path().rstrip(os.sep), branch)
    v = str(get_git_version(root))

    output = [f"Version    : {v[2:-1]}", f"git branch : {branch}"]

    for k, v in mapping.items():
        if sloc[v] > 0:
            line = k + " " * (11 - len(k)) + f": {int(sloc[v])}"
            output.append(line)
    return output


def user_banner() -> list[str]:
    """
    Get user and platform info and create text to print to banner.

    Returns
    -------
    list[str]
        The text for the banner containing user and platform information
    """
    return [
        f"User       : {getuser()}",
        f"Platform   : {get_platform()}",
    ]

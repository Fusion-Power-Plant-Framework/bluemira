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

"""
Mostly just aesthetic and ambiance functions
"""
import os
import sys
import time
import datetime
from textwrap import wrap
from itertools import cycle
import numpy as np
import subprocess  # noqa (S404)
import seaborn as sns
from matplotlib.colors import hex2color

from BLUEPRINT.base.file import get_BP_path
from bluemira.base.look_and_feel import (
    bluemira_warn,
    bluemira_print,
    get_git_files,
    user_banner,
    version_banner,
    _print_color,
    _bprintsingleflush,
)

KEY_TO_PLOT = False
# Calculate the number of lines in this file
try:
    with open(
        os.sep.join([get_BP_path("base"), "lookandfeel.py"]), "r", encoding="utf-8"
    ) as f:
        LOCAL_LINES = len(f.read().splitlines())
except FileNotFoundError:
    # Approximately
    LOCAL_LINES = 550


def color_kwargs(**kwargs):
    """
    Handle matplotlib color keyword arguments.

    Parameters
    ----------
    kwargs

    Returns
    -------
    colors: cycle
        The cycle of colors to use in plotting
    """
    if "color" in kwargs:
        if len(np.shape(kwargs["color"])) == 1:
            if type(kwargs["color"][0]) is str:
                colors = [hex2color(kwargs["color"][0])]
            else:
                colors = [kwargs["color"]]
        else:
            colors = kwargs["color"]
    elif "palette" and "n" in kwargs:
        p = sns.color_palette(kwargs["palette"], kwargs["n"])
        colors = [p[i] for i in range(kwargs["n"])]
    elif "color" not in kwargs:
        colors = ["grey"]
    colors = cycle(colors)
    return colors


def banner():
    """
    Prints the initial banner to the console upon running the BLUEPRINT code
    """
    print(bannerprint(_banner2()))
    v = version_banner()
    v.append(user_banner())
    bluemira_print("\n".join(v))


def _banner2():
    """
    The text for the ASCII art banner
    """
    return """
██████╗ ██╗     ██╗   ██╗███████╗██████╗ ██████╗ ██╗███╗   ██╗████████╗
██╔══██╗██║     ██║   ██║██╔════╝██╔══██╗██╔══██╗██║████╗  ██║╚══██╔══╝
██████╔╝██║     ██║   ██║█████╗  ██████╔╝██████╔╝██║██╔██╗ ██║   ██║
██╔══██╗██║     ██║   ██║██╔══╝  ██╔═══╝ ██╔══██╗██║██║╚██╗██║   ██║
██████╔╝███████╗╚██████╔╝███████╗██║     ██║  ██║██║██║ ╚████║   ██║
╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝
"""


def bannerprint(s, width=73):
    """
    Creates the box around the ASCII art

    Parameters
    ----------
    s: str
        The ASCII art text string
    width: int (default = 73)
        The width of the box (leave this alone for best results)

    Returns
    -------
    ss: str
        The text string of the boxed ASCII art
    """
    t = wrap(s, width=width, replace_whitespace=False, drop_whitespace=False)
    s = [i.split("\n")[-2] for i in t[:-1]]
    lines = ["".join(["| "] + [i] + [" "] * (width - 1 - len(i)) + ["|"]) for i in s]
    h = "".join(["+"] + ["-" * width] + ["+"])
    text = h + "\n" + "\n".join(lines) + "\n" + h
    return _print_color(text, "blue")


def count_slocs(
    directory,
    branch,
    exts=None,
    ignore=None,
):
    """
    Counts lines of code within a given directory for a given git branch

    Parameters
    ----------
    directory: str
        The full path directory of the folder to get git information from
    branch: str
        The git branch string
    exts: list
        The list of file extensions to search the directory for
    ignore: list
        The list of extensions and filenames to ignore

    Returns
    -------
    lines: dict
        The dictionary of number of lines of code per file extension, and the
        total linecount
    """
    if ignore is None:
        ignore = [".git", ".txt", "lookandfeel.py"]

    if exts is None:
        exts = [".py"]

    lines = {}
    for k in exts:
        lines[k] = 0
    files = get_git_files(directory, branch)
    for name in files:
        name = name.decode("utf-8")
        if name.split(os.sep)[-1] not in ignore and name not in ignore:
            for e in exts:
                if name.endswith(e):
                    path = os.sep.join([directory, name])
                    try:
                        with open(path, "r", encoding="utf-8") as file:
                            lines[e] += len(file.read().splitlines())

                    except FileNotFoundError:
                        bluemira_warn(
                            "count_slocs: Probably not on the right git branch"
                        )
                        continue

    lines[".py"] += LOCAL_LINES
    lines["total"] = sum([lines[k] for k in lines.keys()])
    return lines


def bpinfo(string):
    """
    Deprecated standard template for BLUEPRINT information
    """
    return bluemira_print("INFO: " + string)


class BClock:
    """
    Pinched from S. McIntosh ;)
    """

    def __init__(self, n_iter, print_rate=1, width=73):

        # Constructors
        self.i = 0
        self.to = None
        self.n_iter = None

        self.start(n_iter)
        self.rate = print_rate
        self.A = " elapsed"
        self.B = " left"
        self.width = width - len(self.A) - len(self.B) - n_iter * 2 - 4 - 2 * 9

    def start(self, n_iter):
        """
        Start the clock
        """
        self.i = 0
        self.to = time.time()
        self.n_iter = n_iter

    def stop(self):
        """
        Stop the clock
        """
        return time.time() - self.to

    def tock(self):
        """
        Tick the iterations of the clock over
        """
        self.i += 1
        if self.i % self.rate == 0 and self.i > 0:
            elapsed = time.time() - self.to
            remain = int((self.n_iter - self.i) / self.i * elapsed)
            prog_str = "| \r{:1d}/{:1d}".format(self.i, self.n_iter)
            prog_str += " {:0>8}s elapsed".format(
                str(datetime.timedelta(seconds=int(elapsed)))
            )
            prog_str += " {:0>8}s left".format(str(datetime.timedelta(seconds=remain)))
            prog_str += " {:1.1f}%".format(1e2 * self.i / self.n_iter)
            nh = int(self.i / self.n_iter * self.width)
            prog_str += " |" + nh * "#" + (self.width - nh) * "-" + "|"
            # s = bprint(prog_str, end='\r', flush=True)

            sys.stdout.write(
                "\r" + _bprintsingleflush(f"{self.i}/{self.n_iter} {prog_str}")
            )
            sys.stdout.flush()

        if self.i == self.n_iter:
            print("\n")


if __name__ == "__main__":
    from BLUEPRINT import test

    test()

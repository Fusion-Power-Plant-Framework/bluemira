# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

"""
Mostly just aesthetic and ambiance functions
"""
import os
import sys
import time
import datetime
import platform
import functools
from PyQt5 import QtWidgets
from textwrap import wrap, dedent
from getpass import getuser
from itertools import cycle
import numpy as np
import subprocess  # noqa (S404)
import seaborn as sns
from matplotlib.colors import hex2color
import shutil

from BLUEPRINT import __version__
from BLUEPRINT.base.palettes import B_PALETTE
from BLUEPRINT.base.file import get_BP_root, get_BP_path
from BLUEPRINT.base.constants import EXIT_COLOR, ANSI_COLOR

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
    bprint("\n".join(v))


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


def get_git_version(directory):
    """
    Gets the version string of the current git branch, e.g.:
        '0.0.3-74-g70d48be'

    Parameters
    ----------
    directory: str
        The full path directory of the folder to get git information from

    Returns
    -------
    vinfo: str
        The git version string
    """
    return subprocess.check_output(  # noqa (S603, S607)
        ["git", "describe", "--tags", "--always"], cwd=directory
    ).strip()


def get_git_branch(directory):
    """
    Gets the name of the current git branch, e.g. 'develop'

    Parameters
    ----------
    directory: str
        The full path directory of the folder to get git information from

    Returns
    -------
    branch: str
        The git branch string
    """
    out = subprocess.check_output(  # noqa (S603, S607)
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=directory
    )
    return out.strip().decode("utf-8")


def get_git_files(directory, branch):
    """
    Gets the names of the files in the directory of the specified branch name

    Parameters
    ----------
    directory: str
        The full path directory of the folder to get git information from
    branch: str
        The name of the git branch to retrieve the filenames from

    Returns
    -------
    branch: str
        The git branch string
    """
    return subprocess.check_output(  # noqa (S603, S607)
        ["git", "ls-tree", "-r", branch, "--name-only"], cwd=directory
    ).splitlines()


def get_platform():
    """
    Gets the OS platform

    Returns
    -------
    platform: str
        The generic name of the platform (e.g. Linux, Windows)
    """
    return platform.uname()[0]


def user_banner():
    """
    Gets user and platform info and creates text to print to banner

    Returns
    -------
    s: str
        The text for the banner containing user and plaform information
    """
    p = get_platform()
    u = getuser()
    return f"""
\nUser       : {u}
Platform   : {p}
"""


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
                        bpwarn("count_slocs: Probably not on the right git branch")
                        continue

    lines[".py"] += LOCAL_LINES
    lines["total"] = sum([lines[k] for k in lines.keys()])
    return lines


def version_banner():
    """
    Gets the string for the version banner

    Returns
    -------
    s: str
        The string of text describing the version and code information
    """
    mapping = {
        "SLOC": "total",
    }
    root = get_BP_root()
    if not os.path.isdir(f"{root}/.git") or shutil.which("git") is None:
        return [
            f"Version    : {__version__}",
            "git branch : docker",
            "SLOC      : N/A",
        ]
    branch = get_git_branch(root)
    sloc = count_slocs(get_BP_path().rstrip(os.sep), branch)
    v = str(get_git_version(root))

    output = [f"Version    : {v[2:-1]}", f"git branch : {branch}"]

    for k, v in mapping.items():
        if sloc[v] > 0:
            line = k + " " * (11 - len(k)) + f": {int(sloc[v])}"
            output.append(line)
    return output


def _print_color(string, color):
    """
    Creates text to print. NOTE: Does not call print command

    Parameters
    ----------
    string: str
        The text to colour
    color: str
        The color to make the color-string for

    Returns
    -------
    color_text: str
        The string with ANSI color decoration
    """
    return f"{ANSI_COLOR[color]}{string}{EXIT_COLOR}"


def bpwarn(string):
    """
    Standard template for BLUEPRINT warnings
    """
    return bprint("WARNING: " + string, color="red")


def bpinfo(string):
    """
    Deprecated standard template for BLUEPRINT information
    """
    return bprint("INFO: " + string)


def _bprint(string, width=73):
    """
    Creates the text string for boxed text to print to the console.

    Parameters
    ----------
    string: str
        The string of text to colour and box
    width: int (default = 73)
        The width of the box (leave this alone for best results)

    Returns
    -------
    ss: str
        The text string of the boxed text
    """
    strings = string.splitlines()
    bw = width - 4
    t = [
        wrap(s, width=bw, replace_whitespace=False, drop_whitespace=False)
        for s in strings
    ]

    s = [dedent(item) for sublist in t for item in sublist]
    lines = ["".join(["| "] + [i] + [" "] * (width - 2 - len(i)) + [" |"]) for i in s]
    h = "".join(["+"] + ["-" * width] + ["+"])
    return h + "\n" + "\n".join(lines) + "\n" + h


def bprint(string, width=73, color="blue", end=None, flush=False):
    """
    Prints coloured, boxed text to the console. Default template for
    BLUEPRINT information

    Parameters
    ----------
    string: str
        The string of text to colour and box
    width: int (default = 73)
        The width of the box (leave this alone for best results)
    color: str from ['blue', 'red', 'green']
        The color to print the text in
    end: str or None (default = None)
        The value to print after the print operation
    flush: bool (default=False)
        As far as I can tell has no effect
    """
    text = _bprint(string, width=width)
    color_text = _print_color(text, color)
    print(color_text, end=end, flush=flush)


def _bprintsingleflush(string, width=73, color="blue"):
    """
    Creates the text string for coloured, boxed text to flush print to the
    console.

    Parameters
    ----------
    string: str
        The string of text to colour and box
    width: int (default = 73)
        The width of the box (leave this alone for best results)
    color: str from ['blue', 'red', 'green']
        The color to print the text in

    Returns
    -------
    ss: str
        The text string of the boxed coloured text to flush print
    """
    a = width - len(string) - 2
    text = "| " + string + a * " " + " |"
    return _print_color(text, color)


def bprintflush(string):
    """
    Prints a coloured, boxed line to the console and flushes it. Useful for
    updating information

    Parameters
    ----------
    string: str
        The string to colour flush print
    """
    sys.stdout.write("\r" + _bprintsingleflush(string))
    sys.stdout.flush()


@functools.lru_cache(1)
def get_primary_screen_size():
    """
    Get the size in pixels of the primary screen.

    Used for sizing figures to the screen for small screens.

    Returns
    -------
    width: Union[int, None]
        width of the primary screen in pixels. If there is no screen returns None
    height: Union[int, None]
        height of the primary screen in pixels. If there is no screen returns None
    """
    if sys.platform.startswith("linux") and os.getenv("DISPLAY") is None:
        return None, None

    # IPython detection (of sorts)
    app = QtWidgets.QApplication.instance()
    if app is None:
        # if IPython isn't open then a QApplication is created to get screen size
        with QtWidgets.QApplication([]) as app:
            rect = app.primaryScreen().availableGeometry()
    else:
        rect = app.primaryScreen().availableGeometry()

    return rect.width(), rect.height()


def get_figure_scale_factor(figsize):
    """
    Scale figure size to fit on small screens.

    If the screen fits the figure the scale factor is 1.

    Parameters
    ----------
    figsize: np.array(float, float)
        matplotlib figsize width x height

    Returns
    -------
    sf: float
        scale factor to fit screen

    """
    screen_size = get_primary_screen_size()

    if None in screen_size:
        return 1

    dpi = sns.mpl.rcParams["figure.dpi"]

    dpi_size = figsize * dpi
    dpi_size += 0.10 * dpi_size  # space for toolbar

    sf = 1  # scale factor
    for ds, ss in zip(dpi_size, screen_size):
        if ds > ss:
            scale_temp = ss / ds
            if scale_temp < sf:
                sf = scale_temp
    return sf


def plot_defaults(force=False):
    """
    Sets a series of plotting defaults based on machine and user.

    If BLUEPRINT plots are not to your tastes, do not work with your OS, or
    don't fit your screen, please create a user profile for yourself/machine
    here and adjust settings as needed.

    Parameters
    ----------
    force:bool
        force default figsize irrespective of screen size

    """
    if getuser() == "matti" and sys.platform == "win32":

        sns.set(
            context="paper",
            style="white",
            font="Calibri",
            font_scale=2.5,
            color_codes=False,
            rc={
                "axes.labelweight": "bold",
                "axes.titleweight": "bold",
                "axes.titlesize": 20,
                "figure.figsize": [18, 15],
                "lines.linewidth": 4,
                "lines.markersize": 10,
                "contour.negative_linestyle": "solid",
            },
        )
    else:
        figsize = np.array([18, 15])

        sf = 1 if force else get_figure_scale_factor(figsize)

        sns.set(
            context="paper",
            style="ticks",
            font="DejaVu Sans",
            font_scale=2.5 * sf,
            color_codes=False,
            rc={
                "axes.labelweight": "normal",
                # 'axes.titleweight': 'bold',
                "axes.titlesize": 20 * sf,
                "figure.figsize": list(figsize * sf),
                "lines.linewidth": 4 * sf,
                "lines.markersize": 13 * sf,
                "contour.negative_linestyle": "solid",
            },
        )
        sns.set_style(
            {
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.major.size": 8 * sf,
                "ytick.major.size": 8 * sf,
                "xtick.minor.size": 4 * sf,
                "ytick.minor.size": 4 * sf,
                "xtick.color": "k",
            }
        )
    sns.set_palette(B_PALETTE)


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

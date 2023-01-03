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
Convert all example .py files specified or in the underlying directory to .ipynb.
"""


import json
import os
import platform
from argparse import ArgumentParser
from pathlib import Path
from sys import exit

header_comment = "# %%"


def py2nb(py_str):
    """
    Convert a .py file to .ipynb

    Parameters
    ----------
    py_str : str
        Contents of the Python file.

    Returns
    -------
    ipynb : str
        Contents of the Notebook file.
    """
    # remove leading header comment
    if py_str.startswith(header_comment):
        py_str = py_str[len(header_comment) :]

    cells = []
    chunks = py_str.split(f"\n\n{header_comment}")

    for chunk in chunks[1:]:
        chunk = chunk.strip()
        cell_type = "code"
        if chunk.startswith("[markdown]\n"):
            chunk = chunk.replace("[markdown]\n", "", 1)
            cell_type = "markdown"
            chunk = "\n".join(
                [
                    "" if line == "#" else line.replace("# ", "", 1)
                    for line in chunk.split("\n")
                ]
            )

        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": chunk.splitlines(True),
        }

        if cell_type == "code":
            cell.update({"outputs": [], "execution_count": None})

        cells.append(cell)

    nb_str = {
        "cells": cells,
        "metadata": {
            "anaconda-cloud": {},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": platform.python_version(),
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    return nb_str


def equal(orig, new):
    """
    Check ipynb strings are equal.
    Ignore version string as long as using python 3
    """
    orig_check = orig.splitlines(keepends=True)
    new_check = new.splitlines(keepends=True)

    version_str = '"version": "3'

    if orig == "":
        return False
    for old, new in zip(orig_check, new_check):
        if old != new:
            if not (version_str in old and version_str in new):
                return False

    return True


def convert(path, check, modify):
    """
    Convert file to ipynb.

    Parameters
    ----------
    path: str
        path to file
    check: bool
        check existing files or overwrite all
    modify: bool
        modify files

    """
    with open(path, "r") as py_fh:
        py_str = py_fh.read()

    if header_comment in py_str:
        nb_str = py2nb(py_str)
        name, _ = os.path.splitext(path)
        ipynb = name + ".ipynb"

        if check and os.path.isfile(ipynb):
            with open(ipynb, "r") as orig_nb:
                orig_nb_json = orig_nb.read()
        else:
            orig_nb_json = ""

        nb_json = json.dumps(nb_str, indent=2) + "\n"

        if not equal(orig_nb_json, nb_json):
            if not modify:
                return ipynb + " NEEDS UPDATE"

            with open(name + ".ipynb", "w") as nb_fh:
                nb_fh.write(nb_json)
            return ipynb + " UPDATED"
    else:
        return str(path) + " No markdown comments"


def arguments():
    """
    Parse arguments for conversion script
    """
    parser = ArgumentParser(
        description=__doc__
        + "Running this script with no file arguments will convert every .py file in the"
        " current directory and all subdirectories to an .ipynb."
    )
    parser.add_argument(
        "--check", action="store_true", default=False, help="precommit difference check"
    )
    parser.add_argument(
        "--no-modify", action="store_true", default=False, help="don't make changes"
    )
    parser.add_argument(
        "files",
        metavar="files",
        type=str,
        default=[],
        nargs="*",
        help="python files to be converted",
    )

    args = parser.parse_args()
    if args.files == [] and not args.check:
        files = list(Path(__file__).parent.rglob("*.py"))
        try:
            files.pop(files.index(Path(__file__)))
        except ValueError:
            pass
    else:
        files = args.files
        if not files:
            raise ValueError("No files specified")

    return files, args.check, not args.no_modify


if __name__ == "__main__":
    files, check, modify = arguments()
    updated = []
    for file in files:
        update = convert(file, check, modify)
        if update is not None:
            updated.append(update)

    if updated != []:
        print("\n".join(updated))
        if modify:
            print("\nchanges need to be commited")
        exit(1)

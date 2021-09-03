# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Convert all .py files in the underlying directory or specified files to .ipynb.

"""

import glob
import json
import platform
import os
from sys import argv

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


def convert(path):
    """
    Convert file to ipynb.

    Parameters
    ----------
    path: str
        path to file

    """
    with open(path, "r") as py_fh:
        py_str = py_fh.read()
        if header_comment in py_str:
            nb_str = py2nb(py_str)
            name, _ = os.path.splitext(path)
        with open(name + ".ipynb", "w") as nb_fh:
            json.dump(nb_str, nb_fh, indent=2)


print(__doc__)
if len(argv) > 1:
    if argv[1] == "-h":
        print(
            """Running this script with no file arguments will convert every .py file in the
current directory and all subdirectories to an .ipynb.

Usage:
python convert_py_to_ipynb.py
python convert_py_to_ipynb.py <files...>"""
        )
    else:
        for file in argv[1:]:
            convert(file)
else:
    for path in glob.glob(f"{os.path.dirname(__file__)}/**/*.py", recursive=True):
        if path != __file__:
            convert(path)

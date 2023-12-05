# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Update copyright on files
"""

import fileinput
from argparse import ArgumentParser
from datetime import date
from sys import exit
from typing import List


def arguments():
    """
    Parse arguments for copyright script
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        metavar="files",
        type=str,
        default=[],
        nargs="*",
        help="python files to be updated",
    )
    args = parser.parse_args()
    return args.files


def edit_files(files: List[str], copyright_line: str):
    """
    Edit files in place
    """
    for file in files:
        with fileinput.FileInput(file, inplace=True) as fh:
            for line in fh:
                if line.startswith(r"# Copyright") and copyright_line not in line:
                    print(line.replace(line.split(" ", 4)[-2], copyright_line), end="")
                    yield file
                elif (
                    "conf.py" in file
                    and line.startswith(r"copyright")
                    and copyright_line not in line
                ):
                    print(
                        line.replace(line.split(" ", 4)[-2], f'"{copyright_line},'),
                        end="",
                    )
                else:
                    print(line, end="")


def main():
    """Run the copyright updater"""
    edited = list(edit_files(arguments(), rf"2021-{date.today().year}"))  # noqa: DTZ011

    if edited != []:
        for file in edited:
            print(f"Updated {file}")
        exit(1)


main()

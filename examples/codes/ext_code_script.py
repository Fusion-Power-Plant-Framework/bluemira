# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Dummy external code
"""

import os
from argparse import ArgumentParser


def get_filename():
    """Get filename of script"""
    return os.path.realpath(__file__)


def parse_args():
    """
    Parse arguments for copyright script
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--add-header",
        action="store_true",
        default=False,
        help="add header to file",
    )
    parser.add_argument(
        "--number", action="store_true", default=False, help="add line numbers"
    )
    parser.add_argument("infile", type=str)
    parser.add_argument("outfile", type=str)
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    with open(args.infile) as _if:
        lines = _if.readlines()

    with open(args.outfile, "w") as of:
        if args.add_header:
            of.write("# this is a header\n")
        for no, line in enumerate(lines, start=1):
            of.write(f"{no if args.number else ''}    {line}")


if __name__ == "__main__":
    main()

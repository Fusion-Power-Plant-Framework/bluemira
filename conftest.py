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
Used by pytest for configuration like adding command line options.
"""

import matplotlib as mpl

import tests
from bluemira.base.file import try_get_bluemira_private_data_root


def pytest_addoption(parser):
    """
    Adds a custom command line option to pytest to control plotting and longrun.
    """
    parser.addoption(
        "--plotting-on",
        action="store_true",
        default=False,
        help="switch on plotting in tests",
    )
    parser.addoption(
        "--longrun",
        action="store_true",
        dest="longrun",
        default=False,
        help="enable longrundecorated tests",
    )
    parser.addoption(
        "--reactor",
        action="store_true",
        dest="reactor",
        default=False,
        help="enable reactor end-to-end test",
    )

    parser.addoption(
        "--private",
        action="store_true",
        dest="private",
        default=False,
        help="run tests that use private data",
    )

    parser.addoption(
        "--integration",
        action="store_true",
        dest="integration",
        default=False,
        help="enable tests for BLUEPRINT/bluemira integration",
    )


def pytest_configure(config):
    """
    Configures pytest with the plotting and longrun command line options.
    """
    if config.getoption("--plotting-on"):
        tests.PLOTTING = True
    else:
        # We're not displaying plots so use a display-less backend
        mpl.use("Agg")

    options = {
        "longrun": config.option.longrun,
        "reactor": config.option.reactor,
        "private": config.option.private,
        "integration": config.option.integration,
    }
    if options["private"] and try_get_bluemira_private_data_root() is None:
        raise ValueError("You cannot run private tests. Data directory not found")

    strings = []
    for name, value in options.items():
        if not value:
            strings.append(f"not {name}")

    logic_string = " and ".join(strings)

    setattr(config.option, "markexpr", logic_string)

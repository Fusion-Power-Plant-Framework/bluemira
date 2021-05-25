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
Used by pytest for configuration like adding command line options.
"""

import matplotlib as mpl

import tests


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


def pytest_configure(config):
    """
    Configures pytest with the plotting and longrun command line options.
    """
    if config.getoption("--plotting-on"):
        tests.PLOTTING = True
    else:
        # We're not displaying plots so use a display-less backend
        mpl.use("Agg")
    if not config.option.longrun and not config.option.reactor:
        setattr(config.option, "markexpr", "not longrun and not reactor")
    elif not config.option.longrun:
        setattr(config.option, "markexpr", "not longrun")

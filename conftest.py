# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Used by pytest for configuration like adding command line options.
"""

import os
from contextlib import suppress
from pathlib import Path
from unittest import mock

import matplotlib as mpl
import pytest

from bluemira.base.file import get_bluemira_path, try_get_bluemira_private_data_root


def pytest_addoption(parser):
    """
    Adds a custom command line option to pytest to control plotting and longrun.
    """
    parser.addoption(
        "--plotting-on",
        action="store_true",
        default=False,
        help="switch on interactive plotting in tests",
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


def pytest_configure(config):
    """
    Configures pytest with the plotting and longrun command line options.
    """
    if not config.getoption("--plotting-on"):
        # We're not displaying plots so use a display-less backend
        mpl.use("Agg")
        # Disable CAD viewer by mocking out FreeCAD API's displayer.
        # Note that if we use a new CAD backend, this must be changed.
        with suppress(ImportError):
            mock.patch("bluemira.codes._polyscope.ps").start()
        mock.patch("bluemira.codes._freecadapi.show_cad").start()

    options = {
        "longrun": config.option.longrun,
        "reactor": config.option.reactor,
        "private": config.option.private,
    }
    if options["private"] and try_get_bluemira_private_data_root() is None:
        raise ValueError("You cannot run private tests. Data directory not found")

    strings = []
    for name, value in options.items():
        if not value:
            strings.append(f"not {name}")

    logic_string = " and ".join(strings)

    config.option.markexpr = logic_string
    config.option.basetemp = (
        basetemp
        if (basetemp := os.environ.get("PYTEST_TMPDIR", config.option.basetemp))
        else Path(get_bluemira_path("", subfolder="generated_data"), "test_data")
    )


@pytest.fixture(autouse=True)
def _plot_show_and_close(request):
    """Fixture to show and close plots

    Notes
    -----
    Does not do anything if testclass marked with 'classplot'
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cls = request.node.getparent(pytest.Class)

    if cls and "classplot" in cls.keywords:
        yield
    else:
        yield
        plt.show()
        plt.close()


@pytest.fixture(scope="class", autouse=True)
def _plot_show_and_close_class(request):
    """Fixture to show and close plots for marked classes

    Notes
    -----
    Only shows and closes figures on classes marked with 'classplot'
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if "classplot" in request.keywords:
        yield
        plt.show()
        plt.close()
    else:
        yield

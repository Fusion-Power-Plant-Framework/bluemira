# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Used by pytest for configuration like adding command line options.
"""

import doctest
import os
from contextlib import suppress
from pathlib import Path
from unittest import mock

import matplotlib as mpl
import numpy as np
import pytest
from sybil import Sybil
from sybil.parsers.rest import (
    DocTestParser,
    PythonCodeBlockParser,
)

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.file import get_bluemira_path, try_get_bluemira_private_data_root
from bluemira.base.reactor import ComponentManager
from bluemira.geometry.tools import make_circle


def setup_sybil_namespace(namespace):
    """Sybil namespace base creation

    Returns
    -------
    expanded namespace for docstring code blocks
    """
    namespace["MyPlasma"] = type("MyPlasma", (ComponentManager,), {})
    namespace["MyTfCoils"] = type("MyTfCoils", (ComponentManager,), {})
    namespace["build_plasma"] = lambda: namespace["MyPlasma"](
        Component("xyz", children=[PhysicalComponent("sh", make_circle())])
    )
    namespace["build_tf_coils"] = lambda: namespace["MyTfCoils"](
        Component("xy", children=[PhysicalComponent("sh", make_circle())])
    )
    namespace["np"] = np
    namespace["plasma_shape"] = make_circle()
    namespace["wall_shape"] = make_circle()
    namespace["breeding_zone_shape"] = make_circle()
    return namespace


rest_examples = Sybil(
    parsers=[
        DocTestParser(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE),
        PythonCodeBlockParser(),
    ],
    patterns=["*.py", "*.rst"],
    excludes=["index.rst"],  # autoapi files would be tested twice
    fixtures=["tmp_path"],
    setup=setup_sybil_namespace,
)


pytest_collect_file = rest_examples.pytest()


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
        help="enable long running tests",
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

    Raises
    ------
    ValueError
        if private test actived without access to the data
    """
    if not config.option.plotting_on:
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

    if try_get_bluemira_private_data_root() is None and (
        options["private"] or "private" in config.getoption("markexpr", "")
    ):
        raise ValueError("You cannot run private tests. Data directory not found")

    config.option.markexpr = config.getoption(
        "markexpr",
        " and ".join([f"not {name}" for name, value in options.items() if not value]),
    )
    if not config.option.markexpr:
        config.option.markexpr = " and ".join([
            f"not {name}" for name, value in options.items() if not value
        ])

    if "private" not in config.option.markexpr and not options["private"]:
        config.option.markexpr += "and not private"

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

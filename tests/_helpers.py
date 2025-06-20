# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of general helper functions for tests.
"""

import contextlib
import json
from pathlib import Path
from unittest import mock

import pytest

from bluemira.base.file import get_bluemira_root
from bluemira.equilibria.coils import Coil, CoilSet


def combine_text_mock_write_calls(open_mock: mock.MagicMock) -> str:
    """
    Combine the 'write' calls made on a mock created using the
    unittest.mock.mock_open function.

    You may have created a suitable mock object using the following

    .. code-block:: python  # doctest: +SKIP

        with mock.patch("builtins.open", new_callable=mock.mock_open) as open_mock:
            # write a file

        text_written_to_file = combine_text_mock_write_calls(open_mock)

    """
    write_call_args = open_mock.return_value.write.call_args_list
    if len(write_call_args) == 0:
        return ""
    return "".join([call.args[0] for call in write_call_args])


@contextlib.contextmanager
def file_exists(good_file_path: str, isfile_ref: str):
    """
    Context manager to mock os.path.isfile to return True for a specific
    file path.

    This is useful if the code under test checks for the existence of
    several files, but you want to mock such that only one of those
    files exists.

    .. code-block:: python  # doctest: +SKIP

        with file_exists("some/file", "module.under.test.os.path.isfile"):
            # do some work pretending "some/file" is a file

    Yields
    ------
    :
        Mocked file
    """

    def new_isfile(path: str) -> bool:
        if Path(good_file_path).resolve() == Path(path).resolve():
            return True
        return Path(path).exists and not Path(path).is_dir()

    with mock.patch(isfile_ref, new=new_isfile) as is_file_mock:
        yield is_file_mock


def skipif_import_error(*module_name: str) -> pytest.MarkDecorator:
    """Create skipif marker for unimportable modules"""
    skip = []
    for m in module_name:
        try:
            __import__(m)
            skip.append(False)
        except ImportError:  # noqa: PERF203
            skip.append(True)

    if len(module_name) == 1:
        reason = f"dependency {module_name[0]} not found"
    else:
        modules = ", ".join(module_name[no] for no, i in enumerate(skip) if i)
        reason = f"dependencies {modules} not found"

    return pytest.mark.skipif(any(skip), reason=reason)


def add_plot_title(func, request):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cls = request.node.getparent(pytest.Class)
    clstitle = "" if cls is None else cls.name

    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        for fig in list(map(plt.figure, plt.get_fignums())):
            fig.suptitle(
                f"{fig.get_suptitle()} {clstitle}::"
                f"{request.node.getparent(pytest.Function).name}"
            )
        return res

    return wrapper


def read_coil_json(name):
    """Read coil info and return data."""
    root_path = get_bluemira_root()

    file_path = Path(root_path, "tests/equilibria/test_data/coilsets/", name)
    with open(file_path) as f:
        return json.load(f)


def read_in_coilset(filename):
    """Make a coilset from position info. Currents not set."""

    data = read_coil_json(filename)
    coils = []
    for xi, zi, dxi, dzi, name, ctype in zip(
        data["xc"],
        data["zc"],
        data["dxc"],
        data["dzc"],
        data["coil_names"],
        data["coil_types"],
        strict=False,
    ):
        coil = Coil(x=xi, z=zi, dx=dxi, dz=dzi, name=name, ctype=ctype)
        coils.append(coil)
    return CoilSet(*coils)

# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of general helper functions for tests.
"""

import contextlib
from pathlib import Path
from unittest import mock

import pytest


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

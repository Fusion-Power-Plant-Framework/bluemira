# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tests for utilities for external code integration
"""

from unittest import mock

import pytest

from bluemira.codes.utilities import run_subprocess


class TestRunSubprocess:
    MODULE_REF = "bluemira.codes.utilities"

    def setup_method(self):
        self.popen_patch = mock.patch(
            f"{self.MODULE_REF}.subprocess.Popen", new=mock.mock_open()
        )
        self.popen_mock = self.popen_patch.start()
        self.popen_mock.return_value.returncode = 0

    def teardown_method(self):
        self.popen_patch.stop()

    def test_passes_command_and_kwargs_to_Popen(self):
        command = ["git", "status", "--long", "-v"]
        kwargs = {"env": {"env_var": "value"}}

        return_code = run_subprocess(command, **kwargs)

        self.popen_mock.assert_called_once()
        call_args, call_kwargs = self.popen_mock.call_args
        assert call_args[0] == command
        assert call_kwargs["env"] == {"env_var": "value"}
        assert return_code == 0

    @pytest.mark.parametrize("shell", [True, 1, None])
    def test_Popen_shell_kwarg_is_always_False(self, shell):
        command = ["git", "status", "--long", "-v"]
        kwargs = {"shell": shell}

        run_subprocess(command, **kwargs)

        self.popen_mock.assert_called_once()
        _, call_kwargs = self.popen_mock.call_args
        assert call_kwargs["shell"] is False

    def test_cwd_set_using_run_directory(self):
        command = ["git", "status", "--long", "-v"]
        run_directory = "/some/path"

        run_subprocess(command, run_directory=run_directory)

        self.popen_mock.assert_called_once()
        _, call_kwargs = self.popen_mock.call_args
        assert call_kwargs["cwd"] == run_directory

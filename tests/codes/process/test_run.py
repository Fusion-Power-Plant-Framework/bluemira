# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from unittest import mock

import pytest

from bluemira.codes import process
from bluemira.codes.process._run import Run
from bluemira.codes.process.params import ProcessSolverParams
from tests.codes.process.utilities import PARAM_FILE


class TestRun:
    def setup_method(self):
        self.default_pf = ProcessSolverParams.from_json(PARAM_FILE)

        self._subprocess_patch = mock.patch("bluemira.codes.interface.run_subprocess")
        self.run_subprocess_mock = self._subprocess_patch.start()
        self.run_subprocess_mock.return_value = 0

    def teardown_method(self):
        self._subprocess_patch.stop()

    @pytest.mark.parametrize("run_func", ["run", "runinput"])
    def test_run_func_calls_subprocess_with_in_dat_path(self, run_func):
        with mock.patch.object(Run, "_get_epsvmc"):
            run = Run(self.default_pf, "input/path_IN.DAT")
            run._epsvmc = 1
            getattr(run, run_func)()

        assert self.run_subprocess_mock.call_args[0][0] == [
            process.BINARY,
            "-i",
            "input/path_IN.DAT",
        ]

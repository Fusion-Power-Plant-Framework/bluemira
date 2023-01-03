# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
        run = Run(self.default_pf, "input/path_IN.DAT")

        getattr(run, run_func)()

        self.run_subprocess_mock.assert_called_once_with(
            [process.BINARY, "-i", "input/path_IN.DAT"]
        )

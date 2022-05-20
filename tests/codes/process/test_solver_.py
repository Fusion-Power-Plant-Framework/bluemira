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

import filecmp
import os
import tempfile
from unittest import mock

import pytest

from bluemira.base.config import Configuration
from bluemira.base.parameter import ParameterFrame
from bluemira.codes.process import ENABLED
from bluemira.codes.process.solver_ import ProcessSolver, RunMode


class TestProcessSolver:

    MODULE_REF = "bluemira.codes.process.solver_"

    @mock.patch(f"{MODULE_REF}.bluemira_warn")
    def test_bluemira_warning_if_build_config_has_unknown_arg(self, bm_warn_mock):
        build_config = {"not_an_arg": 0}

        ProcessSolver(ParameterFrame(), build_config)

        bm_warn_mock.assert_called_once()


@pytest.mark.longrun
@pytest.mark.skipif(not ENABLED, reason="PROCESS is not installed on the system.")
class TestProcessSolverSystem:

    DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

    def test_run_mode_run_outputs_process_files(self):
        params = Configuration()
        run_dir = tempfile.TemporaryDirectory()
        build_config = {"run_dir": run_dir.name}
        solver = ProcessSolver(params, build_config)

        solver.execute(RunMode.RUN)

        assert os.path.isfile(os.path.join(run_dir.name, "IN.DAT"))
        assert os.path.isfile(os.path.join(run_dir.name, "MFILE.DAT"))

    def test_run_mode_read_updates_params_from_mfile(self):
        params = Configuration()
        assert params.r_tf_in_centre != pytest.approx(2.6354)

        build_config = {"run_dir": self.DATA_DIR}

        solver = ProcessSolver(params, build_config)
        solver.execute(RunMode.READ)

        # Expected value comes from ./test_data/MFILE.DAT
        assert solver.params.r_tf_in_centre == pytest.approx(2.6354)

    def test_run_mode_runinput_does_not_edit_template(self):
        params = Configuration()
        run_dir = tempfile.TemporaryDirectory()
        template_path = os.path.join(self.DATA_DIR, "IN.DAT")
        build_config = {
            "run_dir": run_dir.name,
            "template_in_dir": template_path,
        }
        solver = ProcessSolver(params, build_config)

        solver.execute(RunMode.RUN)

        assert os.path.isfile(os.path.join(run_dir.name, "IN.DAT"))
        filecmp.cmp(os.path.join(run_dir.name, "IN.DAT"), template_path)
        assert os.path.isfile(os.path.join(run_dir.name, "MFILE.DAT"))

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

import copy
import filecmp
import os
import re
import tempfile
from unittest import mock

import pytest

from bluemira.base.config import Configuration
from bluemira.base.parameter import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.process import ENABLED
from bluemira.codes.process._solver import RunMode, Solver
from tests.codes.process import utilities as utils


class TestSolver:

    MODULE_REF = "bluemira.codes.process._solver"

    @classmethod
    def setup_class(cls):
        cls._mfile_patch = mock.patch(
            "bluemira.codes.process._teardown.MFile", new=utils.FakeMFile
        )
        cls.mfile_mock = cls._mfile_patch.start()

        cls._process_dict_patch = mock.patch(
            "bluemira.codes.process._teardown.PROCESS_DICT", new=utils.FAKE_PROCESS_DICT
        )
        cls.process_mock = cls._process_dict_patch.start()

    @classmethod
    def teardown_class(cls):
        cls._mfile_patch.stop()
        cls._process_dict_patch.stop()

    @mock.patch(f"{MODULE_REF}.bluemira_warn")
    def test_bluemira_warning_if_build_config_has_unknown_arg(self, bm_warn_mock):
        build_config = {"not_an_arg": 0, "also_not_an_arg": 0}

        Solver(ParameterFrame(), build_config)

        bm_warn_mock.assert_called_once()
        call_args, _ = bm_warn_mock.call_args
        assert re.match(
            ".* unknown .* arguments: 'not_an_arg', 'also_not_an_arg'", call_args[0]
        )

    def test_none_mode_does_not_alter_parameters(self):
        params = Configuration()
        solver = Solver(copy.deepcopy(params), {})

        solver.execute(RunMode.NONE)

        assert solver.params.to_dict() == params.to_dict()

    def test_get_raw_variables_retrieves_parameters(self):
        params = Configuration()
        solver = Solver(copy.deepcopy(params), {"read_dir": utils.DATA_DIR})
        solver.execute(RunMode.READ)

        assert solver.get_raw_variables("kappa_95") == [1.65]

    def test_get_raw_variables_CodesError_given_solver_not_run(self):
        params = Configuration()
        solver = Solver(copy.deepcopy(params), {"read_dir": utils.DATA_DIR})

        with pytest.raises(CodesError):
            solver.get_raw_variables("kappa_95")

    def test_get_species_fraction_retrieves_parameter_value(self):
        params = Configuration()
        solver = Solver(copy.deepcopy(params), {"read_dir": utils.DATA_DIR})
        solver.execute(RunMode.READ)

        assert solver.get_species_fraction("H") == pytest.approx(0.74267)
        assert solver.get_species_fraction("W") == pytest.approx(5e-5)


@pytest.mark.skipif(not ENABLED, reason="PROCESS is not installed on the system.")
class TestSolverSystem:

    DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

    @pytest.mark.longrun
    def test_run_mode_outputs_process_files(self):
        params = Configuration()
        run_dir = tempfile.TemporaryDirectory()
        build_config = {"run_dir": run_dir.name}
        solver = Solver(params, build_config)

        solver.execute(RunMode.RUN)

        assert os.path.isfile(os.path.join(run_dir.name, "IN.DAT"))
        assert os.path.isfile(os.path.join(run_dir.name, "MFILE.DAT"))

    @pytest.mark.parametrize("run_mode", [RunMode.READ, RunMode.READALL])
    def test_read_mode_updates_params_from_mfile(self, run_mode):
        params = Configuration()
        # Assert here to check the parameter is actually changing
        assert params.r_tf_in_centre != pytest.approx(2.6354)
        build_config = {"read_dir": self.DATA_DIR}

        solver = Solver(params, build_config)
        solver.execute(run_mode)

        # Expected value comes from ./test_data/MFILE.DAT
        assert solver.params.r_tf_in_centre == pytest.approx(2.6354)

    @pytest.mark.longrun
    def test_runinput_mode_does_not_edit_template(self):
        params = Configuration()
        run_dir = tempfile.TemporaryDirectory()
        template_path = os.path.join(self.DATA_DIR, "IN.DAT")
        build_config = {
            "run_dir": run_dir.name,
            "template_in_dat": template_path,
        }

        solver = Solver(params, build_config)
        solver.execute(RunMode.RUN)

        assert os.path.isfile(os.path.join(run_dir.name, "IN.DAT"))
        filecmp.cmp(os.path.join(run_dir.name, "IN.DAT"), template_path)
        assert os.path.isfile(os.path.join(run_dir.name, "MFILE.DAT"))

    def test_get_species_data_returns_row_vectors(self):
        # imp_data.__file__ is returning './__init__.py' for some reason
        # god knows why
        temp, loss_f, z_eff = Solver.get_species_data("H")

        assert isinstance(temp.size, int) == 1 and temp.size > 0
        assert isinstance(loss_f.size, int) == 1 and loss_f.size > 0
        assert isinstance(z_eff.size, int) == 1 and z_eff.size > 0

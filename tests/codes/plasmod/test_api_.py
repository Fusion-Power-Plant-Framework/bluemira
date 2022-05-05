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
import re
from unittest import mock

import pytest

from bluemira.base.config import Configuration
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.api_ import Run, Setup, Solver, Teardown
from bluemira.codes.plasmod.constants import BINARY as PLASMOD_BINARY
from tests._helpers import combine_text_mock_write_calls

_MODULE_REF = "bluemira.codes.plasmod.api_"


class TestPlasmodSetup:
    def setup_method(self):
        self.default_pf = Configuration()
        self.input_file = "/path/to/input.dat"

    def test_inputs_updated_from_problem_settings_on_init(self):
        problem_settings = {
            "v_loop": -1.5e-3,
            "q_heat": 1.5,
            "nx": 25,
        }

        setup = Setup(self.default_pf, problem_settings, self.input_file)

        assert setup.inputs.v_loop == -1.5e-3
        assert setup.inputs.q_heat == 1.5
        assert setup.inputs.nx == 25

    def test_update_inputs_changes_input_values(self):
        new_inputs = {
            "v_loop": -1.5e-3,
            "q_heat": 1.5,
            "nx": 25,
        }
        setup = Setup(self.default_pf, {}, self.input_file)

        setup.update_inputs(new_inputs)

        assert setup.inputs.v_loop == -1.5e-3
        assert setup.inputs.q_heat == 1.5
        assert setup.inputs.nx == 25

    def test_update_inputs_shows_warning_if_input_unknown(self):
        new_inputs = {"not_a_param": -1.5e-3}
        setup = Setup(self.default_pf, {}, self.input_file)

        with mock.patch(f"{_MODULE_REF}.bluemira_warn") as bm_warn:
            setup.update_inputs(new_inputs)

        bm_warn.assert_called_once()

    def test_run_writes_plasmod_dat_file(self):
        problem_settings = {"v_loop": -1.5e-3, "q_heat": 1.5, "nx": 25}
        setup = Setup(self.default_pf, problem_settings, "/some/file/path.dat")

        with mock.patch("builtins.open", new_callable=mock.mock_open) as open_mock:
            setup.run()

        open_mock.assert_called_once_with("/some/file/path.dat", "w")
        output = combine_text_mock_write_calls(open_mock)
        assert re.search(r"^ *v_loop +-0.150+E-02\n", output, re.MULTILINE)
        assert re.search(r"^ *q_heat +0.150+E\+01\n", output, re.MULTILINE)
        assert re.search(r"^ *nx +25\n", output, re.MULTILINE)

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_CodesError_if_writing_to_plasmod_dat_file_fails(self, open_mock):
        problem_settings = {"v_loop": -1.5e-3, "q_heat": 1.5, "nx": 25}
        setup = Setup(
            self.default_pf,
            problem_settings=problem_settings,
            input_file="/some/file/path.dat",
        )
        open_mock.side_effect = OSError

        with pytest.raises(CodesError):
            setup.run()

    def test_mock_does_not_write_dat_file(self):
        setup = Setup(self.default_pf, {}, self.input_file)

        with mock.patch("builtins.open", new_callable=mock.mock_open) as open_mock:
            setup.mock()

        open_mock.assert_not_called()

    def test_read_does_not_write_dat_file(self):
        setup = Setup(self.default_pf, {}, self.input_file)

        with mock.patch("builtins.open", new_callable=mock.mock_open) as open_mock:
            setup.read()

        open_mock.assert_not_called()


class TestPlasmodRun:

    RUN_SUBPROCESS_REF = f"{_MODULE_REF}.run_subprocess"

    def setup_method(self):
        self._run_subprocess_patch = mock.patch(self.RUN_SUBPROCESS_REF)
        self.run_subprocess_mock = self._run_subprocess_patch.start()
        self.run_subprocess_mock.return_value = 0
        self.default_pf = Configuration()

    def teardown_method(self):
        self._run_subprocess_patch.stop()

    @pytest.mark.parametrize(
        "arg, arg_num",
        [
            ("plasmod_binary", 0),
            ("input.dat", 1),
            ("output.dat", 2),
            ("profiles.dat", 3),
        ],
    )
    def test_run_calls_subprocess_with_argument_in_position(self, arg, arg_num):
        run = Run(
            self.default_pf,
            "input.dat",
            "output.dat",
            "profiles.dat",
            binary="plasmod_binary",
        )

        run.run()

        self.run_subprocess_mock.assert_called_once()
        args, _ = self.run_subprocess_mock.call_args
        assert args[0][arg_num] == arg

    def test_run_raises_CodesError_given_run_subprocess_raises_OSError(self):
        self.run_subprocess_mock.side_effect = OSError
        run = Run(self.default_pf, "input.dat", "output.dat", "profiles.dat")

        with pytest.raises(CodesError):
            run.run()

    def test_run_raises_CodesError_given_run_process_returns_non_zero_exit_code(self):
        self.run_subprocess_mock.return_value = 1
        run = Run(self.default_pf, "input.dat", "output.dat", "profiles.dat")

        with pytest.raises(CodesError):
            run.run()


class TestPlasmodTeardown:

    plasmod_out_sample = (
        "     betan      0.14092930140E+0002\n"
        "      fbs       0.14366031154E+0002\n"
        "      rli       0.16682353334E+0002\n"
        " i_flag           1\n"
    )

    def setup_method(self):
        self.default_pf = Configuration()
        self.default_pf

    @pytest.mark.parametrize("run_mode_func", ["run", "read"])
    def test_run_mode_function_updates_plasmod_params_from_file(self, run_mode_func):
        teardown = Teardown(
            self.default_pf, "/path/to/output/file.csv", "/path/to/profiles/file.csv"
        )

        with mock.patch(
            "builtins.open",
            new_callable=mock.mock_open,
            read_data=self.plasmod_out_sample,
        ):
            getattr(teardown, run_mode_func)()

        assert teardown.params["beta_N"] == pytest.approx(0.14092930140e2)
        assert teardown.params["f_bs"] == pytest.approx(0.14366031154e2)
        assert teardown.params["l_i"] == pytest.approx(0.16682353334e2)

    def test_mock_leaves_plasmod_params_with_defaults(self):
        default_pf_copy = copy.deepcopy(self.default_pf)
        teardown = Teardown(
            self.default_pf, "/path/to/output/file.csv", "/path/to/profiles/file.csv"
        )

        with mock.patch(
            "builtins.open",
            new_callable=mock.mock_open,
            read_data=self.plasmod_out_sample,
        ):
            teardown.mock()

        assert teardown.params["beta_N"] == default_pf_copy["beta_N"]
        assert teardown.params["f_bs"] == default_pf_copy["f_bs"]
        assert teardown.params["l_i"] == default_pf_copy["l_i"]

    @pytest.mark.parametrize("run_mode_func", ["run", "read"])
    def test_CodesError_if_output_files_cannot_be_read(self, run_mode_func):
        teardown = Teardown(
            self.default_pf, "/path/to/output/file.csv", "/path/to/profiles/file.csv"
        )

        with mock.patch("builtins.open", side_effect=OSError):
            with pytest.raises(CodesError):
                getattr(teardown, run_mode_func)()

    @pytest.mark.parametrize("run_mode_func", ["run", "read"])
    def test_run_mode_function_opens_both_output_files(self, run_mode_func):
        teardown = Teardown(
            self.default_pf, "/path/to/output/file.csv", "/path/to/profiles/file.csv"
        )

        with mock.patch(
            "builtins.open",
            new_callable=mock.mock_open,
            read_data=self.plasmod_out_sample,
        ) as open_mock:
            getattr(teardown, run_mode_func)()

        assert open_mock.call_count == 2
        call_args = [call.args for call in open_mock.call_args_list]
        assert ("/path/to/output/file.csv", "r") in call_args
        assert ("/path/to/profiles/file.csv", "r") in call_args


class TestPlasmodSolver:
    def setup_method(self):
        self.default_pf = Configuration()

    @pytest.mark.parametrize(
        "key, default",
        [
            ("binary", PLASMOD_BINARY),
            ("problem_settings", {}),
            ("input_file", Solver.DEFAULT_INPUT_FILE),
            ("output_file", Solver.DEFAULT_OUTPUT_FILE),
            ("profiles_file", Solver.DEFAULT_PROFILES_FILE),
        ],
    )
    def test_init_sets_default_build_config_value(self, key, default):
        solver = Solver(self.default_pf)

        assert getattr(solver, key) == default

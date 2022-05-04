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
import re
from unittest import mock

import pytest

from bluemira.base.parameter import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.api_ import Setup
from tests._helpers import combine_text_mock_write_calls


class TestPlasmodSetup:

    MODULE_REF = "bluemira.codes.plasmod.api_"

    def setup_method(self):
        self.default_pf = ParameterFrame()

    def test_inputs_updated_from_problem_settings_on_init(self):
        problem_settings = {
            "v_loop": -1.5e-3,
            "q_heat": 1.5,
            "nx": 25,
        }

        setup = Setup(self.default_pf, problem_settings=problem_settings)

        assert setup.inputs.v_loop == -1.5e-3
        assert setup.inputs.q_heat == 1.5
        assert setup.inputs.nx == 25

    def test_update_inputs_changes_input_values(self):
        new_inputs = {
            "v_loop": -1.5e-3,
            "q_heat": 1.5,
            "nx": 25,
        }
        setup = Setup(self.default_pf)

        setup.update_inputs(new_inputs)

        assert setup.inputs.v_loop == -1.5e-3
        assert setup.inputs.q_heat == 1.5
        assert setup.inputs.nx == 25

    def test_update_inputs_shows_warning_if_input_unknown(self):
        new_inputs = {"not_a_param": -1.5e-3}
        setup = Setup(self.default_pf)

        with mock.patch(f"{self.MODULE_REF}.bluemira_warn") as bm_warn:
            setup.update_inputs(new_inputs)

        bm_warn.assert_called_once()

    def test_run_writes_plasmod_dat_file(self):
        problem_settings = {"v_loop": -1.5e-3, "q_heat": 1.5, "nx": 25}
        setup = Setup(
            self.default_pf,
            problem_settings=problem_settings,
            input_file="/some/file/path.dat",
        )

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
        setup = Setup(self.default_pf)

        with mock.patch("builtins.open", new_callable=mock.mock_open) as open_mock:
            setup.mock()

        open_mock.assert_not_called()

    def test_read_does_not_write_dat_file(self):
        setup = Setup(self.default_pf)

        with mock.patch("builtins.open", new_callable=mock.mock_open) as open_mock:
            setup.read()

        open_mock.assert_not_called()

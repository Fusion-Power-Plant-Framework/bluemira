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

import os
import pytest
from unittest.mock import patch

from bluemira.base.file import get_bluemira_root
from tests.bluemira.codes.process.test_api import FRAME_LIST
from tests.BLUEPRINT.test_reactor import (
    config,
    build_config,
    build_tweaks,
    SmokeTestSingleNullReactor,
)

from bluemira.codes.process.api import PROCESS_ENABLED
from bluemira.codes.process import run


@pytest.mark.skipif(PROCESS_ENABLED is not True, reason="PROCESS install required")
class TestRun:
    test_reactor = SmokeTestSingleNullReactor(config, build_config, build_tweaks)
    test_reactor.add_parameters(FRAME_LIST)
    test_dir = os.path.join(
        get_bluemira_root(), "tests", "bluemira", "codes", "test_data"
    )
    test_indat = os.path.join(test_dir, "IN.DAT")

    def set_runmode(self, runmode):
        self.test_reactor.build_config["process_mode"] = runmode

    def run_PROCESS(self, runmode, **kwargs):
        """
        Set runmode in test reactor and run PROCESS.
        """
        self.set_runmode(runmode)
        return run.Run(self.test_reactor, run_dir=self.test_dir, **kwargs)

    @pytest.mark.parametrize(
        "runmode",
        # fmt: off
        [
            "RUN", "Run", "run",
            "RUNINPUT", "RunInput", "runinput",
            "READ", "Read", "read",
            "READALL", "ReadAll", "readall",
            "MOCK", "Mock", "mock",
        ]
        # fmt: on
    )
    @patch("bluemira.codes.process.run.Run._run")
    @patch("bluemira.codes.process.run.Run._runinput")
    @patch("bluemira.codes.process.run.Run._read")
    @patch("bluemira.codes.process.run.Run._readall")
    @patch("bluemira.codes.process.run.Run._mock")
    def test_runmode(
        self, mock_mock, mock_readall, mock_read, mock_runinput, mock_run, runmode
    ):
        """
        Test that the PROCESS runner accepts valid runmodes and calls the corresponding
        function.
        """
        self.run_PROCESS(runmode)

        # Check correct call was made.
        if runmode.upper() == "RUN":
            assert mock_run.call_count == 1
        elif runmode.upper() == "RERUN":
            assert mock_runinput.call_count == 1
        elif runmode.upper() == "READ":
            assert mock_read.call_count == 1
        elif runmode.upper() == "READALL":
            assert mock_readall.call_count == 1
        elif runmode.upper() == "MOCK":
            assert mock_mock.call_count == 1

    def test_invalid_runmode(self):
        """
        Test that an invalid runmode raise an error.
        """
        with pytest.raises(KeyError):
            self.run_PROCESS("FAKE")

    def test_read_mapping(self):
        """
        Test that parameters with a PROCESS mapping are read correctly.
        """
        with patch("bluemira.codes.process.run.Run._mock"):
            runner = self.run_PROCESS("MOCK")

        # Test that PROCESS params with read = False are not read.
        assert "cp" not in runner.read_mapping
        assert "dp" not in runner.read_mapping

        # Test that PROCESS params with read = True are read correctly.
        assert runner.read_mapping["ep"] == "e"
        assert runner.read_mapping["fp"] == "f"

        # Test that non-PROCESS params with read = True are not read.
        param = self.test_reactor.params.get_param("g")
        assert "PROCESS" not in param.mapping and param.mapping["FAKE_CODE"].read is True
        assert "gp" not in runner.read_mapping

    @patch("bluemira.codes.process.run.Run._load_PROCESS")
    @patch("bluemira.codes.process.run.Run._check_PROCESS_output")
    @patch("bluemira.codes.process.run.Run._run_subprocess")
    @patch("bluemira.codes.process.run.Run._clear_PROCESS_output")
    @patch("bluemira.codes.process.setup.PROCESSInputWriter.add_parameter")
    def test_runinput(
        self, mock_add_parameter, mock_clear, mock_run, mock_check, mock_load
    ):
        """
        Test in Run mode, that the correct functions are called during the run and that
        no BLUEPRINT parameters are called to be written to IN.DAT, as this should only
        occur in Run mode.
        """
        self.run_PROCESS("RUNINPUT")

        # Check that no calls were made to add_parameter.
        assert mock_add_parameter.call_count == 0

        # Check that correct run calls were made.
        assert mock_clear.call_count == 1
        assert mock_run.call_count == 1
        assert mock_check.call_count == 1
        assert mock_load.call_count == 1

    @patch("bluemira.codes.process.run.Run._load_PROCESS")
    @patch("bluemira.codes.process.run.Run._check_PROCESS_output")
    @patch("bluemira.codes.process.run.Run._run_subprocess")
    @patch("bluemira.codes.process.run.Run._clear_PROCESS_output")
    @patch("bluemira.codes.process.setup.PROCESSInputWriter.add_parameter")
    def test_run_with_params_to_update(
        self, mock_add_parameter, mock_clear, mock_run, mock_check, mock_load
    ):
        """
        Test in Rerun mode, that only parameters specified in params_to_update are called
        to be written to IN.DAT and that the correct functions are called during the run.
        """
        self.test_reactor.build_config["params_to_update"] = ["d", "f"]
        self.run_PROCESS("RUN")

        # Check the right amount of calls were made to add_parameter.
        assert mock_add_parameter.call_count == 2

        # Check that the dummy values with write = True were written.
        mock_add_parameter.assert_any_call("dp", 3)
        mock_add_parameter.assert_any_call("fp", 5)

        # Check that correct run calls were made.
        assert mock_clear.call_count == 1
        assert mock_run.call_count == 1
        assert mock_check.call_count == 1
        assert mock_load.call_count == 1

    @patch("bluemira.codes.process.run.Run._load_PROCESS")
    @patch("bluemira.codes.process.run.Run._check_PROCESS_output")
    @patch("bluemira.codes.process.run.Run._run_subprocess")
    @patch("bluemira.codes.process.run.Run._clear_PROCESS_output")
    @patch("bluemira.codes.process.setup.PROCESSInputWriter.add_parameter")
    def test_run_without_params_to_update(
        self, mock_add_parameter, mock_clear, mock_run, mock_check, mock_load
    ):
        """
        Test in Rerun mode, that parameters with a PROCESS mapping and mapping.write set
        as True are called be written to IN.DAT and that the correct functions are called
        during the run.
        """
        self.test_reactor.build_config["params_to_update"] = None
        self.run_PROCESS("RUN")

        # Check the right amount of calls were made to add_parameter.
        number_of_expected_calls = 0
        for param in self.test_reactor.params.get_parameter_list():
            if param.mapping is not None and "PROCESS" in param.mapping:
                if param.mapping["PROCESS"].write:
                    number_of_expected_calls += 1
        assert mock_add_parameter.call_count == number_of_expected_calls

        # Check that the dummy values with write = True were written.
        mock_add_parameter.assert_any_call("dp", 3)
        mock_add_parameter.assert_any_call("fp", 5)

        # Check that correct run calls were made.
        assert mock_clear.call_count == 1
        assert mock_run.call_count == 1
        assert mock_check.call_count == 1
        assert mock_load.call_count == 1

    @patch("bluemira.codes.process.run.Run._load_PROCESS")
    @patch("bluemira.codes.process.run.Run._check_PROCESS_output")
    @patch("bluemira.codes.process.run.Run._run_subprocess")
    @patch("bluemira.codes.process.run.Run._clear_PROCESS_output")
    @patch("bluemira.codes.process.setup.PROCESSInputWriter.add_parameter")
    def test_read(self, mock_add_parameter, mock_clear, mock_run, mock_check, mock_load):
        """
        Test in Read mode, that the correct calls are made and in particular that
        no calls are made to the clear, run, and check process functions.
        """
        self.run_PROCESS("READ")

        # Check that correct run calls were made.
        assert mock_clear.call_count == 0
        assert mock_run.call_count == 0
        assert mock_check.call_count == 0
        assert mock_load.call_count == 1
        assert mock_add_parameter.call_count == 0

    @patch("bluemira.codes.process.run.Run._load_PROCESS")
    @patch("bluemira.codes.process.run.Run._check_PROCESS_output")
    @patch("bluemira.codes.process.run.Run._run_subprocess")
    @patch("bluemira.codes.process.run.Run._clear_PROCESS_output")
    @patch("bluemira.codes.process.setup.PROCESSInputWriter.add_parameter")
    def test_readall(
        self, mock_add_parameter, mock_clear, mock_run, mock_check, mock_load
    ):
        """
        Test in ReadAll mode, that the correct calls are made and in particular that
        no calls are made to the clear, run, and check process functions.
        """
        self.run_PROCESS("READALL")

        # Check that correct run calls were made.
        assert mock_clear.call_count == 0
        assert mock_run.call_count == 0
        assert mock_check.call_count == 0
        assert mock_load.call_count == 1
        assert mock_add_parameter.call_count == 0

    @patch("tests.BLUEPRINT.test_reactor.SmokeTestSingleNullReactor.add_parameter")
    def test_mock(self, mock_add_parameter):
        """
        Test that the right amount of calls are made to the reactor's add_parameter
        function during a mock run.
        """
        self.run_PROCESS("MOCK")

        # Check the right amount of calls were made to add_parameter.
        assert mock_add_parameter.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__])

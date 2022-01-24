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

import pathlib
import shutil
import tempfile
from typing import Dict
from unittest.mock import patch

import pytest

from bluemira.base.builder import BuildConfig
from bluemira.codes.process import ENABLED as PROCESS_ENABLED
from bluemira.codes.process import Solver
from bluemira.codes.process.constants import NAME as PROCESS
from tests.bluemira.codes.process import (
    FRAME_LIST,
    INDIR,
    OUTDIR,
    PROCESSTestConfiguration,
)


class TestRun:
    config = {
        "Name": ("SMOKE-TEST", "Input"),
        "P_el_net": (580, "Input"),
        "tau_flattop": (3600, "Input"),
        "plasma_type": ("SN", "Input"),
        "reactor_type": ("Normal", "Input"),
        "CS_material": ("Nb3Sn", "Input"),
        "PF_material": ("NbTi", "Input"),
        "A": (3.1, "Input"),
        "n_CS": (5, "Input"),
        "n_PF": (6, "Input"),
        "f_ni": (0.1, "Input"),
        "fw_psi_n": (1.06, "Input"),
        "tk_ts": (0.05, "Input"),
        "tk_vv_in": (0.3, "Input"),
        "tk_sh_in": (0.3, "Input"),
        "tk_tf_side": (0.1, "Input"),
        "tk_bb_ib": (0.7, "Input"),
        "tk_sol_ib": (0.225, "Input"),
        "LPangle": (-15, "Input"),
    }

    build_config: Dict[str, BuildConfig]

    def setup_method(self):
        self.params = PROCESSTestConfiguration(self.config)
        self.build_config = {}
        if not pathlib.Path(OUTDIR).exists():
            pathlib.Path(OUTDIR).mkdir()
        self.run_dir = tempfile.mkdtemp(dir=OUTDIR)
        self.read_dir = INDIR

    def teardown_method(self):
        shutil.rmtree(self.run_dir)

    def set_runmode(self, runmode):
        self.build_config["mode"] = runmode

    def run_PROCESS(self, runmode, **kwargs):
        """
        Set runmode in test reactor and run PROCESS.
        """
        self.set_runmode(runmode)
        solver = Solver(
            self.params,
            self.build_config,
            self.run_dir,
            self.read_dir,
            **kwargs,
        )
        solver.run()
        return solver

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
    @patch("bluemira.codes.process.setup.Setup._run")
    @patch("bluemira.codes.process.setup.Setup._runinput")
    @patch("bluemira.codes.process.run.Run._run")
    @patch("bluemira.codes.process.run.Run._runinput")
    @patch("bluemira.codes.process.teardown.Teardown._run")
    @patch("bluemira.codes.process.teardown.Teardown._runinput")
    @patch("bluemira.codes.process.teardown.Teardown._read")
    @patch("bluemira.codes.process.teardown.Teardown._readall")
    @patch("bluemira.codes.process.teardown.Teardown._mock")
    def test_runmode(
        self,
        mock_mock,
        mock_readall,
        mock_read,
        mock_runinput_t,
        mock_run_t,
        mock_runinput_r,
        mock_run_r,
        mock_runinput_s,
        mock_run_s,
        runmode,
    ):
        """
        Test that the PROCESS runner accepts valid runmodes and calls the corresponding
        function.
        """
        with patch("bluemira.codes.process.run.Solver._enabled_check"):
            self.run_PROCESS(runmode)

        # Check correct call was made.
        if runmode.upper() == "RUN":
            assert mock_run_s.call_count == 1
            assert mock_run_r.call_count == 1
            assert mock_run_t.call_count == 1
        elif runmode.upper() == "RERUN":
            assert mock_runinput_s.call_count == 1
            assert mock_runinput_r.call_count == 1
            assert mock_runinput_t.call_count == 1
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

    def test_recv_mapping(self):
        """
        Test that parameters with a PROCESS mapping are read correctly.
        """
        self.params.add_parameters(FRAME_LIST)

        with patch("bluemira.codes.process.teardown.Teardown._mock"):
            runner = self.run_PROCESS("MOCK")

        # Test that PROCESS params with recv = False are not read.
        assert "cp" not in runner._recv_mapping
        assert "dp" not in runner._recv_mapping

        # Test that PROCESS params with recv = True are read correctly.
        assert runner._recv_mapping["ep"] == "e"
        assert runner._recv_mapping["fp"] == "f"

        # Test that non-PROCESS params with recv = True are not read.
        param = self.params.get_param("g")
        assert "PROCESS" not in param.mapping and param.mapping["FAKE_CODE"].recv is True
        assert "gp" not in runner._recv_mapping

    @patch("bluemira.codes.process.teardown.Teardown.load_PROCESS_run")
    @patch("bluemira.codes.process.teardown.Teardown._check_PROCESS_output_files")
    @patch("bluemira.codes.process.run.Run._run_subprocess")
    @patch("bluemira.codes.process.run.Run._clear_PROCESS_output")
    @patch("bluemira.codes.process.setup.PROCESSInputWriter.add_parameter")
    @pytest.mark.skipif(PROCESS_ENABLED is not True, reason="PROCESS install required")
    def test_runinput(
        self,
        mock_add_parameter,
        mock_clear,
        mock_run,
        mock_check,
        mock_load,
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

    @patch("bluemira.codes.process.teardown.Teardown.load_PROCESS_run")
    @patch("bluemira.codes.process.teardown.Teardown._check_PROCESS_output_files")
    @patch("bluemira.codes.process.run.Run._run_subprocess")
    @patch("bluemira.codes.process.run.Run._clear_PROCESS_output")
    @patch("bluemira.codes.process.setup.PROCESSInputWriter.add_parameter")
    @pytest.mark.skipif(PROCESS_ENABLED is not True, reason="PROCESS install required")
    def test_run_with_params_to_update(
        self,
        mock_add_parameter,
        mock_clear,
        mock_run,
        mock_check,
        mock_load,
    ):
        """
        Test in Rerun mode, that only parameters specified in params_to_update are called
        to be written to IN.DAT and that the correct functions are called during the run.
        """
        self.params.add_parameters(FRAME_LIST)

        self.build_config["params_to_update"] = ["d", "f"]
        self.run_PROCESS("RUN")

        # Check the right amount of calls were made to add_parameter.
        assert mock_add_parameter.call_count == 2

        # Check that the dummy values with send = True were written.
        mock_add_parameter.assert_any_call("dp", 3)
        mock_add_parameter.assert_any_call("fp", 5)

        # Check that correct run calls were made.
        assert mock_clear.call_count == 1
        assert mock_run.call_count == 1
        assert mock_check.call_count == 1
        assert mock_load.call_count == 1

    @patch("bluemira.codes.process.teardown.Teardown.load_PROCESS_run")
    @patch("bluemira.codes.process.teardown.Teardown._check_PROCESS_output_files")
    @patch("bluemira.codes.process.run.Run._run_subprocess")
    @patch("bluemira.codes.process.run.Run._clear_PROCESS_output")
    @patch("bluemira.codes.process.setup.PROCESSInputWriter.add_parameter")
    @pytest.mark.skipif(PROCESS_ENABLED is not True, reason="PROCESS install required")
    def test_run_without_params_to_update(
        self,
        mock_add_parameter,
        mock_clear,
        mock_run,
        mock_check,
        mock_load,
    ):
        """
        Test in Rerun mode, that parameters with a PROCESS mapping and mapping.send set
        as True are called be written to IN.DAT and that the correct functions are called
        during the run.
        """
        self.params.add_parameters(FRAME_LIST)

        self.build_config["params_to_update"] = None
        self.run_PROCESS("RUN")

        # Check the right amount of calls were made to add_parameter.
        number_of_expected_calls = 0
        for param in self.params.get_parameter_list():
            if param.mapping is not None and "PROCESS" in param.mapping:
                if param.mapping["PROCESS"].send:
                    number_of_expected_calls += 1
        assert mock_add_parameter.call_count == number_of_expected_calls

        # Check that the dummy values with send = True were written.
        mock_add_parameter.assert_any_call("dp", 3)
        mock_add_parameter.assert_any_call("fp", 5)

        # Check that correct run calls were made.
        assert mock_clear.call_count == 1
        assert mock_run.call_count == 1
        assert mock_check.call_count == 1
        assert mock_load.call_count == 1

    @patch("bluemira.codes.process.teardown.Teardown.load_PROCESS_run")
    @patch("bluemira.codes.process.teardown.Teardown._check_PROCESS_output_files")
    @patch("bluemira.codes.process.run.Run._run_subprocess")
    @patch("bluemira.codes.process.run.Run._clear_PROCESS_output")
    @patch("bluemira.codes.process.setup.PROCESSInputWriter.add_parameter")
    @pytest.mark.skipif(PROCESS_ENABLED is not True, reason="PROCESS install required")
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

    @patch("bluemira.codes.process.teardown.Teardown.load_PROCESS_run")
    @patch("bluemira.codes.process.teardown.Teardown._check_PROCESS_output_files")
    @patch("bluemira.codes.process.run.Run._run_subprocess")
    @patch("bluemira.codes.process.run.Run._clear_PROCESS_output")
    @patch("bluemira.codes.process.setup.PROCESSInputWriter.add_parameter")
    @pytest.mark.skipif(PROCESS_ENABLED is not True, reason="PROCESS install required")
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

    def test_mock(self):
        """
        Test that the right amount of calls are made to the reactor's add_parameter
        function during a mock run.
        """
        run = self.run_PROCESS("MOCK")

        bad_recv = []
        for name in run._recv_mapping.values():
            if run.params.get_param(name).source != f"{PROCESS} (Mock)":
                bad_recv.append(name)

        assert (
            len(bad_recv) == 0
        ), "Parameters were marked as recv in PROCESS mapping but were not mapped back into bluemira."


if __name__ == "__main__":
    pytest.main([__file__])

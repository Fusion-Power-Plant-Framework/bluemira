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
from unittest import mock

import pytest

from bluemira.base.config import Configuration
from bluemira.codes import process
from bluemira.codes.error import CodesError
from bluemira.codes.process._teardown import Teardown
from bluemira.codes.process.mapping import mappings as process_mappings
from bluemira.codes.utilities import add_mapping
from tests._helpers import file_exists
from tests.codes.process import utilities as utils


class TestTeardown:

    MODULE_REF = "bluemira.codes.process._teardown"
    IS_FILE_REF = f"{MODULE_REF}.os.path.isfile"

    @classmethod
    def setup_class(cls):
        cls._mfile_patch = mock.patch(f"{cls.MODULE_REF}.MFile", new=utils.FakeMFile)
        cls.mfile_mock = cls._mfile_patch.start()

        cls._process_dict_patch = mock.patch(
            f"{cls.MODULE_REF}.PROCESS_DICT", new=utils.FAKE_PROCESS_DICT
        )
        cls.process_mock = cls._process_dict_patch.start()

    @classmethod
    def teardown_class(cls):
        cls._mfile_patch.stop()
        cls._process_dict_patch.stop()

    def setup_method(self):
        self.default_pf = Configuration()
        add_mapping(process.NAME, self.default_pf, process_mappings)

    @pytest.mark.parametrize("run_func", ["run", "runinput"])
    def test_run_func_updates_bluemira_params_from_mfile(self, run_func):
        teardown = Teardown(self.default_pf, utils.RUN_DIR, None)

        with file_exists(os.path.join(utils.RUN_DIR, "MFILE.DAT"), self.IS_FILE_REF):
            getattr(teardown, run_func)()

        # Expected value comes from ./test_data/mfile_data.json
        assert teardown.params["tau_e"] == pytest.approx(4.3196)

    @pytest.mark.parametrize("run_func", ["read", "readall"])
    def test_read_func_updates_bluemira_params_from_mfile(self, run_func):
        teardown = Teardown(self.default_pf, None, utils.READ_DIR)

        with file_exists(os.path.join(utils.READ_DIR, "MFILE.DAT"), self.IS_FILE_REF):
            getattr(teardown, run_func)()

        # Expected value comes from ./test_data/mfile_data.json
        assert teardown.params["tau_e"] == pytest.approx(4.3196)

    @pytest.mark.parametrize(
        "run_func, data_dir", [("runinput", utils.RUN_DIR), ("readall", utils.READ_DIR)]
    )
    def test_run_mode_updates_params_from_mfile_given_recv_False(
        self, run_func, data_dir
    ):
        # The two run modes in this test are expected to ignore the
        # 'recv = False' Parameter attribute
        teardown = Teardown(self.default_pf, utils.RUN_DIR, utils.READ_DIR)
        teardown.params.get_param("r_tf_in_centre").mapping[process.NAME].recv = False

        with file_exists(os.path.join(data_dir, "MFILE.DAT"), self.IS_FILE_REF):
            getattr(teardown, run_func)()

        # Expected value comes from ./test_data/mfile_data.json
        assert teardown.params["r_tf_in_centre"] == pytest.approx(2.6354)

    @pytest.mark.parametrize(
        "run_func, data_dir", [("run", utils.RUN_DIR), ("read", utils.READ_DIR)]
    )
    def test_run_mode_does_not_update_params_from_mfile_given_recv_False(
        self, run_func, data_dir
    ):
        teardown = Teardown(self.default_pf, utils.RUN_DIR, utils.READ_DIR)
        teardown.params.get_param("r_tf_in_centre").mapping[process.NAME].recv = False

        with file_exists(os.path.join(data_dir, "MFILE.DAT"), self.IS_FILE_REF):
            getattr(teardown, run_func)()

        # Non-expected value comes from ./test_data/mfile_data.json
        assert teardown.params["r_tf_in_centre"] != pytest.approx(2.6354)

    def test_mock_updates_params_from_mockPROCESS_json_file(self):
        teardown = Teardown(self.default_pf, None, utils.READ_DIR)

        teardown.mock()

        # Expected values come from ./test_data/read/mockPROCESS.json
        assert teardown.params.delta_95 == pytest.approx(0.33333)
        assert teardown.params.B_0 == pytest.approx(5.2742)
        assert teardown.params.r_fw_ib_in == pytest.approx(5.953487)
        assert teardown.params.v_burn == pytest.approx(0.032175)

    def test_mock_raises_CodesError_if_mock_file_does_not_exist(self):
        teardown = Teardown(self.default_pf, None, "./not/a/dir")

        with pytest.raises(CodesError):
            teardown.mock()

    @pytest.mark.parametrize("run_func", ["run", "runinput", "read", "readall"])
    def test_run_func_raises_CodesError_given_mfile_does_not_exist(self, run_func):
        teardown = Teardown(self.default_pf, "./not/a/dir", "./not/a/dir")

        assert not os.path.isfile("./not/a/dir/MFILE.DAT")
        with pytest.raises(CodesError) as codes_err:
            getattr(teardown, run_func)()
        assert "is not a file" in str(codes_err)

    @pytest.mark.parametrize("run_func", ["run", "runinput", "read", "readall"])
    @pytest.mark.parametrize("i_fail", [-1, 0, 2, 100])
    def test_run_func_raises_CodesError_given_process_infeasible_soln(
        self, run_func, i_fail
    ):
        self.mfile_mock.data["ifail"]["scan01"] = i_fail
        teardown = Teardown(self.default_pf, "./some/dir", "./some/dir")

        with file_exists("./some/dir/MFILE.DAT", self.IS_FILE_REF):
            with pytest.raises(CodesError) as codes_err:
                getattr(teardown, run_func)()
            assert "did not find a feasible solution" in str(codes_err)

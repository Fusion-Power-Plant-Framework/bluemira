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
import json
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

_INFEASIBLE_PROCESS_MFILE = (
    "# PROCESS #\n"
    "# Power Reactor Optimisation Code #\n"
    "# PROCESS #\n"
    "# Power Reactor Optimisation Code #\n"
    'PROCESS_version_number__________________________________________________ (procver)_____________________     "2.1.0   R"\n'
    'Date_of_run_____________________________________________________________ (date)________________________     ""\n'
    'Time_of_run_____________________________________________________________ (time)________________________     ""\n'
    'User____________________________________________________________________ (username)____________________     ""\n'
    'PROCESS_run_title_______________________________________________________ (runtitle)____________________     ""\n'
    'PROCESS_tag_number______________________________________________________ (tagno)_______________________     ""\n'
    'PROCESS_git_branch_name_________________________________________________ (branch_name)_________________     ""\n'
    'PROCESS_last_commit_message_____________________________________________ (commsg)______________________     ""\n'
    'Input_filename__________________________________________________________ (fileprefix)__________________     ""\n'
    "# Numerics #\n"
    "VMCON_error_flag________________________________________________________ (ifail)_______________________              {i_fail}\n"
)
FAKE_PROCESS_DICT = {  # Fake for the output of PROCESS's `get_dicts()`
    "DICT_DESCRIPTIONS": {
        "some_property": "its description",
    }
}


class FakeMFile:
    """
    A fake of PROCESS's MFile class.

    It replicates the :code:`.data` attribute with some PROCESS results
    data. This allows us to test the logic in our API without having
    PROCESS installed.
    """

    def __init__(self, filename):
        self.filename = filename
        with open(os.path.join(DATA_DIR, "mfile_data.json"), "r") as f:
            self.data = json.load(f)


class TestTeardown:

    MODULE_REF = "bluemira.codes.process._teardown"

    @classmethod
    def setup_class(cls):
        cls._mfile_patch = mock.patch(f"{cls.MODULE_REF}.MFile", new=FakeMFile)
        cls.mfile_mock = cls._mfile_patch.start()

        cls._process_dict_patch = mock.patch(
            f"{cls.MODULE_REF}.PROCESS_DICT", new=FAKE_PROCESS_DICT
        )
        cls.process_mock = cls._process_dict_patch.start()

    @classmethod
    def teardown_class(cls):
        cls._mfile_patch.stop()
        cls._process_dict_patch.stop()

    def setup_method(self):
        self.default_pf = Configuration()
        add_mapping(process.NAME, self.default_pf, process_mappings)

    @pytest.mark.parametrize("run_func", ["run", "runinput", "read", "readall"])
    def test_run_func_updates_bluemira_params_from_mfile(self, run_func):
        teardown = Teardown(self.default_pf, DATA_DIR)

        getattr(teardown, run_func)()

        # Expected value comes from ./test_data/MFILE.DAT
        assert teardown.params["tau_e"] == pytest.approx(4.3196)

    @pytest.mark.parametrize("run_func", ["runinput", "readall"])
    def test_run_mode_updates_params_from_mfile_given_recv_False(self, run_func):
        teardown = Teardown(self.default_pf, DATA_DIR)
        teardown.params.get_param("r_tf_in_centre").mapping[process.NAME].recv = False

        getattr(teardown, run_func)()

        # Expected value comes from ./test_data/MFILE.DAT
        assert teardown.params["r_tf_in_centre"] == pytest.approx(2.6354)

    @pytest.mark.parametrize("run_func", ["run", "read"])
    def test_run_mode_does_not_update_params_from_mfile_given_recv_False(self, run_func):
        teardown = Teardown(self.default_pf, DATA_DIR)
        teardown.params.get_param("r_tf_in_centre").mapping[process.NAME].recv = False

        getattr(teardown, run_func)()

        # Non-expected value comes from ./test_data/MFILE.DAT
        assert teardown.params["r_tf_in_centre"] != pytest.approx(2.6354)

    def test_mock_updates_params_from_mockPROCESS_json_file(self):
        teardown = Teardown(self.default_pf, DATA_DIR)

        teardown.mock()

        # Expected values come from ./test_data/mockPROCESS.json
        assert teardown.params.delta_95 == pytest.approx(0.33333)
        assert teardown.params.B_0 == pytest.approx(5.2742)
        assert teardown.params.r_fw_ib_in == pytest.approx(5.953487)
        assert teardown.params.v_burn == pytest.approx(0.032175)

    def test_mock_raises_CodesError_if_mock_file_does_not_exist(self):
        teardown = Teardown(self.default_pf, "./not/a/dir")

        with pytest.raises(CodesError):
            teardown.mock()

    @pytest.mark.parametrize("run_func", ["run", "runinput", "read", "readall"])
    def test_run_func_raises_CodesError_given_mfile_does_not_exist(self, run_func):
        teardown = Teardown(self.default_pf, "./not/a/dir")

        assert not os.path.isfile("./not/a/dir/MFILE.DAT")
        with pytest.raises(CodesError):
            getattr(teardown, run_func)()

    @pytest.mark.parametrize("run_func", ["run", "runinput", "read", "readall"])
    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data="")
    def test_run_func_raises_CodesError_given_mfile_empty(self, _, run_func):
        teardown = Teardown(self.default_pf, "./some/dir")

        with file_exists("./some/dir/MFILE.DAT", f"{self.MODULE_REF}.os.path.isfile"):
            with pytest.raises(CodesError):
                getattr(teardown, run_func)()

    @pytest.mark.parametrize("run_func", ["run", "runinput", "read", "readall"])
    @pytest.mark.parametrize("i_fail", [-1, 0, 2, 100])
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_run_func_raises_CodesError_given_process_infeasible_soln(
        self, open_mock, run_func, i_fail
    ):
        teardown = Teardown(self.default_pf, "./some/dir")
        open_mock.read_data = _INFEASIBLE_PROCESS_MFILE.format(i_fail=i_fail)

        with file_exists("./some/dir/MFILE.DAT", f"{self.MODULE_REF}.os.path.isfile"):
            with pytest.raises(CodesError):
                getattr(teardown, run_func)()

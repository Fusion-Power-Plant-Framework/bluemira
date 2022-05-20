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

from unittest import mock

import pytest

from bluemira.base.config import Configuration
from bluemira.codes import process
from bluemira.codes.error import CodesError
from bluemira.codes.process._setup import Setup
from bluemira.codes.process.mapping import mappings as process_mappings
from bluemira.codes.utilities import add_mapping
from tests._helpers import file_exists


class TestSetup:

    MODULE_REF = "bluemira.codes.process._setup"

    def setup_method(self):
        self.default_pf = Configuration()
        add_mapping(process.NAME, self.default_pf, process_mappings)

        self._writer_patch = mock.patch(f"{self.MODULE_REF}.InDat")
        self.writer_cls_mock = self._writer_patch.start()

    def teardown_method(self):
        self._writer_patch.stop()

    def test_run_adds_bluemira_params_to_InDat_writer(self):
        setup = Setup(self.default_pf, "")
        self.writer_cls_mock.return_value.data = {"x": 0}

        setup.run()

        writer = self.writer_cls_mock.return_value
        num_send_params = sum(1 for x in process_mappings.values() if x.send)
        assert writer.add_parameter.call_count == num_send_params
        # Expected value comes from default PROCESS template file
        assert mock.call("pnetelin", 500) in writer.add_parameter.call_args_list

    def test_run_adds_problem_setting_params_to_InDat_writer(self):
        problem_settings = {"input0": 0.0}
        setup = Setup(self.default_pf, "", problem_settings=problem_settings)
        self.writer_cls_mock.return_value.data = {"x": 0}

        setup.run()

        writer = self.writer_cls_mock.return_value
        assert writer.add_parameter.call_count > 0
        assert mock.call("input0", 0.0) in writer.add_parameter.call_args_list

    def test_run_inits_writer_with_template_file_if_file_exists(self):
        setup = Setup(self.default_pf, "", template_in_dat_path="template/path/in.dat")
        self.writer_cls_mock.return_value.data = {"input": 0.0}

        with file_exists("template/path/in.dat", f"{self.MODULE_REF}.os.path.isfile"):
            setup.run()

        self.writer_cls_mock.assert_called_once_with(filename="template/path/in.dat")

    @pytest.mark.parametrize("run_func", ["run", "runinput"])
    def test_run_raises_CodesError_given_no_data_in_template_file(self, run_func):
        setup = Setup(self.default_pf, "", template_in_dat_path="template/path/in.dat")
        self.writer_cls_mock.return_value.data = {}

        with pytest.raises(CodesError):
            getattr(setup, run_func)()

    def test_runinput_does_not_write_bluemira_outputs_to_in_dat(self):
        setup = Setup(self.default_pf, "")
        self.writer_cls_mock.return_value.data = {"x": 0}

        setup.runinput()

        self.writer_cls_mock.return_value.add_parameter.assert_not_called()

    @pytest.mark.parametrize("model_name, model_cls", Setup.MODELS.items())
    def test_models_are_converted_to_Model_classes_given_str_value(
        self, model_name, model_cls
    ):
        setup = Setup(self.default_pf, "")
        # Just use the first member of the enum's name for testing
        field_str = list(model_cls.__members__)[0]
        self.writer_cls_mock.return_value.data = {
            model_name: mock.Mock(get_value=field_str)
        }

        setup.run()

        writer = self.writer_cls_mock.return_value
        call_args = writer.add_parameter.call_args_list
        assert mock.call(model_name, model_cls[field_str].value) in call_args

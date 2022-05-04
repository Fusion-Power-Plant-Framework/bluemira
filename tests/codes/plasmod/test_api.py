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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.base.parameter import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.api import Outputs as PlasmodOutput
from bluemira.codes.plasmod.api import Run as PlasmodRun
from bluemira.codes.plasmod.api import Setup as PlasmodSetup
from bluemira.codes.plasmod.api import Solver as PlasmodSolver
from bluemira.codes.plasmod.api import Teardown as PlasmodTeardown


@pytest.fixture(scope="module")
def fake_parent():
    # Make parent class for solver tasks
    f_p = MagicMock()
    f_p.run_dir = "OUTDIR"
    return f_p


class TestSetup:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self, fake_parent):
        cls = type(self)
        cls.fake_parent = fake_parent
        cls.plasmod_setup = PlasmodSetup(cls.fake_parent, params=ParameterFrame())

    def test_update_inputs(self):
        # test bad variable
        self.fake_parent.problem_settings = {"new_fake_input": 5}
        self.plasmod_setup.update_inputs()
        with pytest.raises(KeyError):
            self.plasmod_setup.io_manager._options["new_fake_input"]
        assert self.plasmod_setup.io_manager._options["contrpovr"] == 0.0

        # test good variable
        self.fake_parent.problem_settings["contrpovr"] = 5
        self.plasmod_setup.update_inputs()
        assert self.plasmod_setup.io_manager._options["contrpovr"] == 5

        # test models
        self.fake_parent.problem_settings["isiccir"] = 0
        self.plasmod_setup.update_inputs()
        assert self.plasmod_setup.io_manager._options["isiccir"].value == 0

        self.fake_parent.problem_settings["isiccir"] = 2
        with pytest.raises(ValueError):
            self.plasmod_setup.update_inputs()


class TestRun:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self, fake_parent):
        cls = type(self)
        cls.fake_parent = fake_parent
        cls.fake_parent.setup_obj.input_file = "IN"
        cls.fake_parent.setup_obj.output_file = "OUT"
        cls.fake_parent.setup_obj.profiles_file = "PROF"
        cls.runner = PlasmodRun(cls.fake_parent)

    def test_run(self):
        with patch("bluemira.codes.plasmod.api.interface.Run._run_subprocess") as rsp:
            self.runner._run()

        assert [str(aa) for aa in rsp.call_args[0][0]] == [
            "plasmod",
            "OUTDIR/IN",
            "OUTDIR/OUT",
            "OUTDIR/PROF",
        ]


class TestTeardown:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self, fake_parent):
        cls = type(self)
        cls.fake_parent = fake_parent
        cls.plasmod_teardown = PlasmodTeardown(cls.fake_parent)

    def test_prepare_outputs(self):
        self.fake_parent._recv_mapping = {"plasmod": "bluemira"}
        self.plasmod_teardown.io_manager = MagicMock()
        self.plasmod_teardown.io_manager.plasmod = True
        with patch(
            "bluemira.codes.plasmod.api.interface.Teardown.prepare_outputs"
        ) as ppo:
            self.plasmod_teardown.prepare_outputs()

        assert ppo.call_args[0][0] == {"bluemira": True}
        assert ppo.call_args[1]["source"] == "PLASMOD"


class TestSolver:
    def test_get_raw_variables(self):
        fakeself = MagicMock()
        fakeself.teardown_obj.io_manager.test1 = 5
        fakeself.teardown_obj.io_manager.test2 = 10

        assert PlasmodSolver.get_raw_variables(fakeself, "test1") == 5
        assert PlasmodSolver.get_raw_variables(fakeself, ["test1", "test2"]) == [5, 10]

    def test_get_profile(self):

        fakeself = MagicMock()
        fakeself.teardown_obj.io_manager.x = 5
        fakeself.teardown_obj.io_manager.ne = 10

        assert PlasmodSolver.get_profile(fakeself, "x") == 5
        assert PlasmodSolver.get_profile(fakeself, "n_e") == 10

        with pytest.raises(ValueError):
            PlasmodSolver.get_profile(fakeself, "myprofile")


class TestOutputs:
    def setup(self):
        with open(
            Path(get_bluemira_path(), "codes/plasmod/PLASMOD_DEFAULT_OUT.json"), "r"
        ) as fh:
            self.output = json.load(fh)
        self.p_out = PlasmodOutput()

    def _read_output_files(self):
        with patch(
            "bluemira.codes.plasmod.api.Outputs.read_file", return_value=self.output
        ):
            self.p_out.read_output_files(None, None)

    def test_read_output_files(self):

        for i in [0, -1, -2, -3]:
            self.output["i_flag"] = i
            with pytest.raises(CodesError):
                self._read_output_files()

        # no error
        self.output["i_flag"] = 1
        self._read_output_files()

        assert self.p_out._options["i_flag"] == 1

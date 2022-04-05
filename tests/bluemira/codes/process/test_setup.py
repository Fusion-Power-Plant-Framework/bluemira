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

from unittest.mock import MagicMock

import pytest

from bluemira.base.parameter import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.process import setup as process_setup
from bluemira.codes.process.api import DEFAULT_INDAT
from bluemira.codes.process.api import ENABLED as PROCESS_ENABLED
from tests.bluemira.codes.process import OUTDIR


@pytest.mark.skipif(PROCESS_ENABLED is not True, reason="PROCESS install required")
class TestPROCESSInputWriter:
    """Load default PROCESS values"""

    writer = process_setup.PROCESSInputWriter()

    def test_change_var(self):
        self.writer.add_parameter("vgap2", 0.55)
        assert self.writer.data["vgap2"].get_value == 0.55


@pytest.mark.skipif(PROCESS_ENABLED is not True, reason="PROCESS install required")
class TestSetup:
    def setup(self):
        fake_parent = MagicMock()
        fake_parent._template_indat = DEFAULT_INDAT
        fake_parent.run_dir = OUTDIR
        self.setup_writer = process_setup.Setup(fake_parent, params=ParameterFrame())

    def test_raises_CodesError_on_bad_template(self):
        self.setup_writer.parent._template_indat = "a/file/IN.DAT"
        with pytest.raises(CodesError):
            self.setup_writer.write_indat()

    @pytest.mark.parametrize("problem_settings", ({}, {"iefrf": 6, "i_tf_sup": 2}))
    def test_write_input_file(self, problem_settings):
        """
        Tests that an input file is written properly (by proxy calls validate models)

        Eventually this should be moved out to PROCESS itself
        """
        self.setup_writer.parent.problem_settings = problem_settings
        self.setup_writer.write_indat()

        with open(OUTDIR + "/IN.DAT", "r") as fh:
            data = fh.readlines()

        for dat in data:
            for k, v in problem_settings.items():
                if dat.startswith(k):
                    assert str(v) in dat

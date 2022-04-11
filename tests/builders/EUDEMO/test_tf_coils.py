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

"""
Tests for EU-DEMO TF Coils build.
"""

import pytest

from bluemira.base.config import Configuration
from bluemira.base.error import BuilderError
from bluemira.base.file import get_bluemira_path
from bluemira.builders.EUDEMO.tf_coils import TFCoilsBuilder

DATA_PATH = get_bluemira_path("builders/EUDEMO/test_data", subfolder="tests")
"""
The path to data to be used in these tests.
"""


class TestTFCoils:
    """
    Test the EU-DEMO TF Coils build methods.
    """

    def setup_method(self):
        self.build_config = {
            "name": "TF Coils",
            "param_class": "TripleArc",
            "runmode": "read",
            "variables_map": {},
            "problem_class": "bluemira.builders.tf_coils::RippleConstrainedLengthGOP",
            "geom_path": DATA_PATH,
        }
        self.params = Configuration().to_dict()
        self.params["n_TF"] = 18

    def test_bad_geom_path(self):
        self.build_config["geom_path"] = "/a/bad/path/6b04dbec4a955560d59cf1e5e885b8de"
        with pytest.raises(BuilderError):
            TFCoilsBuilder(self.params, self.build_config)

    def test_read_from_dir(self):
        builder = TFCoilsBuilder(self.params, self.build_config)
        builder()

    def test_mock(self):
        self.build_config["runmode"] = "mock"
        builder = TFCoilsBuilder(self.params, self.build_config)
        builder()

    def test_run_no_separatrix(self):
        self.build_config["runmode"] = "run"
        with pytest.raises(BuilderError):
            TFCoilsBuilder(self.params, self.build_config)

    @pytest.mark.parametrize(
        "degree,n_children", [(360, 18), (20, 1), (21, 2), (270, 14)]
    )
    def test_build_xyz(self, degree, n_children):
        builder = TFCoilsBuilder(self.params, self.build_config)
        builder.mock()
        result = builder.build_xyz(degree=degree)
        assert len(result.children) == n_children

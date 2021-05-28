# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

from unittest.mock import patch

import pytest

from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.base.parameter import ParameterFrame, ParameterMapping
from BLUEPRINT.systems.config import Configuration

pw = pytest.importorskip(
    "BLUEPRINT.syscodes.PROCESSwrapper"
)  # Skip the tests if PROCESS not installed


FRAME = ParameterFrame(
    # fmt: off
    [
        ["a", None, 0, None, None, None],
        ["b", None, 1, None, None, None, None],
        ["c", None, 2, None, None, None, {"PROCESS": ParameterMapping("cp", False, False)}],
        ["d", None, 3, None, None, None, {"PROCESS": ParameterMapping("dp", False, True)}],
        ["e", None, 4, None, None, None, {"PROCESS": ParameterMapping("ep", True, False)}],
        ["f", None, 5, None, None, None, {"PROCESS": ParameterMapping("fp", True, True)}],
        ["g", None, 6, None, None, None, {"FAKE_CODE": ParameterMapping("gp", True, True)}]
    ]
    # fmt: on
)

PROCESS_OBS_VAR = {
    "ni": "ni wang",
    "ni wang": "ni peng",
    "garden": "shrubbery",
}


@pytest.mark.parametrize(
    "read_all,expected",
    [
        [
            True,
            {
                "cp": "c",
                "dp": "d",
                "ep": "e",
                "fp": "f",
            },
        ],
        [False, {"ep": "e", "fp": "f"}],
    ],
)
def test_PROCESS_read_mapping(read_all, expected):
    output = pw.get_PROCESS_read_mapping(FRAME, read_all)
    assert output == expected


@patch("BLUEPRINT.syscodes.PROCESSwrapper.OBS_VARS", PROCESS_OBS_VAR)
def test_update_obsolete_vars():
    str1 = pw.update_obsolete_vars("ni")
    str2 = pw.update_obsolete_vars("garden")
    assert str1 == "ni peng" and str2 == "shrubbery"


class TestPROCESSInputWriter:
    """Load default PROCESS values"""

    writer = pw.PROCESSInputWriter()

    def test_change_var(self):
        self.writer.add_parameter("vgap2", 0.55)
        assert self.writer.data["vgap2"].get_value == 0.55


class TestMFileReader:
    fp = get_BP_path("syscodes/test_data", subfolder="tests")

    @classmethod
    def setup_class(cls):
        mapping = {
            p[-1]["PROCESS"].name: p[0]
            for p in Configuration.params
            if len(p) == 7 and "PROCESS" in p[-1]
        }
        cls.bmfile = pw.BMFile(cls.fp, mapping)
        return cls

    def test_extraction(self):
        inp = [p[0] for p in Configuration.params if len(p) == 7 and "PROCESS" in p[-1]]
        out = self.bmfile.extract_outputs(inp)
        assert len(inp) == len(out)


class TestPROCESSRunner:
    P = pw.PROCESSRunner(ParameterFrame([]))  # default run

    def test_run(self):
        try:
            self.P.run()
        except Exception as e:
            self.fail(f"PROCESSRunner raised {e}...")

    def test_read_mapping(self):
        p = pw.PROCESSRunner(FRAME)
        assert p.read_parameter_mapping == {"ep": "e", "fp": "f"}

    @patch("BLUEPRINT.syscodes.PROCESSwrapper.PROCESSInputWriter")
    def test_write_mapping(self, mock_writer_cls):
        mock_writer_obj = mock_writer_cls.return_value
        p = pw.PROCESSRunner(FRAME)
        assert mock_writer_obj.add_parameter.call_count == 2
        mock_writer_obj.add_parameter.assert_any_call("dp", 3)
        mock_writer_obj.add_parameter.assert_any_call("fp", 5)


if __name__ == "__main__":
    pytest.main([__file__])

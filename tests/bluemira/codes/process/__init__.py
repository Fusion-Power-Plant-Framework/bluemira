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
Initialise PROCESS test module and set some useful constants.
"""

import os

from bluemira.base.config import SingleNull
from bluemira.base.config_schema import ConfigurationSchema
from bluemira.base.file import get_bluemira_root
from bluemira.base.parameter import Parameter, ParameterMapping

INDIR = os.path.join(
    get_bluemira_root(), "tests", "bluemira", "codes", "process", "test_data"
)
OUTDIR = os.path.join(
    get_bluemira_root(), "tests", "bluemira", "codes", "process", "test_generated_data"
)

FRAME_LIST = [
    # [var, name, value, unit, description, source{name, recv, send}]
    ["a", None, 0, "", None, None],
    ["b", None, 1, "", None, None, None],
    ["c", None, 2, "", None, None, {"PROCESS": ParameterMapping("cp", False, False)}],
    ["d", None, 3, "", None, None, {"PROCESS": ParameterMapping("dp", True, False)}],
    ["e", None, 4, "", None, None, {"PROCESS": ParameterMapping("ep", False, True)}],
    ["f", None, 5, "", None, None, {"PROCESS": ParameterMapping("fp", True, True)}],
    ["g", None, 6, "", None, None, {"FAKE_CODE": ParameterMapping("gp", True, True)}],
]


class PROCESSTestSchema(ConfigurationSchema):
    a: Parameter
    b: Parameter
    c: Parameter
    d: Parameter
    e: Parameter
    f: Parameter
    g: Parameter


class PROCESSTestConfiguration(SingleNull):
    params = SingleNull.params + FRAME_LIST

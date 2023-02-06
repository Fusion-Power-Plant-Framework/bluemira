# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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

from copy import deepcopy
from dataclasses import dataclass

from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.parameter_frame._parameter import ParamDictT
from bluemira.builders import cryostat
from bluemira.gen_params import (
    add_to_dict,
    create_parameterframe,
    def_param,
    get_param_classes,
)


def test_def_param():
    dp = deepcopy(ParamDictT.__annotations__)
    del dp["name"]
    assert def_param().keys() == dp.keys()


@dataclass
class PF(ParameterFrame):
    A: Parameter[float]


def test_add_to_dict():
    json_d = {}
    params_d = {}
    add_to_dict(PF, json_d, params_d)
    assert json_d == {
        "A": {
            "value": "float",
            "unit": "str",
            "source": "str",
            "description": "",
            "long_name": "",
        }
    }
    assert params_d == {"A": Parameter[float]}


def test_create_parameterframe():
    params_d = {"A": Parameter[float]}
    outstr = (
        "from dataclasses import dataclass\n\n"
        "from bluemira.base.parameterframe import Parameter, ParameterFrame\n\n\n"
    )
    dclassstr = "@dataclass\nclass {}Params(ParameterFrame):\n    A: Parameter[float]\n"
    assert create_parameterframe(params_d) == outstr + dclassstr.format("Reactor")
    assert create_parameterframe(
        params_d, name="MyParams", header=False
    ) == dclassstr.format("My")


def test_get_param_classes():
    keys = [
        "CryostatBuilder: CryostatBuilderParams",
        "CryostatDesigner: CryostatDesignerParams",
    ]
    out = get_param_classes(cryostat)
    for no, (k, v) in enumerate(out.items()):
        assert ParameterFrame in v.__bases__
        assert k == keys[no]

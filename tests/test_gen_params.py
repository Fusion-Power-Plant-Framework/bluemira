# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

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

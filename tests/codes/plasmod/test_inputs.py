# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import io

import pytest

from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.api._inputs import MODEL_MAP, PlasmodInputs


class TestPlasmodInputs:
    def test_write_writes_a_line_for_every_parameter(self):
        stream = io.StringIO()
        params = PlasmodInputs()

        params.write(stream)

        stream.seek(0)
        output = stream.read()
        lines = [line for line in output.split("\n") if line.strip()]
        params_dict = vars(params)
        assert len(lines) == len(params_dict)
        for param in params_dict:
            assert any(param in line for line in lines)

    @pytest.mark.parametrize(("model", "enum_cls"), MODEL_MAP.items())
    def test_model_is_converted_to_enum_on_init(self, model, enum_cls):
        # Just get the first member of the enum to test with
        enum_member = next(iter(enum_cls.__members__.values()))
        values = {model: enum_member.value}

        params = PlasmodInputs(**values)

        assert getattr(params, model) == getattr(enum_cls, enum_member.name)

    @pytest.mark.parametrize(("model", "enum_cls"), MODEL_MAP.items())
    def test_model_is_converted_to_enum_on_init_using_name(self, model, enum_cls):
        # Just get the first member enum name to test with
        enum_name = next(iter(enum_cls.__members__.keys()))
        values = {model: enum_name}

        params = PlasmodInputs(**values)

        assert getattr(params, model) == getattr(enum_cls, enum_name)

    @pytest.mark.parametrize("model", MODEL_MAP.keys())
    def test_CodesError_if_model_not_convertible_to_enum(self, model):
        values = {model: "NOT_AN_ENUM_VALUE"}

        with pytest.raises(CodesError):
            PlasmodInputs(**values)

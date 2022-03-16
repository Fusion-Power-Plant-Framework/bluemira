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

import copy
from typing import Type

import pytest
from pint import DimensionalityError, Unit

from bluemira.base.parameter import Parameter, ParameterFrame, ParameterMapping, _unitify
from BLUEPRINT.systems.baseclass import ReactorSystem


class Dummy(ReactorSystem):
    # Parameter DataFrame - automatically loaded as attributes: self.Var_name
    # Var_name    Name    Value     Unit        Description        Source
    inputs: dict
    default_params = [
        [
            "coolant",
            "Coolant",
            "Water",
            "dimensionless",
            "Divertor coolant type",
            "Common sense",
        ],
        ["T_in", "Coolant inlet T", 80, "°C", "Coolant inlet T", None],
        ["T_out", "Coolant outlet T", 120, "°C", "Coolant inlet T", None],
        ["P_in", "Coolant inlet P", 8, "MPa", "Coolant inlet P", None],
        ["dP", "Coolant pressure drop", 1, "MPa", "Coolant pressure drop", None],
        [
            "rm_cl",
            "RM clearance",
            0.02,
            "m",
            "Radial and poloidal clearance between in-vessel components",
            "Not so common sense",
        ],
    ]

    def __init__(self, inputs):
        self.inputs = inputs

        self._init_params(self.inputs)


class DummyPF(ReactorSystem):
    inputs: Type[ParameterFrame]
    default_params = [
        ["T_out", "Coolant outlet T", 120, "°C", "Coolant inlet T", None],
        ["dP", "Coolant pressure drop", 1, "MPa", "Coolant presdrop", None],
    ]

    def __init__(self):
        self._init_params(self.inputs)


class TestParameterMapping:
    def setup(self):
        self.pm = ParameterMapping("Name", send=True, recv=False)

    @pytest.mark.parametrize(
        "attr, value",
        zip(
            ["name", "mynewattr", "_frozen", "unit"],
            ["NewName", "Hello", ["custom", "list"], "MW"],
        ),
    )
    def test_no_keyvalue_change(self, attr, value):

        with pytest.raises(KeyError):
            setattr(self.pm, attr, value)

    def test_value_change(self):

        for var in ["send", "recv"]:
            with pytest.raises(ValueError):
                setattr(self.pm, var, "A string")

        assert self.pm.send
        assert not self.pm.recv
        self.pm.send = False
        self.pm.recv = True
        assert not self.pm.send
        assert self.pm.recv

    def test_tofrom_dict(self):
        assert self.pm == ParameterMapping.from_dict(self.pm.to_dict())


class TestUnitify:
    def test_convert_string_to_unit(self):
        uf = _unitify("MW")
        pu = Unit("MW")

        assert isinstance(uf, Unit)
        assert isinstance(pu, Unit)
        assert uf == pu

        mperc = _unitify("M%")
        mperc_u = Unit("M%")
        assert isinstance(mperc, Unit)
        assert isinstance(mperc_u, Unit)
        assert mperc == mperc_u

        with pytest.raises(TypeError):
            _unitify(None)


class TestParameter:
    p = Parameter(
        "r_0",
        "Major radius",
        9,
        "m",
        "marjogrgrbg",
        "Input",
        {"PROCESS": ParameterMapping("rmajor", False, True, "m")},
    )
    p_str = (
        "Major radius [m]: r_0 = 9 (marjogrgrbg)\n"
        "    {'PROCESS': {'name': 'rmajor', 'recv': False, 'send': True, 'unit': 'm'}}"
    )
    g = Parameter(
        "B_0",
        "Toroidal field at R_0",
        5.7,
        "T",
        "Toroidal field at the centre of the plasma",
        "Input",
    )
    g_str = "Toroidal field at R_0 [T]: B_0 = 5.7 (Toroidal field at the centre of the plasma)"

    def test_p(self, capsys):
        # make a copy of p
        _p = Parameter(**self.p.to_dict())

        # Assert printing is correct
        print(_p, end="")
        out, err = capsys.readouterr()
        assert out == self.p_str

        # Assert no type change on value change
        assert isinstance(_p, Parameter)
        _p.value = 4
        assert isinstance(_p, Parameter)
        _p.source = "here"

        # Assert value as expected
        assert _p == 4
        assert _p.source == "here"
        assert _p == _p.value

        # Assert attribute access
        assert _p.description == "marjogrgrbg"

        # Assert equal to as well as not equal to works
        assert not _p == 9
        _p.value = 9
        assert _p == 9

        # Assert source and value history
        assert _p.value_history == [9, 4, 9]
        assert _p.source_history == ["Input", "here", None]

        # Maths
        _p += 1
        _p -= 1
        _p *= 1
        _p /= 1
        assert isinstance(_p, Parameter)

        a = _p + 1
        b = _p - 1
        c = _p * 1
        d = _p / 1
        assert all([a == 10, b == 8, c == 9, d == 9])

        assert self.p + _p == 18

    @pytest.mark.parametrize(
        "param, ignore_var", ((p, True), (p, False), (g, True), (g, False))
    )
    def test_to_from_dict(self, param, ignore_var):
        p_dict = param.to_dict(ignore_var=ignore_var)
        mapping = {
            "var": param.var,
            "name": param.name,
            "value": param.value,
            "unit": param.unit,
            "description": param.description,
            "source": param.source,
            "mapping": param.mapping,
        }
        if ignore_var:
            mapping.pop("var")
        assert p_dict == mapping
        if ignore_var:
            p_dict["var"] = param.var
        p_new = Parameter(**p_dict)
        if ignore_var:
            p_dict.pop("var")
        assert id(self.p) != id(p_new)
        assert p_dict.keys() == p_new.to_dict(ignore_var=ignore_var).keys()
        assert all(
            [
                p_dict[key] == new_val
                for (key, new_val) in p_new.to_dict(ignore_var=ignore_var).items()
            ]
        )

    @pytest.mark.parametrize("param", (p, g))
    def test_to_from_list(self, param):
        p_list = param.to_list()
        assert p_list == [
            param.var,
            param.name,
            param.value,
            param.unit,
            param.description,
            param.source,
            param.mapping,
        ]

        p_new = Parameter(*p_list)
        assert id(self.p) != id(p_new)
        assert all(
            [
                old_val == new_val
                for (old_val, new_val) in zip(param.to_list(), p_new.to_list())
            ]
        )

    @pytest.mark.parametrize("param, expected", [(p, p_str), (g, g_str)])
    def test_to_str(self, param, expected):
        assert f"{param}" == expected

    def test_nounit_error(self):
        with pytest.raises(TypeError):
            Parameter("p", "param", 1.0)

    def test_source_warning(self, caplog):
        warning_str = "The source of the value of p not consistently known"
        p = Parameter("p", "param", 1.0, "m")

        # check we get a warning if source hasn't been defined
        p.value = 5
        assert len(caplog.messages) == 1
        out = caplog.messages[0]
        assert warning_str in out

    def test_source_no_warning(self, caplog):
        p = Parameter("p", "param", 1.0, "m")
        p.source = "Input"

        # check we don't get a warning if source has been defined
        p.value = 5
        assert len(caplog.messages) == 0

        p.source = "new"
        p.value = 10

        assert len(caplog.messages) == 0

    def test_source_warning_init_update(self, caplog):
        warning_str = "The source of the value of p not consistently known"
        p = Parameter("p", "param", 1.0, "m", source="Input")

        # check we don't get a warning if source has been defined
        p.value = 5
        assert len(caplog.messages) == 0

        p.value = 10
        assert len(caplog.messages) == 1
        out = caplog.messages[0]
        assert warning_str in out

    def test_source_inplace_warning(self, caplog):
        warning_str = "The source of the value of p not consistently known"
        p = Parameter("p", "param", 1.0, "m", source="Input")
        p += 5

        # source should now be reset to False, so check we get a warning if we update
        p += 5
        assert len(caplog.messages) == 1
        out = caplog.messages[0]
        assert warning_str in out

    def test_source_inplace_no_warning(self, caplog):
        p = Parameter("p", "param", 1.0, "m", source="Input")
        p += 5
        p.source = "new"

        # check we don't get a warning if source has been redefined after value update
        p += 5
        assert len(caplog.messages) == 0

    def test_mappingtype_enforcement(self):

        p = Parameter("p", "param", 1.0, "m", source="Input")

        for val in ("string", {"A": "dict"}):
            with pytest.raises(TypeError):
                p.mapping = val

        mapping = ParameterMapping("param_name", send=True, recv=True, unit="T")
        p.mapping = {"MYCODE": mapping}

        assert p.mapping["MYCODE"] == mapping

        p.mapping = {"MYOTHERCODE": mapping}

        assert p.mapping["MYCODE"] == mapping
        assert p.mapping["MYOTHERCODE"] == mapping

        del p.mapping["MYOTHERCODE"]

        assert p.mapping.keys() == set(["MYCODE"])


def _converted_helper(test_pm, source):
    """
    Dunno why this only works without static or
    class decorator....dont question it
    """
    pm = copy.deepcopy(test_pm).to_dict()
    pm["source"] = source
    return pm


def _gen_unit_conversions():
    answers = 0.05, 5
    units = "cm", "m"
    conversion_str = ": Units converted from centimetre to metre", ""

    names = "tuple", "list", "dict", "param", "param_from_dict", "list_param"
    parameterisations = []
    for ans, unit, conv_str in zip(answers, units, conversion_str):
        parameterisations += [[(answers[1], names[0], unit), ans, names[0] + conv_str]]
        parameterisations += [[[answers[1], names[1], unit], ans, names[1] + conv_str]]
        parameterisations += [
            [
                {"value": answers[1], "source": names[2], "unit": unit},
                ans,
                names[2] + conv_str,
            ]
        ]

        test_pm = Parameter(
            **{"var": "R_0", "value": answers[1], "source": names[3], "unit": unit}
        )

        parameterisations += [[test_pm, ans, names[3] + conv_str]]
        parameterisations += [
            [
                _converted_helper(test_pm, names[4]),
                ans,
                names[4] + conv_str,
            ]
        ]
        parameterisations += [
            [
                list(_converted_helper(test_pm, names[5]).values()),
                ans,
                names[5] + conv_str,
            ]
        ]
    return parameterisations


def _gen_record_list():
    rec_list = [
        [
            "R_0",
            "Fake Radius",
            10,
            "m",
            "Fake Description.",
            "Old Source",
        ],
        [
            "B_0",
            "Fake Field",
            10,
            "T",
            "Fake Description.",
            "Old Source",
        ],
        ["n_TF", "Fake Num", 16, "dimensionless", "Fake Description.", "Old Source"],
    ]
    rec_list_param = [Parameter(*p) for p in rec_list]
    rec_dict_param = {p[0]: Parameter(*p) for p in rec_list}
    return rec_list, rec_list_param, rec_dict_param


class TestParameterFrame:
    default_params = [
        ["Name", "Reactor name", "Cambridge", "dimensionless", None, "Input"],
        ["plasma_type", "Type of plasma", "SN", "dimensionless", None, "Input"],
        ["op_mode", "Mode of operation", "Pulsed", "dimensionless", None, "Input"],
        ["blanket_type", "Blanket type", "HCPB", "dimensionless", None, "Input"],
        [
            "n_TF",
            "Number of TF coils",
            16,
            "dimensionless",
            None,
            "Input",
            {"PROCESS": ParameterMapping("n_tf", True, False)},
        ],
        ["n_PF", "Number of PF coils", 6, "dimensionless", None, "Input"],
        ["n_CS", "Number of CS coil divisions", 5, "dimensionless", None, "Input"],
        [
            "TF_ripple_limit",
            "TF coil ripple limit",
            0.6,
            "%",
            None,
            "Input",
            {"PROCESS": ParameterMapping("ripmax", True, False)},
        ],
        [
            "A",
            "Plasma aspect ratio",
            3.1,
            "dimensionless",
            None,
            "Input",
            {"PROCESS": ParameterMapping("aspect", True, False)},
        ],
        [
            "R_0",
            "Major radius",
            9,
            "m",
            None,
            "Input",
            {"PROCESS": ParameterMapping("rmajor", False, True)},
        ],
        [
            "B_0",
            "Toroidal field at R_0",
            6,
            "T",
            None,
            "Input",
            {"PROCESS": ParameterMapping("bt", False, True)},
        ],
    ]

    def setup(self):
        self.params = ParameterFrame(self.default_params)

    def test_keys(self):
        assert list(self.params.keys()) == [p[0] for p in self.default_params]

    def test_copy(self):
        params_copy = self.params.copy()
        assert id(params_copy) != id(self.params)
        assert params_copy == self.params

    def test_neq(self):
        params_copy = self.params.copy()
        assert params_copy == self.params
        params_copy.A *= 0.5
        assert params_copy != self.params

    def test_get(self):
        assert isinstance(self.params.get("n_TF"), int)
        out = self.params.get("n_TF_", 4)
        assert isinstance(out, int)
        assert out == 4

    def test_get_param(self):
        assert isinstance(self.params.get_param("n_TF"), Parameter)
        assert isinstance(self.params.get_param(["n_TF", "n_PF"]), ParameterFrame)

    def test_get_failure(self):
        assert self.params.n_TF == 16
        with pytest.raises(AttributeError):
            self.params.get_param("notathing")

    def test_add(self):
        p = Parameter("R_0", "Major radius", 9.5, "m", "marjogrgrbg", None)
        g = Parameter("T_0", "something new", 5.7, "T", "bla", None)
        self.params.add_parameter(p)
        assert self.params.R_0 == 9.5
        self.params.add_parameter(g)
        assert self.params.T_0 == 5.7

    def test_add_multiple_parameters(self):
        p = Parameter("R_0", "Major radius", 8.9, "m", "marjogrgrbg", None)
        g = Parameter("B_0", "Field", 5.5, "T", "bla", None)
        self.params.add_parameters([p, g])
        assert self.params["R_0"] == 8.9
        assert self.params["B_0"] == 5.5

    def test_add_non_matching_keys(self):
        """
        Test that a parameter that doesn't exist isn't added.
        """
        params_copy = self.params.copy()
        p = {"P_0": 8.9}
        params_copy.update_kw_parameters(p)
        assert params_copy == self.params

    def test_add_parameter_value(self):
        """
        Test that when a parameter is added, only the value is used.
        """
        params_copy = self.params.copy()
        p = {
            "R_0": Parameter(
                "R_0", "Really bad parameter", 10, "m", "Don't change me", None
            )
        }
        params_copy.update_kw_parameters(p)
        assert params_copy.R_0 == 10
        assert params_copy.get_param("R_0").name == self.params.get_param("R_0").name
        assert params_copy.get_param("R_0").unit == self.params.get_param("R_0").unit
        # TODO: Remove - Users should be able to define their own description IMO
        # Perhaps we should prevent a change of description if they have set it though
        # assert (
        #     params_copy.get_param("R_0").description
        #     == self.params.get_param("R_0").description
        # )
        # The source should be updated because it has changed
        # assert params_copy.get("R_0").source == self.params.get("R_0").source

    @pytest.mark.parametrize(
        "record_list",
        _gen_record_list(),
        ids=["list_format", "params_format", "params_kw"],
    )
    def test_add_parameters_source(self, record_list):
        """
        Test that a source passed to directly to add_parameters is updated correctly in
        each parameter in record_list (where record_list is a list of lists,
        a list of parameters and a dict of parameters).
        """
        params_copy = self.params.copy()
        new_source = "New Source"
        params_copy.add_parameters(record_list, source=new_source)
        assert params_copy.get_param("R_0").source == new_source
        assert params_copy.get_param("B_0").source == new_source
        assert params_copy.get_param("n_TF").source == new_source

    def test_complex_update_kw(self):
        params_copy = self.params.copy()

        params = {
            "R_0": 9.0,
            "n_TF": {
                "value": 16,
                "description": "Number of TF coils needed for this study",
            },
            "A": {
                "value": 3.1,
                "unit": "dimensionless",
                "description": "some description",
            },
        }
        params_copy.update_kw_parameters(params)

        assert params_copy.R_0.value == 9.0
        assert params_copy.n_TF == 16
        assert params_copy.A.description == "some description"

    def test_bad_index(self):
        """
        Test that the ParameterFrame fails gracefully on a bad index.
        """
        with pytest.raises(KeyError) as ex_info:
            self.params["BadIndex"]
        assert str(ex_info.value) == "'Var name BadIndex not present in ParameterFrame'"

    def test_undefined_attr(self):
        """
        Test that the ParameterFrame fails gracefully on a bad attribute.
        """
        with pytest.raises(ValueError) as ex_info:
            self.params.bad_attr = 5
        assert str(ex_info.value) == "Attribute bad_attr not defined in ParameterFrame."

    def test_mismatched_attr(self):
        """
        Test that the ParameterFrame fails gracefully on a mismatched attribute.

        We assume that if the units match its ok and copy over the value and source
        """
        with pytest.raises(DimensionalityError) as ex_info:
            self.params.R_0 = self.params.get_param("B_0")
        assert (
            str(ex_info.value)
            == "Cannot convert from 'tesla' ([mass] / [current] / [time] ** 2) to 'meter' ([length])"
        )

    def test_mismatched_attr_name(self):
        with pytest.raises(ValueError):
            ParameterFrame({"B_O": self.params.R_0})

        with pytest.raises(ValueError):
            ParameterFrame().add_parameters({"B_O": self.params.R_0})

    def test_starstar(self):
        def starstar(pf):
            t = {}
            for k, v in pf.items():
                t[k] = v
            return t

        td = starstar(self.params)
        # Need to list otherwise the object is a dict_items object
        assert list(td.items()) == self.params.items()

    def test_to_from_verbose_dict(self):
        d = self.params.to_dict(verbose=True)
        assert all(
            d[key]
            == {
                "name": self.params.get_param(key).name,
                "value": self.params.get_param(key).value,
                "unit": self.params.get_param(key).unit,
                "description": self.params.get_param(key).description,
                "source": self.params.get_param(key).source,
                "mapping": self.params.get_param(key).mapping,
            }
            for key in self.params.keys()
        )
        new_params = ParameterFrame.from_dict(d)
        assert id(self.params) != id(new_params)
        assert self.params == new_params

    def test_set_values_from_dict(self):
        d = self.params.to_dict()
        assert d.keys() == self.params.keys()
        assert all([d[key]["value"] == self.params[key] for key in self.params.keys()])
        params_copy = self.params.copy()
        params_copy.R_0 = 60.0
        params_copy.B_0 = 0.0
        params_copy.update_kw_parameters(d)
        assert self.params == params_copy

    def test_to_from_list(self):
        test_list = self.params.to_list()
        assert all(
            [
                all(
                    [
                        param_list[0] == self.params.get_param(param_list[0]).var,
                        param_list[1] == self.params.get_param(param_list[0]).name,
                        param_list[2] == self.params.get_param(param_list[0]).value,
                        param_list[3] == self.params.get_param(param_list[0]).unit,
                        param_list[4]
                        == self.params.get_param(param_list[0]).description,
                        param_list[5] == self.params.get_param(param_list[0]).source,
                    ]
                )
                for param_list in test_list
            ]
        )
        new_params = ParameterFrame.from_list(test_list)
        assert id(self.params) != id(new_params)
        assert self.params == new_params

    def test_to_from_verbose_json(self):
        j = self.params.to_json(verbose=True, return_output=True)
        new_params = ParameterFrame.from_json(j)
        assert id(self.params) != id(new_params)
        assert self.params == new_params

    def test_to_from_verbose_json_file(self, tmpdir):
        json_path = tmpdir.join("verbose.json")
        j = self.params.to_json(output_path=json_path, verbose=True, return_output=True)
        new_params = ParameterFrame.from_json(json_path)
        assert id(self.params) != id(new_params)
        assert self.params == new_params

    def test_set_values_from_json(self):
        j = self.params.to_json(return_output=True)
        params_copy = self.params.copy()
        params_copy.R_0 = 60.0
        params_copy.B_0 = 0.0
        params_copy.set_values_from_json(j)
        assert params_copy == self.params

        # Check we can set a new source from the json update
        params_copy.set_values_from_json(j, source="New Values")
        assert params_copy != self.params
        assert params_copy.items() == self.params.items()
        for key in self.params.keys():
            assert params_copy.get_param(key).source == "New Values"

    def test_set_values_from_json_file(self, tmpdir):
        json_path = tmpdir.join("concise.json")
        j = self.params.to_json(output_path=json_path, return_output=True)
        params_copy = self.params.copy()
        params_copy.R_0 = 60.0
        params_copy.B_0 = 0.0
        params_copy.set_values_from_json(json_path)
        assert self.params == params_copy

        # Check we can set a new source from the json update
        params_copy.set_values_from_json(json_path, source="New Values")
        assert params_copy != self.params
        assert params_copy.items() == self.params.items()
        for key in self.params.keys():
            assert params_copy.get_param(key).source == "New Values"

    def test_to_from_verbose_json_file_validation(self, tmpdir):
        json_path = tmpdir.join("concise_invalid.json")
        j = self.params.to_json(output_path=json_path, return_output=True)
        with pytest.raises(ValueError) as ex_info:
            new_params = ParameterFrame.from_json(json_path)
        assert (
            str(ex_info.value)
            == "Creating a ParameterFrame using from_json requires a verbose json format."
        )

    def test_set_values_from_json_file_validation(self, tmpdir):
        json_path = tmpdir.join("verbose_invalid.json")
        j = self.params.to_json(output_path=json_path, verbose=True, return_output=True)
        with pytest.raises(ValueError) as ex_info:
            self.params.set_values_from_json(json_path)
        assert (
            str(ex_info.value)
            == "Setting the values on a ParameterFrame using set_values_from_json requires a concise json format."
        )

    def test_parameter_source_update(self):
        params_copy = self.params.copy()

        # Updating the value should create a new history record with a False source.
        params_copy.R_0 = 6.5
        assert len(params_copy.R_0.history()) == 2
        assert params_copy.R_0 == 6.5
        assert params_copy.R_0.source is None

        # Updating the source shouldn't create a new history record.
        params_copy.R_0.source = "Updated"
        assert len(params_copy.R_0.history()) == 2
        assert params_copy.R_0 == 6.5
        assert params_copy.R_0.source == "Updated"

        # Source can be set to None.
        params_copy.R_0 = 6.8
        params_copy.R_0.source = None
        assert len(params_copy.R_0.history()) == 3
        assert params_copy.R_0 == 6.8
        assert params_copy.R_0.source is None

    @pytest.mark.parametrize(
        "value, source",
        zip([(6.8, "New Value"), {"value": 6.8, "source": None}], ["New Value", None]),
        ids=["tuple", "dict"],
    )
    def test_parameter_source(self, value, source):
        params_copy = self.params.copy()

        params_copy.R_0 = value
        assert len(params_copy.R_0.history()) == 2
        assert params_copy.R_0 == 6.8
        assert params_copy.R_0.source == source

    @pytest.mark.parametrize(
        "var, value, source",
        _gen_unit_conversions(),
        ids=["tuple", "list", "dict", "param", "param_from_dict", "list_param"]
        + [
            i + "noconvert"
            for i in ["tuple", "list", "dict", "param", "param_from_dict", "list_param"]
        ],
    )
    def test_converted_unit(self, var, value, source):
        params_copy = self.params.copy()
        params_copy.R_0 = var
        assert params_copy.R_0.value == value
        assert params_copy.R_0.source == source

    def test_bad_unit_conversion(self):
        params_copy = self.params.copy()

        p = {
            "R_0": Parameter(
                "R_0", "Really bad parameter", 10, "m/N/s", "Don't change me", None
            )
        }
        with pytest.raises(DimensionalityError):
            params_copy.update_kw_parameters(p)

    def test_set_parameter(self):
        params_copy = self.params.copy()
        params_copy.set_parameter("n_CS", 10, "dimensionless", "hello")

        assert params_copy.n_CS.value == 10
        assert params_copy.n_CS.source == "hello"

        params_copy.set_parameter(
            "n_CS",
            Parameter(var="n_CS", value=15, unit="dimensionless", source="hello2"),
        )

        assert params_copy.n_CS.value == 15
        assert params_copy.n_CS.source == "hello2"

        # No unit provided
        params_copy.set_parameter("n_CS", 10, source="hello")
        assert params_copy.n_CS.value == 10
        assert params_copy.n_CS.source == "hello"


class TestReactorSystem:

    # def setup(self):
    #     self.params = ParameterFrame(with_defaults=True)

    # @classmethod
    # def teardown_class(cls):
    #     ParameterFrame._clean()
    DIV = Dummy({"T_in": {"value": 88, "unit": "celsius"}})

    def test_input_handling(self):
        assert self.DIV.params.T_in == 88

    def test_add_parameter(self):
        self.DIV.add_parameter("R_0", "Major radius", 8.8, "m", None, None)

        assert self.DIV.params.R_0 == 8.8
        assert self.DIV.params["R_0"] == 8.8

        self.DIV.add_parameter(
            "n_TF", "Number of TF coils", 17, "dimensionless", None, "Input"
        )
        assert self.DIV.params.n_TF == 17
        assert self.DIV.params["n_TF"] == 17

    def test_add_parameters(self):
        p = [
            ["n_TF", "Number of TF coils", 200, "dimensionless", None, "Input"],
            ["n_PF", "Number of PF coils", 6, "dimensionless", None, "Input"],
        ]
        self.DIV.add_parameters(p)
        assert self.DIV.params.n_TF == 200

    def test_use_pf_to_call_rs(self):
        self.DIV.add_parameter(
            "dP", "Coolant pressure drop", 1.99, "MPa", "Coolant presdrop", None
        )
        assert self.DIV.params.dP != 1
        assert self.DIV.params.dP == 1.99

        assert not hasattr(self.DIV, "n_TF")

    def test_use_interface_to_call_rs(self):
        ts = Dummy({"dP": 10})

        assert ts.params.dP == 10


if __name__ == "__main__":
    pytest.main([__file__])

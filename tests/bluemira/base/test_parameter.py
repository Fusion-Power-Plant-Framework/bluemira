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

import pytest
from typing import Type

from BLUEPRINT.systems.baseclass import ReactorSystem

from bluemira.base.parameter import Parameter, ParameterFrame, ParameterMapping


class Dummy(ReactorSystem):
    # Parameter DataFrame - automatically loaded as attributes: self.Var_name
    # Var_name    Name    Value     Unit        Description        Source
    inputs: dict
    default_params = [
        ["coolant", "Coolant", "Water", None, "Divertor coolant type", "Common sense"],
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


class TestParameter:
    p = Parameter(
        "r_0",
        "Major radius",
        9,
        "m",
        "marjogrgrbg",
        "Input",
        {"PROCESS": ParameterMapping("rmajor", False, True)},
    )
    p_str = "r_0 = 9 m (Major radius) : marjogrgrbg {'PROCESS': {'name': 'rmajor', 'read': False, 'write': True}}"
    g = Parameter(
        "B_0",
        "Toroidal field at R_0",
        5.7,
        "T",
        "Toroidal field at the centre of the plasma",
        "Input",
    )
    g_str = "B_0 = 5.7 T (Toroidal field at R_0) : Toroidal field at the centre of the plasma"

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
        assert str(param) == expected

    def test_source_warning(self, caplog):
        warning_str = "The source of the value of p not consistently known"
        p = Parameter("p", "param", 1.0)

        # check we get a warning if source hasn't been defined
        p.value = 5
        assert len(caplog.messages) == 1
        out = caplog.messages[0]
        assert warning_str in out

    def test_source_no_warning(self, caplog):
        p = Parameter("p", "param", 1.0)
        p.source = "Input"

        # check we don't get a warning if source has been defined
        p.value = 5
        assert len(caplog.messages) == 0

        p.source = "new"
        p.value = 10

        assert len(caplog.messages) == 0

    def test_source_warning_init_update(self, caplog):
        warning_str = "The source of the value of p not consistently known"
        p = Parameter("p", "param", 1.0, source="Input")

        # check we don't get a warning if source has been defined
        p.value = 5
        assert len(caplog.messages) == 0

        p.value = 10
        assert len(caplog.messages) == 1
        out = caplog.messages[0]
        assert warning_str in out

    def test_source_inplace_warning(self, caplog):
        warning_str = "The source of the value of p not consistently known"
        p = Parameter("p", "param", 1.0, source="Input")
        p += 5

        # source should now be reset to False, so check we get a warning if we update
        p += 5
        assert len(caplog.messages) == 1
        out = caplog.messages[0]
        assert warning_str in out

    def test_source_inplace_no_warning(self, caplog):
        p = Parameter("p", "param", 1.0, source="Input")
        p += 5
        p.source = "new"

        # check we don't get a warning if source has been redefined after value update
        p += 5
        assert len(caplog.messages) == 0


class TestParameterFrame:
    default_params = [
        ["Name", "Reactor name", "Cambridge", "N/A", None, "Input"],
        ["plasma_type", "Type of plasma", "SN", "N/A", None, "Input"],
        ["op_mode", "Mode of operation", "Pulsed", "N/A", None, "Input"],
        ["blanket_type", "Blanket type", "HCPB", "N/A", None, "Input"],
        [
            "n_TF",
            "Number of TF coils",
            16,
            "N/A",
            None,
            "Input",
            {"PROCESS": ParameterMapping("n_tf", True, False)},
        ],
        ["n_PF", "Number of PF coils", 6, "N/A", None, "Input"],
        ["n_CS", "Number of CS coil divisions", 5, "N/A", None, "Input"],
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
            "N/A",
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

    # ParameterFrame.set_default_parameters(default_params)

    def setup(self):
        self.params = ParameterFrame(self.default_params)

    # @classmethod
    # def teardown_class(cls):
    #     ParameterFrame._clean()

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
                "R_0", "Really bad parameter", 10, "m/N/s", "Don't change me", None
            )
        }
        params_copy.update_kw_parameters(p)
        assert params_copy.R_0 == 10
        assert params_copy.get_param("R_0").name == self.params.get_param("R_0").name
        assert params_copy.get_param("R_0").unit == self.params.get_param("R_0").unit
        assert (
            params_copy.get_param("R_0").description
            == self.params.get_param("R_0").description
        )
        # The source should be updated because it has changed
        # assert params_copy.get("R_0").source == self.params.get("R_0").source

    def test_add_parameters_source_using_class_format(self):
        """
        Test that a source passed to directly to add_parameters is updated correctly in
        each parameter in record_list (where record_list is a list of parameter objects).
        """
        params_copy = self.params.copy()
        record_list = [
            Parameter(
                "R_0", "Fake Radius", 10, "Fake Units", "Fake Description.", "Old Source"
            ),
            Parameter(
                "B_0", "Fake Field", 10, "Fake Units", "Fake Description.", "Old Source"
            ),
            Parameter(
                "n_TF", "Fake Num", 16, "Fake Units", "Fake Description.", "Old Source"
            ),
        ]
        new_source = "New Source"
        params_copy.add_parameters(record_list, source=new_source)
        assert params_copy.get_param("R_0").source == new_source
        assert params_copy.get_param("B_0").source == new_source
        assert params_copy.get_param("n_TF").source == new_source

    def test_add_parameters_source_using_list_format(self):
        """
        Test that a source passed to directly to add_parameters is updated correctly in
        each parameter in record_list (where record_list is a list of lists).
        """
        params_copy = self.params.copy()
        record_list = [
            ["R_0", "Fake Radius", 10, "Fake Units", "Fake Description.", "Old Source"],
            ["B_0", "Fake Field", 10, "Fake Units", "Fake Description.", "Old Source"],
            ["n_TF", "Fake Num", 16, "Fake Units", "Fake Description.", "Old Source"],
        ]
        new_source = "New Source"
        params_copy.add_parameters(record_list, source=new_source)
        assert params_copy.get_param("R_0").source == new_source
        assert params_copy.get_param("B_0").source == new_source
        assert params_copy.get_param("n_TF").source == new_source

    def test_add_parameters_source_via_update_kw_parameters(self):
        """
        Test that a source passed to directly to add_parameters is updated correctly in
        each parameter in record_list (where record_list is a dict of Parameter objects).
        Note in this case record_list and source are passed to update_kw_parameters.
        """
        params_copy = self.params.copy()
        record_list = {
            "R_0": Parameter(
                "R_0", "Fake Radius", 10, "Fake Units", "Fake Description.", "Old Source"
            ),
            "B_0": Parameter(
                "B_0", "Fake Field", 10, "Fake Units", "Fake Description.", "Old Source"
            ),
            "n_TF": Parameter(
                "n_TF", "Fake Num", 16, "Fake Units", "Fake Description.", "Old Source"
            ),
        }
        new_source = "New Source"
        params_copy.add_parameters(record_list, source=new_source)
        assert params_copy.get_param("R_0").source == new_source
        assert params_copy.get_param("B_0").source == new_source
        assert params_copy.get_param("n_TF").source == new_source

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
        """
        with pytest.raises(ValueError) as ex_info:
            self.params.R_0 = self.params.get_param("B_0")
        assert (
            str(ex_info.value)
            == "Mismatch between parameter var B_0 and attribute to be set R_0."
        )

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
        j = self.params.to_json(verbose=True)
        new_params = ParameterFrame.from_json(j)
        assert id(self.params) != id(new_params)
        assert self.params == new_params

    def test_to_from_verbose_json_file(self, tmpdir):
        json_path = tmpdir.join("verbose.json")
        j = self.params.to_json(output_path=json_path, verbose=True)
        new_params = ParameterFrame.from_json(json_path)
        assert id(self.params) != id(new_params)
        assert self.params == new_params

    def test_set_values_from_json(self):
        j = self.params.to_json()
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
        j = self.params.to_json(output_path=json_path)
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
        j = self.params.to_json(output_path=json_path)
        with pytest.raises(ValueError) as ex_info:
            new_params = ParameterFrame.from_json(json_path)
        assert (
            str(ex_info.value)
            == "Creating a ParameterFrame using from_json requires a verbose json format."
        )

    def test_set_values_from_json_file_validation(self, tmpdir):
        json_path = tmpdir.join("verbose_invalid.json")
        j = self.params.to_json(output_path=json_path, verbose=True)
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

    def test_parameter_source_tuple(self):
        params_copy = self.params.copy()

        # Set both value and source from a tuple
        params_copy.R_0 = (6.8, "New Value")
        assert len(params_copy.R_0.history()) == 2
        assert params_copy.R_0 == 6.8
        assert params_copy.R_0.source == "New Value"

    def test_parameter_source_dict(self):
        params_copy = self.params.copy()

        # Set both value and source from a dict
        params_copy.R_0 = {"value": 6.9, "source": None}
        assert len(params_copy.R_0.history()) == 2
        assert params_copy.R_0 == 6.9
        assert params_copy.R_0.source is None


class TestReactorSystem:

    # def setup(self):
    #     self.params = ParameterFrame(with_defaults=True)

    # @classmethod
    # def teardown_class(cls):
    #     ParameterFrame._clean()
    DIV = Dummy({"T_in": 88})

    def test_input_handling(self):
        assert self.DIV.params.T_in == 88

    def test_add_parameter(self):
        self.DIV.add_parameter("R_0", "Major radius", 8.8, "m", None, None)

        assert self.DIV.params.R_0 == 8.8
        assert self.DIV.params.R_0 == 8.8

        self.DIV.add_parameter("n_TF", "Number of TF coils", 17, "N/A", None, "Input")
        assert self.DIV.params.n_TF == 17
        assert self.DIV.params["n_TF"] == 17

    def test_add_parameters(self):
        p = [
            ["n_TF", "Number of TF coils", 200, "N/A", None, "Input"],
            ["n_PF", "Number of PF coils", 6, "N/A", None, "Input"],
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

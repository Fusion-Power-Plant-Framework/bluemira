import io
from dataclasses import dataclass
from typing import Union
from unittest import mock

import pytest

from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.base.parameter_frame import make_parameter_frame, parameter_frame
from bluemira.base.parameter_frame._parameter import ParamDictT

FRAME_DATA = {
    "height": {"value": 180.5, "unit": "cm"},
    "age": {"value": 30, "unit": "years"},
}


@dataclass
class BasicFrame(ParameterFrame):
    height: Parameter[float]
    age: Parameter[int]


@parameter_frame
class BasicFrameDec:
    height: Parameter[float]
    age: Parameter[int]


class TestParameterFrame:
    def test_init_from_dict_sets_valid_entries(self):
        frame = BasicFrame.from_dict(FRAME_DATA)

        assert frame.height.value == 180.5
        assert frame.height.unit == "cm"
        assert frame.age.value == 30
        assert frame.age.unit == "years"

    @pytest.mark.parametrize(
        "name, value",
        [("name", 100), ("value", "wrong type"), ("value", 30.5), ("unit", 0.5)],
    )
    def test_from_dict_TypeError_given_invalid_type(self, name, value):
        data = {
            "height": {"value": 180.5, "unit": "cm"},
            "age": {"value": 30, "unit": "years"},
        }
        data["age"][name] = value

        with pytest.raises(TypeError):
            BasicFrame.from_dict(data)

    def test_from_dict_ValueError_given_unknown_parameter(self):
        data = {
            "height": {"value": 180.5, "unit": "m"},
            "age": {"value": 30, "unit": "years"},
            "weight": {"value": 60, "unit": "kg"},
        }

        with pytest.raises(ValueError):
            BasicFrame.from_dict(data)

    def test_from_dict_ValueError_given_missing_parameter(self):
        data = {"height": {"value": 180.5, "unit": "m"}}

        with pytest.raises(ValueError):
            BasicFrame.from_dict(data)

    def test_to_dict_returns_input_to_from_dict(self):
        frame = BasicFrame.from_dict(FRAME_DATA)

        out_dict = frame.to_dict()

        assert out_dict == FRAME_DATA

    def test_initialising_using_untyped_generic_parameter_is_allowed(self):
        @dataclass
        class GenericFrame(ParameterFrame):
            x: Parameter

        frame = GenericFrame.from_dict({"x": {"value": 10}})

        assert frame.x.value == 10

    @pytest.mark.parametrize("value", ["OK", ["OK"]])
    def test_TypeError_given_field_has_Union_Parameter_type(self, value):
        @dataclass
        class GenericFrame(ParameterFrame):
            x: Parameter[Union[str, list]]

        with pytest.raises(TypeError):
            GenericFrame.from_dict({"x": {"value": value}})

    def test_TypeError_given_field_does_not_have_Parameter_type(self):
        @dataclass
        class BadFrame(ParameterFrame):
            x: int

        with pytest.raises(TypeError):
            BadFrame.from_dict({"x": {"value": 10}})

    def test_a_default_frame_is_empty(self):
        assert len(ParameterFrame().to_dict()) == 0

    def test_decorated_frame_equal_to_inherited(self):
        inherited_frame = BasicFrame.from_dict(FRAME_DATA)
        decorated_frame = BasicFrameDec.from_dict(FRAME_DATA)

        assert isinstance(decorated_frame, ParameterFrame)
        assert decorated_frame.height.value == inherited_frame.height.value
        assert decorated_frame.age.value == inherited_frame.age.value

    def test_from_json_reads_json_string(self):
        json_str = """{
            "height": {"value": 180.5, "unit": "cm"},
            "age": {"value": 30, "unit": "years"}
        }"""

        frame = BasicFrame.from_json(json_str)

        assert frame.height.value == 180.5
        assert frame.height.unit == "cm"
        assert frame.age.value == 30
        assert frame.age.unit == "years"

    def test_from_json_reads_json_io(self):
        json_str = """{
            "height": {"value": 180.5, "unit": "cm"},
            "age": {"value": 30, "unit": "years"}
        }"""
        json_io = io.StringIO(json_str)

        frame = BasicFrame.from_json(json_io)

        assert frame.height.value == 180.5
        assert frame.height.unit == "cm"
        assert frame.age.value == 30
        assert frame.age.unit == "years"

    def test_from_json_reads_from_file(self):
        json_str = """{
            "height": {"value": 180.5, "unit": "cm"},
            "age": {"value": 30, "unit": "years"}
        }"""
        with mock.patch(
            "builtins.open", new_callable=mock.mock_open, read_data=json_str
        ):
            frame = BasicFrame.from_json("./some/path")

        assert frame.height.value == 180.5
        assert frame.height.unit == "cm"
        assert frame.age.value == 30
        assert frame.age.unit == "years"

    def test_parameter_frames_with_eq_parameters_are_equal(self):
        frame1 = BasicFrame.from_dict(FRAME_DATA)
        frame2 = BasicFrame.from_dict(FRAME_DATA)

        assert frame1 == frame2

    def test_parameter_frames_with_different_parameters_are_not_equal(self):
        @dataclass
        class OtherFrame(ParameterFrame):
            height: Parameter[float]
            age: Parameter[int]
            weight: Parameter[float]

        frame1 = BasicFrame.from_dict(FRAME_DATA)
        frame2 = OtherFrame.from_dict(
            {**FRAME_DATA, "weight": {"value": 58.2, "unit": "kg"}}
        )

        assert frame1 != frame2

    @pytest.mark.parametrize("x", [1, "str", Parameter("x", 0.1, "m")])
    def test_frame_ne_to_non_frame(self, x):
        frame = BasicFrame.from_dict(FRAME_DATA)

        assert frame != x
        assert x != frame

    def test_update_values_edits_frames_values(self):
        frame = BasicFrame.from_dict(FRAME_DATA)

        frame.update_values({"height": 160.4}, source="a test")

        assert frame.height.value == 160.4
        assert frame.height.source == "a test"
        assert frame.age.value == 30
        assert frame.age.source != "a test"

    def _call_tabulate(self, head_keys):
        with mock.patch("bluemira.base.parameter_frame._frame.tabulate") as m_tb:
            BasicFrame.from_dict(FRAME_DATA).tabulate(keys=head_keys)

        (table_rows,), call_kwargs = m_tb.call_args

        return call_kwargs["headers"], table_rows

    @pytest.mark.parametrize(
        "head_keys, result",
        zip(
            [None, ["name", "value"]],
            [ParamDictT.__annotations__.keys(), ["name", "value"]],
        ),
    )
    def test_tabulate_headers(self, head_keys, result):
        headers, _ = self._call_tabulate(head_keys)
        assert set(headers) == set(result)

    def _get_data_keys_and_values(self, head_keys):
        # The columns and rows of the parameterframe are sorted
        data_keys = sorted(FRAME_DATA.keys())

        if head_keys is not None:
            fd_keys_list = list(set(head_keys) - set(FRAME_DATA.keys()))
        else:
            fd_keys_list = list(FRAME_DATA.keys())

        data_values = list(FRAME_DATA.values())
        data_values_index = sorted(
            range(len(fd_keys_list)), key=fd_keys_list.__getitem__
        )

        return data_keys, data_values, data_values_index

    @pytest.mark.parametrize("head_keys", [None, ["name", "value"]])
    def test_tabulate_method_columns_have_correct_num_of_NA(self, head_keys):
        # Number of 'N/A' equal to headers without name - number of filled keys

        nn_headers, table_rows = self._call_tabulate(head_keys)
        nn_headers.pop(nn_headers.index("name"))

        _, data_values, data_values_index = self._get_data_keys_and_values(head_keys)

        for tr, dvi in zip(table_rows, data_values_index):
            assert len([i for i, x in enumerate(tr) if x == "N/A"]) == len(
                nn_headers - data_values[dvi].keys()
            )

    @pytest.mark.parametrize("head_keys", [None, ["name", "value"]])
    def test_tabulate_method_columns_have_correct_data(self, head_keys):
        headers, table_rows = self._call_tabulate(head_keys)

        (
            data_keys,
            data_values,
            data_values_index,
        ) = self._get_data_keys_and_values(head_keys)

        for no, (tr, dvi) in enumerate(zip(table_rows, data_values_index)):
            # name is correct
            assert tr[0] == data_keys[no]
            for ind, val in data_values[dvi].items():
                try:
                    assert tr[headers.index(ind)] == FRAME_DATA[data_keys[no]][ind]
                except ValueError as ve:
                    if ind in head_keys:
                        raise ve

    def test_iterating_returns_parameters_in_declaration_order(self):
        frame = BasicFrame.from_dict(FRAME_DATA)

        params = []
        for param in frame:
            params.append(param)

        assert all(isinstance(p, Parameter) for p in params)
        assert [p.name for p in params] == ["height", "age"]

    def test_ValueError_creating_frame_from_non_superset_frame(self):
        @dataclass
        class OtherFrame(ParameterFrame):
            height: Parameter[float]
            age: Parameter[int]
            weight: Parameter[float]

        basic_frame = BasicFrame.from_dict(FRAME_DATA)

        with pytest.raises(ValueError):
            OtherFrame.from_frame(basic_frame)

    def test_from_json_ValueError_given_non_string_or_buffer(self):
        with pytest.raises(ValueError) as error:
            BasicFrame.from_json(["x"])
        assert "Cannot read JSON" in str(error)


class TestParameterSetup:
    def test_params_None(self):
        with pytest.raises(ValueError):
            make_parameter_frame(FRAME_DATA, None)
        with pytest.raises(TypeError):
            make_parameter_frame(None, BasicFrame)
        params = make_parameter_frame(None, None)
        assert params is None

    @pytest.mark.parametrize(
        "frame",
        [
            FRAME_DATA,
            BasicFrame.from_dict(FRAME_DATA),
            BasicFrameDec.from_dict(FRAME_DATA),
        ],
    )
    def test_params_type(self, frame):
        params = make_parameter_frame(frame, BasicFrame)
        assert isinstance(params, BasicFrame)


def test_changes_to_parameters_are_propagated_between_frames():
    """
    This tests a key mechanic that updates to parameters in a frame
    made from some base frame, are propagated to the base frame.
    This allows builders/designers to update parameters at the reactor
    level, whilst still only working with their specific frame.
    """

    @dataclass
    class SubFrame(ParameterFrame):
        height: Parameter[float]

    base_frame = BasicFrame.from_dict(FRAME_DATA)
    slim_frame = SubFrame.from_frame(base_frame)

    slim_frame.height.value = 200.5

    assert base_frame.height.value == 200.5

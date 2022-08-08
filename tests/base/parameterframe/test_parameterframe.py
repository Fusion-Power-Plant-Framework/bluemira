import io
from dataclasses import dataclass
from typing import Union
from unittest import mock

import pytest

from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.base.parameter_frame import parameter_frame


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
        frame = BasicFrame.from_dict(
            {
                "height": {"name": "height", "value": 180.5, "unit": "cm"},
                "age": {"name": "age", "value": 30, "unit": "years"},
            }
        )

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
            "height": {"name": "height", "value": 180.5, "unit": "m"},
            "age": {"name": "age", "value": 30, "unit": "years"},
        }
        data["age"][name] = value

        with pytest.raises(TypeError):
            BasicFrame.from_dict(data)

    def test_from_dict_ValueError_given_unknown_parameter(self):
        data = {
            "height": {"name": "height", "value": 180.5, "unit": "m"},
            "age": {"name": "age", "value": 30, "unit": "years"},
            "weight": {"name": "weight", "value": 60, "unit": "kg"},
        }

        with pytest.raises(ValueError):
            BasicFrame.from_dict(data)

    def test_from_dict_ValueError_given_missing_parameter(self):
        data = {"height": {"name": "height", "value": 180.5, "unit": "m"}}

        with pytest.raises(ValueError):
            BasicFrame.from_dict(data)

    def test_to_dict_returns_input_to_from_dict(self):
        data = {
            "height": {"name": "height", "value": 180.5, "unit": "cm"},
            "age": {"name": "age", "value": 30, "unit": "years"},
        }
        frame = BasicFrame.from_dict(data)

        out_dict = frame.to_dict()

        assert out_dict == data

    def test_initialising_using_untyped_generic_parameter_is_allowed(self):
        @dataclass
        class GenericFrame(ParameterFrame):
            x: Parameter

        frame = GenericFrame.from_dict({"x": {"name": "x", "value": 10}})

        assert frame.x.value == 10

    @pytest.mark.parametrize("value", ["OK", ["OK"]])
    def test_TypeError_given_field_has_Union_Parameter_type(self, value):
        @dataclass
        class GenericFrame(ParameterFrame):
            x: Parameter[Union[str, list]]

        with pytest.raises(TypeError):
            GenericFrame.from_dict({"x": {"name": "x", "value": value}})

    def test_TypeError_given_field_does_not_have_Parameter_type(self):
        @dataclass
        class BadFrame(ParameterFrame):
            x: int

        with pytest.raises(TypeError):
            BadFrame.from_dict({"x": {"name": "x", "value": 10}})

    def test_a_default_frame_is_empty(self):
        assert len(ParameterFrame().to_dict()) == 0

    def test_decorated_frame_equal_to_inherited(self):
        data = {
            "height": {"name": "height", "value": 180.5, "unit": "cm"},
            "age": {"name": "age", "value": 30, "unit": "years"},
        }
        inherited_frame = BasicFrame.from_dict(data)
        decorated_frame = BasicFrameDec.from_dict(data)

        assert isinstance(decorated_frame, ParameterFrame)
        assert decorated_frame.height.value == inherited_frame.height.value
        assert decorated_frame.age.value == inherited_frame.age.value

    def test_from_json_reads_json_string(self):
        json_str = """{
            "height": {"name": "height", "value": 180.5, "unit": "cm"},
            "age": {"name": "age", "value": 30, "unit": "years"}
        }"""

        frame = BasicFrame.from_json(json_str)

        assert frame.height.value == 180.5
        assert frame.height.unit == "cm"
        assert frame.age.value == 30
        assert frame.age.unit == "years"

    def test_from_json_reads_json_io(self):
        json_str = """{
            "height": {"name": "height", "value": 180.5, "unit": "cm"},
            "age": {"name": "age", "value": 30, "unit": "years"}
        }"""
        json_io = io.StringIO(json_str)

        frame = BasicFrame.from_json(json_io)

        assert frame.height.value == 180.5
        assert frame.height.unit == "cm"
        assert frame.age.value == 30
        assert frame.age.unit == "years"

    def test_from_json_reads_from_file(self):
        json_str = """{
            "height": {"name": "height", "value": 180.5, "unit": "cm"},
            "age": {"name": "age", "value": 30, "unit": "years"}
        }"""
        with mock.patch(
            "builtins.open", new_callable=mock.mock_open, read_data=json_str
        ):
            frame = BasicFrame.from_json("./some/path")

        assert frame.height.value == 180.5
        assert frame.height.unit == "cm"
        assert frame.age.value == 30
        assert frame.age.unit == "years"

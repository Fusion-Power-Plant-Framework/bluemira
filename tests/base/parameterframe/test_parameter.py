import copy
import re

import pytest

from bluemira.base.parameter_frame import NewParameter as Parameter


class TestParameter:

    SERIALIZED_PARAM = {
        "name": "my_param",
        "value": 100,
        "description": "A parameter for testing on.",
        "source": "test",
        "unit": "dimensionless",
        "long_name": "My parameter",
    }

    def test_fields_set_on_init(self):
        param = Parameter(**self.SERIALIZED_PARAM)

        assert param.name == "my_param"
        assert param.value == 100
        assert param.description == "A parameter for testing on."
        assert param.long_name == "My parameter"
        assert param.source == "test"
        assert param.unit == "dimensionless"

    def test_initial_value_added_to_history_on_init(self):
        param = Parameter(**self.SERIALIZED_PARAM)

        history = param.history()
        assert len(history) == 1
        assert history[0].value == 100
        assert history[0].source == "test"

    def test_value_added_to_history_on_being_set(self):
        param = Parameter(**self.SERIALIZED_PARAM)

        param.value = 200
        history = param.history()

        assert len(history) == 2
        assert history[1].value == 200
        assert history[1].source == ""

    def test_value_added_to_history_given_set_value_called(self):
        param = Parameter(**self.SERIALIZED_PARAM)

        param.set_value(150, "test_case")
        history = param.history()

        assert len(history) == 2
        assert history[1].value == 150
        assert history[1].source == "test_case"

    @pytest.mark.parametrize(
        "arg, value",
        [
            ("name", 1),
            ("unit", 0),
            ("source", [100]),
            ("description", 0.5),
            ("long_name", ["a", "b"]),
        ],
    )
    def test_TypeError_on_init_given_arg_incorrect_type(self, arg, value):
        kwargs = copy.deepcopy(self.SERIALIZED_PARAM)
        kwargs[arg] = value

        with pytest.raises(TypeError):
            Parameter(**kwargs)

    def test_to_dict_returns_equal_dict_as_used_in_init(self):
        param = Parameter(**self.SERIALIZED_PARAM)

        output = param.to_dict()

        assert output == self.SERIALIZED_PARAM

    @pytest.mark.parametrize("value_type", [list, str])
    def test_TypeError_given_value_type_different_to_arg_type(self, value_type):
        with pytest.raises(TypeError):
            Parameter(**self.SERIALIZED_PARAM, _value_types=(value_type,))

    def test_int_values_converted_to_float_given_float_value_type(self):
        param = Parameter(**self.SERIALIZED_PARAM, _value_types=(float,))

        assert isinstance(param.value, float)

    def test_repr_contains_name_value_and_unit(self):
        param = Parameter(**self.SERIALIZED_PARAM)

        assert re.search("my_param=100dimensionless", repr(param))

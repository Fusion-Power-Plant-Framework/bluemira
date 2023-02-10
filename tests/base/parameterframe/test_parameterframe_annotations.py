from __future__ import annotations

from dataclasses import dataclass

from bluemira.base.parameter_frame import Parameter, ParameterFrame


@dataclass
class PFrame(ParameterFrame):

    a: Parameter[float]
    b: Parameter[int]


def test_future_annotations_are_typed():
    """
    The purpose of the test is to check we can still do type validation
    when we've imported annotations (because of delayed evaluations).
    """

    d = {
        "a": {"value": 3.14, "unit": ""},
        "b": {"value": 1, "unit": ""},
    }

    f = PFrame.from_dict(d)

    # importing annotations converts the typing to a string
    assert isinstance(PFrame.__annotations__["a"], str)
    assert not isinstance(f._types["a"], str)

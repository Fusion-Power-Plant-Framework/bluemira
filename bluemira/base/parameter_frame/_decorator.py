from dataclasses import make_dataclass
from typing import Type, TypeVar

from bluemira.base.parameter_frame._frame import NewParameterFrame

_T = TypeVar("_T")
_RetT = TypeVar("_RetT", bound=NewParameterFrame)


def parameter_frame(cls: Type[_T]) -> Type[_RetT]:
    """Decorator to convert class definition to a NewParameterFrame"""
    return make_dataclass(
        cls.__name__, cls.__annotations__.items(), bases=(NewParameterFrame,)
    )

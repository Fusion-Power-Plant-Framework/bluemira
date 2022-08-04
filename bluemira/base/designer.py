import abc
from typing import Dict, Generic, Type, TypeVar, Union

from bluemira.base.parameter import ParameterFrame

_DesignerReturnT = TypeVar("_DesignerReturnT")
_ParamT = TypeVar("_ParamT", bound=ParameterFrame)


class Designer(abc.ABC, Generic[_DesignerReturnT]):
    @abc.abstractproperty
    def param_cls(self) -> Type[ParameterFrame]:
        pass

    def __init__(self, params: Union[ParameterFrame, Dict]):
        super().__init__()
        self.params = self._init_params(params)

    @abc.abstractmethod
    def run(self) -> _DesignerReturnT:
        pass

    def _init_params(self, params: Union[Dict, _ParamT]) -> _ParamT:
        if isinstance(params, dict):
            return self.param_cls.from_dict(params)
        elif isinstance(params, ParameterFrame):
            return params
        raise TypeError(
            f"Cannot interpret type '{type(params)}' as {self.param_cls.__name__}."
        )

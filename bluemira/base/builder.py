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

"""
Interfaces for builder and build steps classes
"""

from __future__ import annotations

import abc
from typing import Dict, Generic, Optional, Type, TypeVar, Union

from bluemira.base.components import Component
from bluemira.base.designer import Designer
from bluemira.base.parameter import ParameterFrame

_ComponentManagerT = TypeVar("_ComponentManagerT")
_ParamT = TypeVar("_ParamT", bound=ParameterFrame)


def _remove_suffix(s: str, suffix: str) -> str:
    # Python 3.9 has str.removesuffix()
    if suffix and s.endswith(suffix):
        return s[: -len(suffix)]
    return s


class Builder(abc.ABC, Generic[_ComponentManagerT]):
    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
        designer: Optional[Designer] = None,
    ):
        super().__init__()
        self.name = build_config.get(
            "name", _remove_suffix(self.__class__.__name__, "Builder")
        )
        self.params = self._init_params(params)
        self.build_config = build_config
        self.designer = designer

    @abc.abstractproperty
    def param_cls(self) -> Type[_ParamT]:
        pass

    @abc.abstractmethod
    def build(self) -> _ComponentManagerT:
        return Component(self.name)

    def _init_params(self, params: Union[Dict, _ParamT]) -> _ParamT:
        if isinstance(params, dict):
            return self.param_cls.from_dict(params)
        elif isinstance(params, ParameterFrame):
            return params
        raise TypeError(
            f"Cannot interpret type '{type(params)}' as {self.param_cls.__name__}."
        )

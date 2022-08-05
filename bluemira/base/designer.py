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
Interfaces for designer classes.
"""

import abc
from typing import Dict, Generic, Type, TypeVar, Union

from bluemira.base.parameter import ParameterFrame

_DesignerReturnT = TypeVar("_DesignerReturnT")
_ParameterFrameT = TypeVar("_ParameterFrameT", bound=ParameterFrame)


class Designer(abc.ABC, Generic[_DesignerReturnT]):
    """
    Base class for 'Designers' that solver design problems as part of
    building a reactor component.

    Parameters
    ----------
    params: Union[_ParameterFrameT, Dict]
        The parameters required by the designer.
    """

    def __init__(self, params: Union[_ParameterFrameT, Dict]):
        super().__init__()
        self.params = self._init_params(params)

    @abc.abstractproperty
    def param_cls(self) -> Type[ParameterFrame]:
        """The class to hold this designer's parameters."""
        pass

    @abc.abstractmethod
    def run(self) -> _DesignerReturnT:
        """Run the design."""
        pass

    def _init_params(self, params: Union[Dict, _ParameterFrameT]) -> _ParameterFrameT:
        if isinstance(params, dict):
            return self.param_cls.from_dict(params)
        elif isinstance(params, ParameterFrame):
            return params
        raise TypeError(
            f"Cannot interpret type '{type(params)}' as {self.param_cls.__name__}."
        )

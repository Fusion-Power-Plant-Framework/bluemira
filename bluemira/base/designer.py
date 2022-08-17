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
from typing import Generic, TypeVar

from bluemira.base.parameter_frame import ParameterSetup

_DesignerReturnT = TypeVar("_DesignerReturnT")


class Designer(ParameterSetup, Generic[_DesignerReturnT]):
    """
    Base class for 'Designers' that solver design problems as part of
    building a reactor component.

    Parameters
    ----------
    params: Union[_ParameterFrameT, Dict]
        The parameters required by the designer.

    Notes
    -----
    If there are no parameters associated with a concrete builder, set
    `param_cls` to `None` and pass `None` into this class's constructor.
    If param_cls is not `None` `param_cls` is set up with an empty dictionary.
    """

    @abc.abstractmethod
    def run(self) -> _DesignerReturnT:
        """Run the design."""
        pass

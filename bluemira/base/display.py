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
Module containing interfaces and basic implementation for 3D displaying functionality.
"""

import abc
from dataclasses import dataclass
from typing import Optional, Tuple

from bluemira.base.look_and_feel import bluemira_warn
from BLUEPRINT.utilities.tools import get_module  # TODO: PR to move to bluemira


@dataclass(frozen=True)
class DisplayOptions:
    rgb: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    transparency: float = 0.0


class Displayer(abc.ABC):
    def __init__(
        self,
        options: DisplayOptions = DisplayOptions(),
        api: str = "bluemira.geometry._freecadapi",
    ):
        self._options = options
        self._display_func = get_module(api).display

    @property
    def options(self) -> DisplayOptions:
        return self._options

    @abc.abstractmethod
    def display(self, obj, options: Optional[DisplayOptions] = None) -> None:
        self._display_func(obj, options)


class Displayable:
    _displayer: Displayer

    @property
    def display_options(self) -> DisplayOptions:
        return self._displayer.options

    @display_options.setter
    def display_options(self, value: DisplayOptions):
        self._displayer.options = value

    def display(self, options: Optional[DisplayOptions] = None) -> None:
        self._displayer.display(self, options)


class BasicDisplayer(Displayer):
    def display(self, obj, options: Optional[DisplayOptions]) -> None:
        if options is None:
            options = self.options

        try:
            super().display(obj, options)
        except Exception as e:
            bluemira_warn(f"Unable to display object {obj} - {e}")

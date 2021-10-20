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

from bluemira.base.error import DisplayError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.tools import get_module


@dataclass
class DisplayOptions:
    """
    The options that are available for displaying objects in 3D

    Parameters
    ----------
    rgb: Tuple[float, float, float]
        The RBG colour to display the object, by default (0.5, 0.5, 0.5).
    transparency: float
        The transparency to display the object, by default 0.0.
    """

    rgb: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    transparency: float = 0.0


class Displayer(abc.ABC):
    """
    Abstract base class to handle displaying objects

    Parameters
    ----------
    options: Optional[DisplayOptions]
        The options to use to display the object, by default None in which case the
        default values for the DisplayOptions class are used.
    api: str
        The API to use for displaying. This must implement a display method with
        signature (objs, options), where objs are the primitive 3D object to display. By
        default uses the FreeCAD api at bluemira.geometry._freecadapi.
    """

    def __init__(
        self,
        options: Optional[DisplayOptions] = None,
        api: str = "bluemira.geometry._freecadapi",
    ):
        self._options = DisplayOptions() if options is None else options
        self._display_func = get_module(api).display

    @property
    def options(self) -> DisplayOptions:
        """
        The options that will be used to display the object.
        """
        return self._options

    @options.setter
    def options(self, val: DisplayOptions) -> None:
        self._options = val

    @abc.abstractmethod
    def display(self, obj, options: Optional[DisplayOptions] = None) -> None:
        """
        Display the object by calling the display function within the API.

        Parameters
        ----------
        obj
            The object to display
        options: Optional[DisplayOptions]
            The options to use to display the object, by default None in which case the
        default values for the DisplayOptions class are used.
        """
        self._display_func(obj, options)


class Displayable:
    """
    Mixin class to make a class displayable by imparting a display method and options.

    The implementing class must set the _displayer attribute to an instance of the
    appropriate Displayer class.
    """

    _displayer: Displayer

    @property
    def display_options(self) -> DisplayOptions:
        """
        The options that will be used to display the object.
        """
        return self._displayer.options

    @display_options.setter
    def display_options(self, value: DisplayOptions):
        if not isinstance(value, DisplayOptions):
            raise DisplayError(
                "Display options must be set to a DisplayOptions instance."
            )
        self._displayer.options = value

    def display(self, options: Optional[DisplayOptions] = None) -> None:
        """
        Default method to call display the object by calling into the Displayer's display
        method.

        Parameters
        ----------
        options: Optional[DisplayOptions]
            If not None then override the object's display_options with the provided
            options. By default None.
        """
        self._displayer.display(self, options)


class BasicDisplayer(Displayer):
    """
    A basic implementation of a Displayer that can display provided primitive objects
    with the provided options.
    """

    def display(self, obj, options: Optional[DisplayOptions] = None) -> None:
        """
        Display the primitive objects with the provided options.

        Parameters
        ----------
        obj
            The CAD primitive objects to be displayed.
        options: Optional[DisplayOptions]
            The options to use to display the primitives.
        """
        if options is None:
            options = self.options

        try:
            super().display(obj, options)
        except Exception as e:
            bluemira_warn(f"Unable to display object {obj} - {e}")

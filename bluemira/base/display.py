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
Module containing the base display classes.
"""
import abc
import dataclasses
from typing import Optional, Tuple
from bluemira.utilities.tools import get_module
from .error import DisplayError
from .look_and_feel import bluemira_warn
from . import _matplotlib_plot


class DisplayOptions():
    """
    The options that are available for displaying objects
    """
    _options = None

    def asdict(self):
        return self._options


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
        default None is used -> no display output.
    """

    def __init__(
        self,
        options: Optional[DisplayOptions] = None,
        api: str = None,
    ):
        self._options = DisplayOptions() if options is None else options
        self._display_func: str

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
    def _display(self, obj, options: Optional[DisplayOptions] = None, *args, **kwargs) -> None:
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
        return self._display_func(obj, options, *args, **kwargs)


# # =============================================================================
# # Plot2D
# # =============================================================================
class Plot2DOptions(DisplayOptions):
    """
    The options that are available for 2D-plotting objects
    """
    pass


class Plotter2D(Displayer):

    def __init__(
        self,
        options: Optional[Plot2DOptions] = None,
        api: str = 'bluemira.base._matplotlib_plot',
    ):
        super().__init__(options, api)
        self._options = _matplotlib_plot.MatplotlibOptions() if options is None else \
            options
        self._display_func = get_module(api).plot2d

    def _display(self, obj, options: Optional[Plot2DOptions] = None, *args, **kwargs) -> None:
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

        #try:
        return super()._display(obj, options, *args, **kwargs)
        #except Exception as e:
        #    bluemira_warn(f"Unable to display object {obj} - {e}")

    def plot2d(self, obj, options: Optional[DisplayOptions] = None, *args, **kwargs) -> \
            None:
        """
        2D plot the object by calling the display function within the API.
        Parameters
        ----------
        obj
            The object to display
        options: Optional[DisplayOptions]
            The options to use to display the object, by default None in which case the
        default values for the DisplayOptions class are used.
        """
        return self._display(obj, options, *args, **kwargs)


class Plottable2D:
    """
    Mixin class to make a class displayable by imparting a display method and options.
    The implementing class must set the _displayer attribute to an instance of the
    appropriate Displayer class.
    """

    _plotter2d: Plotter2D = None

    @property
    def plot2d_options(self) -> Plot2DOptions:
        """
        The options that will be used to display the object.
        """
        return self._plotter2d.options

    @plot2d_options.setter
    def plot2d_options(self, value: Plot2DOptions):
        if not isinstance(value, Plot2DOptions):
            raise DisplayError(
                "Display options must be set to a Plot2DOptions instance."
            )
        self._plotter2d.options = value

    def plot2d(self, options: Optional[Plot2DOptions] = None, *args, **kwargs) -> None:
        """
        Default method to call display the object by calling into the Displayer's display
        method.
        Parameters
        ----------
        options: Optional[DisplayOptions]
            If not None then override the object's display_options with the provided
            options. By default None.
        """
        return self._plotter2d.plot2d(self, options, *args, **kwargs)


# # =============================================================================
# # PlotCAD
# # =============================================================================
class PlotCADOptions(DisplayOptions):
    """
    The options that are available for 2D-plotting objects
    """
    pass


class PlotterCAD(Displayer):

    def plotcad(self, obj, options: Optional[DisplayOptions] = None, *args, **kwargs) -> None:
        """
        2D plot the object by calling the display function within the API.
        Parameters
        ----------
        obj
            The object to display
        options: Optional[DisplayOptions]
            The options to use to display the object, by default None in which case the
        default values for the DisplayOptions class are used.
        """
        return self.__display(obj, options, *args, **kwargs)


class PlottableCAD:
    """
    Mixin class to make a class displayable by imparting a display method and options.
    The implementing class must set the _displayer attribute to an instance of the
    appropriate Displayer class.
    """

    _plottercad: PlotterCAD = None

    @property
    def plotcad_options(self) -> PlotCADOptions:
        """
        The options that will be used to display the object.
        """
        return self._plottercad.options

    @plotcad_options.setter
    def plotcad_options(self, value: PlotCADOptions):
        if not isinstance(value, PlotCADOptions):
            raise DisplayError(
                "Display options must be set to a Plot2DOptions instance."
            )
        self._plottercad.options = value

    def plotcad(self, options: Optional[PlotCADOptions] = None, *args, **kwargs) -> None:
        """
        Default method to call display the object by calling into the Displayer's display
        method.
        Parameters
        ----------
        options: Optional[DisplayOptions]
            If not None then override the object's display_options with the provided
            options. By default None.
        """
        self._plottercad.plotcad(self, options, *args, **kwargs)

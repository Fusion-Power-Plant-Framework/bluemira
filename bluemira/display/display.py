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
from typing import Optional
from bluemira.utilities.tools import get_module
from .error import DisplayError


class DisplayOptions:
    """
    The options that are available for displaying objects.
    """

    _options = None

    def as_dict(self):
        """
        Returns the instance as a dictionary.
        """
        return self._options


class Displayer(abc.ABC):
    """
    Abstract base class to handle displaying objects.

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
    def _display(self, obj, options: Optional[DisplayOptions] = None, *args, **kwargs):
        """
        Display the object by calling the display function within the API.

        Parameters
        ----------
        obj
            The object to display
        options: Optional[DisplayOptions]
            The options to use to display the object, by default None in which case the
            default values for the DisplayOptions class are used.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        return self._display_func(obj, options, *args, **kwargs)


# # =============================================================================
# # Plot2D
# # =============================================================================
class Plot2DOptions(DisplayOptions):
    """
    The options that are available for plotting objects in 2D.
    """

    pass


class Plotter2D(Displayer):
    """
    A class for plotting primitive objects in 3D.
    """

    def __init__(
        self,
        options: Optional[Plot2DOptions] = None,
        api: str = "bluemira.display._matplotlib_plot",
    ):
        super().__init__(options, api)
        self._options = get_module(api)._Plot2DOptions() if options is None else options
        self._display_func = get_module(api).plot_2d

    def _display(self, obj, options: Optional[Plot2DOptions] = None, *args, **kwargs):
        """
        Display the primitive objects with the provided options.

        Parameters
        ----------
        obj
            The CAD primitive objects to be displayed.
        options: Optional[DisplayOptions]
            The options to use to display the primitives.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        if options is None:
            options = self.options

        return super()._display(obj, options, *args, **kwargs)

    def plot_2d(
        self, obj, options: Optional[Plot2DOptions] = None, *args, **kwargs
    ) -> None:
        """
        2D plot the object by calling the display function within the API.

        Parameters
        ----------
        obj
            The object to display
        options: Optional[DisplayOptions]
            The options to use to display the object, by default None in which case the
        default values for the DisplayOptions class are used.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        return self._display(obj, options, *args, **kwargs)


class Plottable2D:
    """
    Mixin class to make a class plottable in 2D by imparting a plot2d method and
    options.

    Notes
    -----
    The implementing class must set the _plotter2D attribute to an instance of the
    appropriate Plotter2D class.
    """

    _plotter_2d: Plotter2D = None

    @property
    def plot_2d_options(self) -> Plot2DOptions:
        """
        The options that will be used to display the object.
        """
        return self._plotter_2d.options

    @plot_2d_options.setter
    def plot_2d_options(self, value: Plot2DOptions):
        if not isinstance(value, Plot2DOptions):
            raise DisplayError(
                "Display options must be set to a Plot2DOptions instance."
            )
        self._plotter_2d.options = value

    def plot_2d(self, options: Optional[Plot2DOptions] = None, *args, **kwargs) -> None:
        """
        Default method to call display the object by calling into the Displayer's display
        method.

        Parameters
        ----------
        options: Optional[DisplayOptions]
            If not None then override the object's display_options with the provided
            options. By default None.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        return self._plotter_2d.plot_2d(self, options, *args, **kwargs)


# # =============================================================================
# # Plot3D
# # =============================================================================
class Plot3DOptions(DisplayOptions):
    """
    The options that are available for plotting objects in 3D
    """

    pass


class Plotter3D(Displayer):
    """
    A class for plotting primitive objects in 3D.
    """

    def __init__(
        self,
        options: Optional[Plot3DOptions] = None,
        api: str = "bluemira.display._matplotlib_plot",
    ):
        super().__init__(options, api)
        self._options = get_module(api)._Plot3DOptions() if options is None else options
        self._display_func = get_module(api).plot_3d

    def _display(self, obj, options: Optional[Plot3DOptions] = None, *args, **kwargs):
        """
        Display the primitive objects with the provided options.

        Parameters
        ----------
        obj
            The primitive objects to be displayed.
        options: Optional[DisplayOptions]
            The options to use to display the primitives.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        if options is None:
            options = self.options

        return super()._display(obj, options, *args, **kwargs)

    def plot_3d(
        self, obj, options: Optional[Plot3DOptions] = None, *args, **kwargs
    ) -> None:
        """
        3D plot the object by calling the display function within the API.

        Parameters
        ----------
        obj
            The object to display
        options: Optional[DisplayOptions]
            The options to use to display the object, by default None in which case the
        default values for the DisplayOptions class are used.
        """
        return self._display(obj, options, *args, **kwargs)


class Plottable3D:
    """
    Mixin class to make a class plottable in 3D by imparting a plot3D method and options.

    Notes
    -----
    The implementing class must set the _plotter3d attribute to an instance of the
    appropriate Displayer class.
    """

    _plotter_3d: Plotter3D = None

    @property
    def plot_3d_options(self) -> Plot3DOptions:
        """
        The options that will be used to display the object.
        """
        return self._plotter_3d.options

    @plot_3d_options.setter
    def plot_3d_options(self, value: Plot3DOptions):
        if not isinstance(value, Plot3DOptions):
            raise DisplayError(
                "Display options must be set to a Plot3DOptions instance."
            )
        self._plotter_3d.options = value

    def plot_3d(self, options: Optional[Plot3DOptions] = None, *args, **kwargs) -> None:
        """
        Default method to call display the object by calling into the Displayer's display
        method.

        Parameters
        ----------
        options: Optional[DisplayOptions]
            If not None then override the object's display_options with the provided
            options. By default None.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        return self._plotter_3d.plot_3d(self, options, *args, **kwargs)


# # =============================================================================
# # PlotCAD
# # =============================================================================
class DisplayCADOptions(DisplayOptions):
    """
    The options that are available for displaying CAD objects
    """

    pass


class DisplayerCAD(Displayer):
    """
    A class for displaying CAD representations of primitive objects.
    """

    def __init__(
        self,
        options: Optional[DisplayCADOptions] = None,
        api: str = "bluemira.display._freecad_plot",
    ):
        super().__init__(options, api)
        self._options = (
            get_module(api)._DisplayCADOptions() if options is None else options
        )
        self._display_func = get_module(api).show_cad

    def _display(
        self, obj, options: Optional[DisplayCADOptions] = None, *args, **kwargs
    ) -> None:
        """
        Display the primitive objects with the provided options.

        Parameters
        ----------
        obj
            The CAD primitive objects to be displayed.
        options: Optional[DisplayOptions]
            The options to use to display the primitives.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        if options is None:
            options = self.options

        return super()._display(obj, options, *args, **kwargs)

    def show_cad(
        self, obj, options: Optional[DisplayCADOptions] = None, *args, **kwargs
    ) -> None:
        """
        Display the CAD object by calling the display function within the API.

        Parameters
        ----------
        obj
            The object to display
        options: Optional[DisplayOptions]
            The options to use to display the object, by default None in which case the
            default values for the DisplayOptions class are used.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        return self._display(obj, options, *args, **kwargs)


class DisplayableCAD:
    """
    Mixin class to make a class displayable by imparting a plotcad method and options.

    Notes
    -----
    The implementing class must set the _plottercad attribute to an instance of the
    appropriate Displayer class.
    """

    _displayer_cad: DisplayerCAD = None

    @property
    def displayer_cad_options(self) -> DisplayCADOptions:
        """
        The options that will be used to display the object.
        """
        return self._displayer_cad.options

    @displayer_cad_options.setter
    def displayer_cad_options(self, value: DisplayCADOptions):
        if not isinstance(value, DisplayCADOptions):
            raise DisplayError(
                "Display options must be set to a PlotCADOptions instance."
            )
        self._displayer_cad.options = value

    def show_cad(
        self, options: Optional[DisplayCADOptions] = None, *args, **kwargs
    ) -> None:
        """
        Default method to call display the object by calling into the Displayer's display
        method.

        Parameters
        ----------
        options: Optional[DisplayOptions]
            If not None then override the object's display_options with the provided
            options. By default None.
        """
        self._displayer_cad.show_cad(self, options, *args, **kwargs)

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
api for plotting using matplotlib
"""
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt

from typing import Optional, Union, List

import bluemira.geometry as geo
from bluemira.display import display
from .error import DisplayError

import copy
import numpy as np

DEFAULT = {
    # flags to enable points, wires, and faces plot
    "show_points": True,
    "show_wires": True,
    "show_faces": True,
    # matplotlib set of options to plot points, wires, and faces. If an empty dictionary
    # is specified, the default color plot of matplotlib is used.
    "poptions": {"s": 10, "facecolors": "red", "edgecolors": "black", "zorder": 30},
    "woptions": {"color": "black", "linewidth": "0.5", "zorder": 20},
    "foptions": {"color": "blue", "zorder": 10},
    # projection plane
    "plane": "xz",
    # palette
    # TODO: it's use is still in progress
    "palette": None,
    # discretization properties for plotting wires (and faces)
    "ndiscr": 100,
    "byedges": True,
}


# Note: This class cannot be an instance of dataclasses 'cause it fails when inserting
# a field that is a Base.Placement (or something that contains a Base.Placement). The
# given error is "TypeError: can't pickle Base.Placement objects"
class _Plot2DOptions(display.Plot2DOptions):
    """
    The options that are available for plotting objects in 3D.

    Parameters
    ----------
    show_points: bool
        If True, points are plotted. By default True.
    show_wires: bool
        If True, wires are plotted. By default True.
    show_faces: bool
        If True, faces are plotted. By default True.
    poptions: Dict
        Dictionary with matplotlib options for points. By default  {"s": 10,
        "facecolors": "blue", "edgecolors": "black"}
    woptions: Dict
        Dictionary with matplotlib options for wires. By default {"color": "black",
        "linewidth": "0.5"}
    foptions: Dict
        Dictionary with matplotlib options for faces. By default {"color": "red"}
    plane: [str, Plane]
        The plane on which the object is projected for plotting. As string, possible
        options are "xy", "xz", "zy". By default 'xz'.
    palette:
        The colour palette.
    ndiscr: int
        The number of points to use when discretising a wire or face.
    byedges: bool
        If True then wires or faces will be discretised respecting their edges.
    """

    def __init__(self, **kwargs):
        self._options = copy.deepcopy(DEFAULT)
        if kwargs:
            for k in kwargs:
                if k in self._options:
                    self._options[k] = kwargs[k]

    @property
    def show_points(self):
        """
        If true, points are plotted.
        """
        return self._options["show_points"]

    @show_points.setter
    def show_points(self, val):
        self._options["show_points"] = val

    @property
    def show_wires(self):
        """
        If True, wires are plotted.
        """
        return self._options["show_wires"]

    @show_wires.setter
    def show_wires(self, val):
        self._options["show_wires"] = val

    @property
    def show_faces(self):
        """
        If True, faces are plotted.
        """
        return self._options["show_faces"]

    @show_faces.setter
    def show_faces(self, val):
        self._options["show_faces"] = val

    @property
    def poptions(self):
        """
        Dictionary with matplotlib options for points.
        """
        return self._options["poptions"]

    @poptions.setter
    def poptions(self, val):
        self._options["poptions"] = val

    @property
    def woptions(self):
        """
        Dictionary with matplotlib options for wires.
        """
        return self._options["woptions"]

    @woptions.setter
    def woptions(self, val):
        self._options["woptions"] = val

    @property
    def foptions(self):
        """
        Dictionary with matplotlib options for faces.
        """
        return self._options["foptions"]

    @foptions.setter
    def foptions(self, val):
        self._options["foptions"] = val

    @property
    def plane(self):
        """
        The plane on which the object is projected for plotting. As string, possible
        options are "xy", "xz", "zy".
        """
        return self._options["plane"]

    @plane.setter
    def plane(self, val):
        self._options["plane"] = val

    @property
    def palette(self):
        """
        The colour palette.
        """
        return self._options["palette"]

    @palette.setter
    def palette(self, val):
        self._options["palette"] = val

    @property
    def ndiscr(self):
        """
        The number of points to use when discretising a wire or face.
        """
        return self._options["ndiscr"]

    @ndiscr.setter
    def ndiscr(self, val):
        self._options["ndiscr"] = val

    @property
    def byedges(self):
        """
        If True then wires or faces will be discretised respecting their edges.
        """
        return self._options["byedges"]

    @byedges.setter
    def byedges(self, val):
        self._options["byedges"] = val


# The definition of this class is necessary to maintain the consistency with the
# display architecture.
class _Plot3DOptions(display.Plot3DOptions, _Plot2DOptions):
    pass


# Note: when plotting points, it can happen that markers are not centred properly as
# described in https://github.com/matplotlib/matplotlib/issues/11836
class BasePlotter(ABC):
    """
    Base utility plotting class
    """

    def __init__(self, options: Optional[_Plot2DOptions] = None, *args, **kwargs):
        # discretization points representing the shape in global coordinate system
        self._data = []
        # modified discretization points for plotting (e.g. after plane transformation)
        self._data_to_plot = []
        self.ax = None
        self.options = _Plot2DOptions(**kwargs) if options is None else options
        self.set_plane(self.options._options["plane"])

    def set_plane(self, plane):
        """Set the plotting plane"""
        if plane == "xy":
            # Base.Placement(origin, axis, angle)
            self.options._options["plane"] = geo.plane.BluemiraPlane()
        elif plane == "xz":
            # Base.Placement(origin, axis, angle)
            self.options._options["plane"] = geo.plane.BluemiraPlane(
                axis=(1.0, 0.0, 0.0), angle=-90.0
            )
        elif plane == "zy":
            # Base.Placement(origin, axis, angle)
            self.options._options["plane"] = geo.plane.BluemiraPlane(
                axis=(0.0, 1.0, 0.0), angle=90.0
            )
        elif isinstance(plane, geo.plane.BluemiraPlane):
            self.options._options["plane"] = plane
        else:
            DisplayError(f"{plane} is not a valid plane")

    @abstractmethod
    def _check_obj(self, obj):
        """Internal function that check if obj is an instance of the correct class"""
        pass

    @abstractmethod
    def _check_options(self):
        """Internal function that check if it is needed to plot something"""
        pass

    def initialize_plot_2d(self, ax=None):
        """Initialize the plot environment"""
        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot()
        else:
            self.ax = ax

    def show_plot_2d(self):
        """Function to show a plot"""
        self.ax.set_aspect("equal")
        plt.show(block=True)

    @abstractmethod
    def _populate_data(self, obj, *args, **kwargs):
        """
        Internal function that makes the plot. It fills self._data and
        self._data_to_plot
        """
        pass

    @abstractmethod
    def _make_plot_2d(self, *args, **kwargs):
        """
        Internal function that makes the plot. It should use self._data and
        self._data_to_plot, so _populate_data should be called before.
        """
        pass

    def plot_2d(self, obj, ax=None, show: bool = True, *args, **kwargs):
        """2D plotting method"""
        self._check_obj(obj)

        if not self._check_options():
            self.ax = ax
        else:
            self.initialize_plot_2d(ax)
            self._populate_data(obj, *args, **kwargs)
            self._make_plot_2d(*args, **kwargs)

            if show:
                self.show_plot_2d()
        return self.ax

    ################################################
    #                 3D functions                 #
    ################################################
    def initialize_plot_3d(self, ax=None):
        """Initialize the plot environment"""
        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot(projection="3d")
        else:
            self.ax = ax

    def show_plot_3d(self):
        """Function to show a plot"""
        self.ax.set_aspect("auto")
        plt.show(block=True)

    @abstractmethod
    def _make_plot_3d(self, *args, **kwargs):
        """Internal function that makes the plot. It should use self._data and
        self._data_to_plot, so _populate_data should be called before.
        """
        pass

    def plot_3d(self, obj, ax=None, show: bool = True, *args, **kwargs):
        """3D plotting method"""
        self._check_obj(obj)

        if not self._check_options():
            self.ax = ax
        else:
            self.initialize_plot_3d()
            # this function can be common to 2D and 3D plot
            # self._data is used for 3D plot
            # self._data_to_plot is used for 2D plot
            # TODO: probably better to rename self._data_to_plot into
            #  self._projected_data or self._data2d
            self._populate_data(obj, *args, **kwargs)
            self._make_plot_3d(*args, **kwargs)

            if show:
                self.show_plot_3d()

        return self.ax


class PointsPlotter(BasePlotter):
    """
    Base utility plotting class for points
    """

    def _check_obj(self, obj):
        # TODO: create a function that checks if the obj is a cloud of 3D or 2D points
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if not self.options.show_points:
            return False
        return True

    def _populate_data(self, points, *args, **kwargs):
        self._data = points
        # apply rotation matrix given by options['plane']
        self.rot = self.options._options["plane"].to_matrix().T
        self.temp_data = np.c_[self._data, np.ones(len(self._data))]
        self._data_to_plot = self.temp_data.dot(self.rot).T
        self._data_to_plot = self._data_to_plot[0:2]

    def _make_plot_2d(self, *args, **kwargs):
        if self.options.show_points:
            self.ax.scatter(*self._data_to_plot, **self.options._options["poptions"])

    def _make_plot_3d(self, *args, **kwargs):
        if self.options.show_points:
            self.ax.scatter(*self._data.T, **self.options._options["poptions"])


class WirePlotter(BasePlotter):
    """
    Base utility plotting class for bluemira wires
    """

    def _check_obj(self, obj):
        if not isinstance(obj, geo.wire.BluemiraWire):
            raise ValueError(f"{obj} must be a BluemiraWire")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if not self.options.show_points and not self.options.show_wires:
            return False

        return True

    def _populate_data(self, wire, *args, **kwargs):
        self._pplotter = PointsPlotter(self.options)
        new_wire = wire.deepcopy()
        # # change of plane integrated in PointsPlotter2D. Not necessary here.
        # new_wire.change_plane(self.options._options['plane'])
        pointsw = new_wire.discretize(
            ndiscr=self.options._options["ndiscr"],
            byedges=self.options._options["byedges"],
        )
        self._pplotter._populate_data(pointsw)
        self._data = pointsw
        self._data_to_plot = self._pplotter._data_to_plot

    def _make_plot_2d(self):
        if self.options.show_wires:
            self.ax.plot(*self._data_to_plot, **self.options._options["woptions"])

        if self.options.show_points:
            self._pplotter.ax = self.ax
            self._pplotter._make_plot_2d()

    def _make_plot_3d(self, *args, **kwargs):
        if self.options.show_wires:
            self.ax.plot(*self._data.T, **self.options._options["woptions"])

        if self.options.show_points:
            self._pplotter.ax = self.ax
            self._pplotter._make_plot_3d()


class FacePlotter(BasePlotter):
    """Base utility plotting class for bluemira faces"""

    def _check_obj(self, obj):
        if not isinstance(obj, geo.face.BluemiraFace):
            raise ValueError(f"{obj} must be a BluemiraFace")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if (
            not self.options.show_points
            and not self.options.show_wires
            and not self.options.show_faces
        ):
            return False

        return True

    def _populate_data(self, face, *args, **kwargs):
        self._data = []
        self._wplotters = []
        # TODO: the for must be done using face._shape.Wires because FreeCAD
        #  re-orient the Wires in the correct way for display. Find another way to do
        #  it (maybe adding this function to the freecadapi.
        for w in face._shape.Wires:
            boundary = geo.wire.BluemiraWire(w)
            wplotter = WirePlotter(self.options)
            self._wplotters.append(wplotter)
            wplotter._populate_data(boundary)
            self._data.append(wplotter._data)

        self._data_to_plot = [[], []]
        for w in self._wplotters:
            self._data_to_plot[0] += w._data_to_plot[0].tolist() + [None]
            self._data_to_plot[1] += w._data_to_plot[1].tolist() + [None]

    def _make_plot_2d(self):
        if self.options.show_faces:
            self.ax.fill(*self._data_to_plot, **self.options._options["foptions"])

        for w in self._wplotters:
            w.ax = self.ax
            w._make_plot_2d()

    def _make_plot_3d(self, *args, **kwargs):
        """
        Internal function that makes the plot. It should use self._data and
        self._data_to_plot, so _populate_data should be called before.
        """
        # TODO: to be implemented
        pass


def _validate_plot_inputs(parts, options, default_options):
    """
    Validate the lists of parts and options, applying some default options
    """
    if not isinstance(parts, list):
        parts = [parts]

    if options is None:
        options = [default_options] * len(parts)
    elif not isinstance(options, list):
        options = [options] * len(parts)

    if len(options) != len(parts):
        raise DisplayError(
            "If options for plot are provided then there must be as many options as "
            "there are parts to plot."
        )
    return parts, options


def _get_plotter_class(part):
    """
    Get the plotting class for a BluemiraGeo object.
    """
    if isinstance(part, geo.wire.BluemiraWire):
        plot_class = WirePlotter
    elif isinstance(part, geo.face.BluemiraFace):
        plot_class = FacePlotter
    else:
        raise DisplayError(
            f"{part} object cannot be plotted. No Plotter available for {type(part)}"
        )
    return plot_class


def plot_2d(
    parts: Union[geo.base.BluemiraGeo, List[geo.base.BluemiraGeo]],
    options: Optional[Union[_Plot2DOptions, List[_Plot2DOptions]]] = None,
    ax=None,
    show: bool = True,
):
    """
    The implementation of the display API for FreeCAD parts.

    Parameters
    ----------
    parts: Union[Part.Shape, List[Part.Shape]]
        The parts to display.
    options: Optional[Union[Plot2DOptions, List[Plot2DOptions]]]
        The options to use to display the parts.
    ax: Optional[Axes]
        The axes onto which to plot
    show: bool
        Whether or not to show the plot immediately (default=True)
    """
    parts, options = _validate_plot_inputs(parts, options, _Plot2DOptions())

    for part, option in zip(parts, options):
        plot_class = _get_plotter_class(part)
        plotter = plot_class(option)
        ax = plotter.plot_2d(part, ax, False)

    if show:
        plotter.show_plot_2d()

    return ax


def plot_3d(
    parts: Union[geo.base.BluemiraGeo, List[geo.base.BluemiraGeo]],
    options: Optional[Union[_Plot3DOptions, List[_Plot3DOptions]]] = None,
    ax=None,
    show: bool = False,
):
    """
    The implementation of the display API for BluemiraGeo parts.

    Parameters
    ----------
    parts: Union[Part.Shape, List[Part.Shape]]
        The parts to display.
    options: Optional[Union[Plot3DOptions, List[Plot3Options]]]
        The options to use to display the parts.
    ax: Optional[Axes]
        The axes onto which to plot
    show: bool
        Whether or not to show the plot immediately (default=True)
    """
    parts, options = _validate_plot_inputs(parts, options, _Plot3DOptions())

    for part, option in zip(parts, options):
        plot_class = _get_plotter_class(part)
        plotter = plot_class(option)
        ax = plotter.plot_3d(part, ax, False)

    if show:
        plotter.show_plot_3d()

    return ax

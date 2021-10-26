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

from typing import Optional, Union, List, Dict

import bluemira.geometry as geo
from . import display
from .error import DisplayError

import copy
import numpy as np

DEFAULT = {}
# flags to enable points, wires, and faces plot
DEFAULT["flag_points"] = True
DEFAULT["flag_wires"] = True
DEFAULT["flag_faces"] = True
# matplotlib set of options to plot points, wires, and faces. If an empty dictionary
# is specified, the default color plot of matplotlib is used.
DEFAULT["poptions"] = {"s": 10, "facecolors": "red", "edgecolors": "black"}
DEFAULT["woptions"] = {"color": "black", "linewidth": "0.5"}
DEFAULT["foptions"] = {"color": "blue"}
# projection plane
DEFAULT["plane"] = "xz"
# palette
# Todo: it's use is still in progress
DEFAULT["palette"] = None
# discretization properties for plotting wires (and faces)
DEFAULT["ndiscr"] = 100
DEFAULT["byedges"] = True


# Note: This class cannot be an instance of dataclasses 'cause it fails when inserting
# a field that is a Base.Placement (or something that contains a Base.Placement). The
# given error is "TypeError: can't pickle Base.Placement objects"
class MatplotlibOptions(display.Plot2DOptions):
    """ The options that are available for plotting objects in 3D
    Parameters
    ----------
    flag_points: bool
        If true, points are plotted. By default True.
    flag_wires: bool
        if true, wires are plotted. By default True.
    flag_faces: bool
        if true, faces are plotted. By default True.
    poptions: Dict
        dictionary with matplotlib options for points. By default  {"s": 10,
        "facecolors": "blue", "edgecolors": "black"}
    woptions: Dict
        dictionary with matplotlib options for wires. By default {"color": "black",
        "linewidth": "0.5"}
    foptions: Dict
        dictionary with matplotlib options for faces. By default {"color": "red"}
    plane: [str, Plane]
        The plane on which the object is projected for plotting. As string, possible
        options are "xy", "xz", "zy". By default 'xz'.
    palette:
        palette
    """

    def __init__(self, **kwargs):
        self._options = copy.deepcopy(DEFAULT)
        if kwargs:
            for k in kwargs:
                if k in self.options:
                    self.options[k] = kwargs[k]
        # Todo: in this way class attributes are not seen till runtime. Not sure if
        #  this should be changed manually declaring all the attributes.
        for k in self._options:
            setattr(self, k, self._options[k])


# Note: when plotting points, it can happen that markers are not centered properly as
# described in https://github.com/matplotlib/matplotlib/issues/11836
class BasePlotter2D(ABC):
    """
    Base utility plotting class
    """

    def __init__(self, options: Optional[MatplotlibOptions] = None, *args, **kwargs):
        self._data = []  # data passed to the BasePlotter2D
        self._data_to_plot = []  # real data that is plotted
        self.ax = None
        self.options = options
        self.set_plane(self.options.plane)
        if kwargs:
            for k in kwargs:
                if k in self.options.asdict():
                    # Todo: probably it could be better to store the plane as a
                    #  dictionary or a tuple (e.g. (base, direction, angle) and create
                    #  the real plane in "set_plane". In this way we could use a
                    #  dataclass for MatplotlibOptions.
                    if k == "plane":
                        self.set_plane(kwargs[k])
                    else:
                        setattr(self.options, k, kwargs[k])

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, val: MatplotlibOptions) -> None:
        self._options = MatplotlibOptions() if val is None else val

    def set_plane(self, plane):
        """Set the plotting plane"""
        if plane == "xy":
            # Base.Placement(origin, axis, angle)
            self.options.plane = geo.plane.BluemiraPlane()
        elif plane == "xz":
            # Base.Placement(origin, axis, angle)
            self.options.plane = geo.plane.BluemiraPlane(axis=(1.0, 0.0, 0.0),
                                                        angle=-90.0)
        elif plane == "zy":
            # Base.Placement(origin, axis, angle)
            self.options.plane = geo.plane.BluemiraPlane(axis=(0.0, 1.0, 0.0),
                                                         angle=90.0)
        elif isinstance(plane, geo.plane.BluemiraPlane):
            self.options.plane = plane
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

    def initialize_plot(self, ax=None):
        """Initialize the plot environment"""
        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot()
        else:
            self.ax = ax

    def show_plot(self, aspect: str = "equal", block=True):
        """Function to show a plot"""
        plt.gca().set_aspect(aspect)
        plt.show(block=block)

    @abstractmethod
    def _populate_data(self, obj, *args, **kwargs):
        """Internal function that makes the plot. It fills self._data and
        self._data_to_plot
        """
        pass

    @abstractmethod
    def _make_plot(self, *args, **kwargs):
        """Internal function that makes the plot. It should use self._data and
        self._data_to_plot, so _populate_data should be called before.
        """
        pass

    def plot2d(self, obj, ax=None, show: bool = False, block: bool = False, *args,
             **kwargs):
        """2D plotting method"""
        self._check_obj(obj)

        if not self._check_options():
            self.ax = ax
        else:
            self.initialize_plot(ax)
            self._populate_data(obj, *args, **kwargs)
            self._make_plot(*args, **kwargs)

            if show:
                self.show_plot(block=block)
        return self.ax

    def __call__(
        self, obj, ax=None, show: bool = False, block: bool = False, *args, **kwargs
    ):
        return self.plot2d(obj, ax=ax, show=show, block=block, *args, **kwargs)

class PointsPlotter2D(BasePlotter2D):
    """
    Base utility plotting class for points
    """

    def _check_obj(self, obj):
        # Todo: create a function that checks if the obj is a cloud of 3D or 2D points
        return True

    def _check_options(self):
        print(self.options.asdict())
        # Check if nothing has to be plotted
        if not self.options.flag_points:
            return False
        return True

    def _populate_data(self, points, *args, **kwargs):
        self._data = points.tolist() if not isinstance(points, list) else points
        #apply rotation matrix given by options.plane
        self.rot = self.options.plane.to_matrix().T
        self.temp_data = np.array(self._data)
        self.temp_data = np.c_[self.temp_data, np.ones(len(self.temp_data))]
        self._data_to_plot = self.temp_data.dot(self.rot).T
        self._data_to_plot = self._data_to_plot[0:2]

    def _make_plot(self, *args, **kwargs):
        if self.options.flag_points:
            self.ax.scatter(*self._data_to_plot, **self.options.poptions)


class WirePlotter2D(BasePlotter2D):
    """
    Base utility plotting class for bluemira wires
    """

    def _check_obj(self, obj):
        if not isinstance(obj, geo.wire.BluemiraWire):
            raise ValueError(f"{obj} must be a BluemiraWire")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if not self.options.flag_points and not self.options.flag_wires:
            return False

        return True

    def _populate_data(self, wire, *args, **kwargs):
        self._pplotter = PointsPlotter2D(self.options)
        new_wire = wire.deepcopy()
        # # change of plane integrated in PointsPlotter2D. Not necessary here.
        # new_wire.change_plane(self.options.plane)
        pointsw = new_wire.discretize(ndiscr=self.options.ndiscr,
                                      byedges=self.options.byedges)
        self._pplotter._populate_data(pointsw)
        self._data = pointsw
        self._data_to_plot = self._pplotter._data_to_plot

    def _make_plot(self):
        if self.options.flag_wires:
            self.ax.plot(*self._data_to_plot, **self.options.woptions)

        if self.options.flag_points:
            self._pplotter.ax = self.ax
            self._pplotter._make_plot()


class FacePlotter2D(BasePlotter2D):
    """Base utility plotting class for bluemira faces"""

    def _check_obj(self, obj):
        if not isinstance(obj, geo.face.BluemiraFace):
            raise ValueError(f"{obj} must be a BluemiraFace")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if (
            not self.options.flag_points
            and not self.options.flag_wires
            and not self.options.flag_faces
        ):
            return False

        return True

    def _populate_data(self, face, *args, **kwargs):
        self._data = []
        self._wplooters = []
        # Todo: the for must be done using face._shape.Wires because FreeCAD
        #  re-orient the Wires in the correct way for display. Find another way to do
        #  it (maybe adding this function to the freecadapi.
        for w in face._shape.Wires:
            boundary = geo.wire.BluemiraWire(w)
            wplotter = WirePlotter2D(self.options)
            self._wplooters.append(wplotter)
            wplotter._populate_data(boundary)

            self._data.append(wplotter._data)

        self._data_to_plot = [[], []]
        for w in self._wplooters:
            self._data_to_plot[0] += w._data_to_plot[0].tolist() + [None]
            self._data_to_plot[1] += w._data_to_plot[1].tolist() + [None]

    def _make_plot(self):
        if self.options.flag_faces:
            self.ax.fill(*self._data_to_plot, **self.options.foptions)

        for w in self._wplooters:
            w.ax = self.ax
            w._make_plot()


class FaceCompoundPlotter2D(FacePlotter2D):
    """
    Base utility plotting class for shape compounds.
    """

    # Todo: this is just a test class. A strategy for filling faces with a color
    #  defined by a palette has still not been defined.
    def _check_obj(self, objs):
        """Check if objects in objs are of the correct type for this class"""
        self._acceptable_classes = [geo.face.BluemiraFace]
        if not hasattr(objs, "__len__"):
            objs = [objs]
        check = False
        for c in self._acceptable_classes:
            check = check or (all(isinstance(o, c) for o in objs))
            if check:
                return objs
        raise TypeError(
            f"Only {self._boundary_classes} objects can be used for {self.__class__}"
        )

    def _populate_data(self, objs, *args, **kwargs):
        self._fplotters = []
        for id, obj in enumerate(objs):
            temp_fplotter = FacePlotter2D(self.options)
            temp_fplotter._populate_data(obj)
            self._data += [temp_fplotter._data]
            self._data_to_plot += [temp_fplotter._data_to_plot]
            self._fplotters.append(temp_fplotter)

    def _make_plot(self):
        if "palette" in self.options.asdict():
            import seaborn as sns
            palette = sns.color_palette(self.options.palette, len(self._fplotters))
        else:
            palette = self.options.foptions["color"]

        for id, fplotter in enumerate(self._fplotters):
            fplotter.ax = self.ax
            fplotter.options.foptions['color'] = palette[id]
            fplotter._make_plot()


def plot2d(
    parts: Union[geo.base.BluemiraGeo, List[geo.base.BluemiraGeo]],
    options: Optional[Union[MatplotlibOptions, List[MatplotlibOptions]]] = None,
    ax = None, show: bool = False, block: bool = True,
):
    """
    The implementation of the display API for FreeCAD parts.
    Parameters
    ----------
    parts: Union[Part.Shape, List[Part.Shape]]
        The parts to display.
    options: Optional[Union[Plot2DOptions, List[Plot2DOptions]]]
        The options to use to display the parts.
    """
    if not isinstance(parts, list):
        parts = [parts]

    if options is None:
        options = [MatplotlibOptions()] * len(parts)
    elif not isinstance(options, list):
        options = [options] * len(parts)

    if len(options) != len(parts):
        raise DisplayError(
            "If options for plot are provided then there must be as many options as "
            "there are parts to plot."
        )

    for part, option in zip(parts, options):
        if isinstance(part, geo.wire.BluemiraWire):
            plotter = WirePlotter2D(option)
        elif isinstance(part, geo.face.BluemiraFace):
            plotter = FacePlotter2D(option)
        else:
            raise DisplayError(
                f"{part} object cannot be plotted. No Plotter available for {type(part)}"
            )
        ax = plotter.plot2d(part, ax, False, False)

    if show:
        plotter.show_plot(block=block)

    return ax

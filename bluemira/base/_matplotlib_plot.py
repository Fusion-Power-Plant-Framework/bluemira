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
from abc import abstractmethod
import matplotlib.pyplot as plt

from typing import Optional, Union, List, Dict

import bluemira.geometry as geo
from . import display
from .error import DisplayError

import dataclasses
from dataclasses import field

DEFAULT = {}
DEFAULT["flags"] = {"points": True, "wires": True, "faces": True}
DEFAULT["poptions"] = {"s": 10, "facecolors": "blue", "edgecolors": "black"}
DEFAULT["woptions"] = {"color": "black", "linewidth": "0.5"}
DEFAULT["foptions"] = {"color": "red"}
DEFAULT["plane"] = "xy"
DEFAULT["palette"] = None
DEFAULT["ndiscr"] = 100
DEFAULT["byedges"] = True


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
        options are "xy", "xz", "yz". By default 'xz'.
    palette:
        palette
    """

    def __init__(self, **kwargs):
        self._options = DEFAULT.copy()
        if kwargs:
            for k in kwargs:
                if k in self.options:
                    if k == "plane":
                        self.set_plane(kwargs[k])
                    else:
                        self.options[k] = kwargs[k]
        for k in self._options:
            setattr(self, k, self._options[k])

# Note: when plotting points, it can happen that markers are not centered properly as
# described in https://github.com/matplotlib/matplotlib/issues/11836


class BasePlotter():
    """
    Base utility plotting class
    """

    def __init__(self, options: Optional[MatplotlibOptions] = None, *args, **kwargs):
        self._data = []  # data passed to the BasePlotter
        self._data_to_plot = []  # real data that is plotted
        self.ax = None
        self.options = options
        self.set_plane(self.options.plane)
        if kwargs:
            for k in kwargs:
                if k in self.options.asdict():
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

    @property
    def flag_points(self):
        """Set the flag to plot points"""
        return self.options.flags["points"]

    @flag_points.setter
    def flag_points(self, value):
        self.options.flags["points"] = value

    @property
    def flag_wires(self):
        """Set the flag to plot wires"""
        return self.options.flags["wires"]

    @flag_wires.setter
    def flag_wires(self, value):
        self.options.flags["wires"] = value

    @property
    def flag_faces(self):
        """Set the flag to plot faces"""
        return self.options.flags["faces"]

    @flag_faces.setter
    def flag_faces(self, value):
        self.options.flags["faces"] = value

    @property
    def poptions(self):
        """Plot options for points"""
        return self.options.poptions

    @property
    def woptions(self):
        """Plot options for wires"""
        return self.options.woptions

    @property
    def foptions(self):
        """Plot options for faces"""
        return self.options.foptions

    def change_poptions(self, value):
        """Function to change the plot options for points
        Parameters
        ----------
        value: dict or tuple
            If dict, all the points plot options are replaced by the new dict
            if tuple:(plot_key, value), the specified plot_key is added/replaced
            into the plot options dictionary with the specified value.
        """
        if isinstance(value, dict):
            self.options.poptions = value
        elif isinstance(value, tuple):
            self.options.poptions[value[0]] = value[1]
        else:
            raise ValueError(f"{value} is not a valid dict or tuple(key, value)")

    def change_woptions(self, value: [dict, tuple]):
        """Function to change the plot options for wires
        Parameters
        ----------
        value: dict or tuple
            If dict, all the wires plot options are replaced by the new dict
            if tuple:(plot_key, value), the specified plot_key is added/replaced
            into the plot options dictionary with the specified value.
        """
        if isinstance(value, dict):
            self.options.woptions = value
        elif isinstance(value, tuple):
            self.options.woptions[value[0]] = value[1]
        else:
            raise ValueError(f"{value} is not a valid dict or tuple(key, value)")

    def change_foptions(self, value: [dict, tuple]):
        """Function to change the plot options for faces
        Parameters
        ----------
        value: dict or tuple
            If dict, all the faces plot options are replaced by the new dict
            if tuple:(plot_key, value), the specified plot_key is added/replaced
            into the plot options dictionary with the specified value.
        """
        if isinstance(value, dict):
            self.options.foptions = value
        elif isinstance(value, tuple):
            self.options.foptions[value[0]] = value[1]
        else:
            raise ValueError(f"{value} is not a valid dict or tuple(key, value)")

    def set_plane(self, plane):
        """Set the plotting plane"""
        if plane == "xy":
            # Base.Placement(origin, axis, angle)
            self.options.plane = geo.plane.BluemiraPlane()
        elif plane == "xz":
            # Base.Placement(origin, axis, angle)
            self.options.plane = geo.plane.BluemiraPlane(axis=(1.0, 0.0, 0.0),
                                                        angle=-90.0)
        elif plane == "yz":
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
    def _make_plot(self, obj, *args, **kwargs):
        """Internal function that makes the plot. It fills self._data and
        self._data_to_plot
        """
        pass

    def plot(self, obj, ax=None, show: bool = False, block: bool = False, *args,
             **kwargs):
        """2D plotting method"""
        self._check_obj(obj)

        if not self._check_options():
            self.ax = ax
        else:
            self.initialize_plot(ax)

            self._make_plot(obj, *args, **kwargs)

            if show:
                self.show_plot(block=block)
        return self.ax

    def __call__(
        self, obj, ax=None, show: bool = False, block: bool = False, *args, **kwargs
    ):
        return self.plot(obj, ax=ax, show=show, block=block, *args, **kwargs)

class PointsPlotter(BasePlotter):
    """
    Base utility plotting class for points
    """

    def _check_obj(self, obj):
        # Todo: create a function that checks if the obj is a cloud of 3D or 2D points
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if not self.flag_points:
            return False
        # check if no options have been specified
        if not self.poptions:
            return False
        return True

    def _make_plot(self, points, *args, **kwargs):
        self._data = points.tolist() if not isinstance(points, list) else points
        self._data_to_plot = points[0:2]
        self.ax.scatter(*self._data_to_plot, **self.poptions)


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
        if not self.flag_points and not self.flag_wires:
            return False

        # check if no options have been specified
        if not self.poptions and not self.woptions:
            return False

        return True

    def _make_plot(self, wire, ndiscr: int = 100, byedges: bool = True):
        new_wire = wire.deepcopy()
        new_wire.change_plane(self.options.plane)
        pointsw = new_wire.discretize(ndiscr=ndiscr, byedges=byedges).T
        self._data = pointsw.tolist()
        self._data_to_plot = pointsw[0:2]

        if self.flag_wires:
            self.ax.plot(*self._data_to_plot, **self.options.woptions)

        if self.flag_points:
            pplotter = PointsPlotter(self.options)
            self.ax = pplotter(self._data_to_plot, self.ax, show=False)


class FacePlotter(BasePlotter):
    """Base utility plotting class for bluemira faces"""

    def _check_obj(self, obj):
        if not isinstance(obj, geo.face.BluemiraFace):
            raise ValueError(f"{obj} must be a BluemiraFace")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if not self.flag_points and not self.flag_wires and not self.flag_faces:
            return False

        # check if no options have been specified
        if (
            not self.poptions
            and not self.woptions
            and not self.foptions
        ):
            return False

        return True

    def _make_plot(self, face, ndiscr: int = 100, byedges: bool = True):
        self._data = [[], [], []]

        # Todo: the for must to be done using face._shape.Wires because FreeCAD
        #  re-orient the Wires in the correct way for display. Find another way to do
        #  it (maybe adding this function to the freecadapi.
        for w in face._shape.Wires:
            boundary = geo.wire.BluemiraWire(w)
            wplotter = WirePlotter(self.options)
            if not self.flag_wires and not self.flag_points:
                wplotter._make_data(boundary, ndiscr, byedges)
            else:
                wplotter(
                    boundary, ax=self.ax, show=False, ndiscr=ndiscr, byedges=byedges
                )

                self._data[0] += wplotter._data[0][::-1] + [None]
                self._data[1] += wplotter._data[1][::-1] + [None]
                self._data[2] += wplotter._data[2][::-1] + [None]

        self._data[0] = self._data[0][:-1]
        self._data[1] = self._data[1][:-1]
        self._data[2] = self._data[2][:-1]

        self._data_to_plot = self._data[0:2]

        if self.flag_faces and self.foptions:
            plt.fill(*self._data_to_plot, **self.foptions)


class FaceCompoundPlotter(FacePlotter):
    """
    Base utility plotting class for shape compounds
    """

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

    def _make_plot(self, objs, ndiscr, byedges):
        if "palette" in self.options.asdict():
            import seaborn as sns

            palette = sns.color_palette(self.options.palette, len(objs))
            print(f"palette: {palette}")
        else:
            palette = self.options.foptions["color"]

        for id, obj in enumerate(objs):
            temp_fplotter = FacePlotter(self.options)
            temp_fplotter.change_foptions(("color", palette[id]))
            print(temp_fplotter.foptions)
            self.ax = temp_fplotter(
                obj, ax=self.ax, show=False, ndiscr=ndiscr, byedges=byedges
            )
            self._data += [temp_fplotter._data]
            self._data_to_plot += [temp_fplotter._data_to_plot]


def plot2d(
    parts: Union[geo.base.BluemiraGeo, List[geo.base.BluemiraGeo]],
    options: Optional[Union[MatplotlibOptions, List[MatplotlibOptions]]] = None,
    ax = None, show: bool = True, block: bool = True,
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
            plotter = WirePlotter(option)
        elif isinstance(part, geo.face.BluemiraFace):
            plotter = FacePlotter(option)
        else:
            raise DisplayError(
                f"{part} object cannot be plotted. No Plotter available for {type(part)}"
            )
        ax = plotter.plot(part, ax, False, False, option.ndiscr, option.byedges)

    if show:
        plotter.show_plot(block=block)

    return ax
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
"""Plotter module"""

from abc import ABC, abstractmethod

# matplotlib import
import matplotlib.pyplot as plt

# bluemira geometry import
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane

from .error import PlottingError

DEFAULT = {}
DEFAULT["plot_flag"] = {"poptions": True, "woptions": True, "foptions": True}
DEFAULT["poptions"] = {"s": 10, "facecolors": "blue", "edgecolors": "black"}
DEFAULT["woptions"] = {"color": "black", "linewidth": "0.5"}
DEFAULT["foptions"] = {"color": "red"}
DEFAULT["plane"] = "xz"


class Plottable:
    """Plottable class"""

    def __init__(self, plotter=None, obj=None):
        self._plotter = plotter
        self._plottable_object = obj

    def plot(self, **kwargs):
        """Plotting method"""
        self._plotter.plot(self._plottable_object, **kwargs)


class BasePlotter(ABC):
    """
    Base utility plotting class
    """

    def __init__(self, **kwargs):
        self.data = []  # data passed to the BasePlotter
        self.plot_data = []  # real data that is plotted
        if kwargs:
            for k in kwargs:
                if k in self.options:
                    if k == "plane":
                        self.set_plane(kwargs[k])
                    else:
                        self.options[k] = kwargs[k]

    @property
    def plot_points(self):
        return self.options["plot_flag"]["poptions"]

    @plot_points.setter
    def plot_points(self, value):
        self.options["plot_flag"]["poptions"] = value

    @property
    def plot_wires(self):
        return self.options["plot_flag"]["woptions"]

    @plot_wires.setter
    def plot_wires(self, value):
        self.options["plot_flag"]["woptions"] = value

    @property
    def plot_faces(self):
        return self.options["plot_flag"]["foptions"]

    @plot_faces.setter
    def plot_faces(self, value):
        self.options["plot_flag"]["foptions"] = value

    @property
    def poptions(self):
        return self.options["poptions"]

    def change_poptions(self, value):
        if isinstance(value, dict):
            self.options["poptions"] = value
        elif isinstance(value, tuple):
            self.options["poptions"][value[0]] = value[1]
        else:
            raise ValueError(f"{value} is not a valid dict or tuple(key, value)")

    def set_plane(self, plane):
        """Set the plotting plane"""
        if plane == "xy":
            # Base.Placement(origin, axis, angle)
            self.options["plane"] = BluemiraPlane()
        elif plane == "xz":
            # Base.Placement(origin, axis, angle)
            self.options["plane"] = BluemiraPlane(axis=(1.0, 0.0, 0.0), angle=-90.0)
        elif plane == "yz":
            # Base.Placement(origin, axis, angle)
            self.options["plane"] = BluemiraPlane(axis=(0.0, 1.0, 0.0), angle=90.0)
        elif isinstance(plane, BluemiraPlane):
            self.options["plane"] = plane
            pass
        else:
            PlottingError(f"{plane} is not a valid plane")

    @abstractmethod
    def _check_obj(self, obj):
        """Internal function that check if obj is an instance of the correct class"""
        pass

    @abstractmethod
    def _check_options(self):
        """Internal function that check if it is needed to plot something"""
        pass

    @abstractmethod
    def _make_data(self, obj, *args, **kwargs):
        """Internal function that initialize self.data and self.plot_data"""
        pass

    @abstractmethod
    def _make_plot(self):
        """Internal function that makes the plot"""
        pass

    def __call__(
        self, obj, ax=None, show: bool = False, block: bool = False, *args, **kwargs
    ):
        """2D plotting method"""
        self._check_obj(obj)

        if not self._check_options():
            self.ax = ax
        else:
            if ax is None:
                fig = plt.figure()
                self.ax = fig.add_subplot()
            else:
                self.ax = ax

            self._make_data(obj, *args, **kwargs)
            self._make_plot()

            if show:
                plt.gca().set_aspect("equal")
                plt.show(block=block)
        return self.ax


class PointsPlotter(BasePlotter):
    """
    Base utility plotting class for points
    """

    def __init__(self, ax=None, **kwargs):
        self.options = DEFAULT
        super().__init__(**{**self.options, **kwargs})

    def _check_obj(self, obj):
        # Todo: create a function that ckeck if the obj is a cloud of 3D or 2D points
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if not self.plot_points:
            return False
        # check if no options have been specified
        if not self.options["poptions"]:
            return False
        return True

    def _make_data(self, points, *args, **kwargs):
        self.data = points.tolist()
        self.plot_data = points[0:2]

    def _make_plot(self):
        self.ax.scatter(*self.plot_data, **self.options["poptions"])


class WirePlotter(BasePlotter):
    """
    Base utility plotting class for bluemira wires
    """

    def __init__(self, **kwargs):
        self.options = DEFAULT
        super().__init__(**{**self.options, **kwargs})

    def _check_obj(self, obj):
        if not isinstance(obj, BluemiraWire):
            raise ValueError(f"{obj} must be a BluemiraWire")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if not self.plot_points and not self.plot_wires:
            return False

        # check if no options have been specified
        if not self.options["poptions"] and not self.options["woptions"]:
            return False

        return True

    def _make_data(self, wire, ndiscr, byedges):
        new_wire = wire.deepcopy()
        new_wire.change_plane(self.options["plane"])
        pointsw = new_wire.discretize(ndiscr=ndiscr, byedges=byedges).T
        self.data = pointsw.tolist()
        self.plot_data = pointsw[0:2]

    def _make_plot(self):
        if self.plot_wires:
            self.ax.plot(*self.plot_data, **self.options["woptions"])

        if self.plot_points:
            pplotter = PointsPlotter(**self.options)
            self.ax = pplotter(self.plot_data, self.ax, show=False)


class FacePlotter(BasePlotter):
    """
    Base utility plotting class for bluemira faces
    """

    def __init__(self, **kwargs):
        self.options = DEFAULT
        super().__init__(**{**self.options, **kwargs})

    def _check_obj(self, obj):
        if not isinstance(obj, BluemiraFace):
            raise ValueError(f"{obj} must be a BluemiraFace")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if not self.plot_points and not self.plot_wires and not self.plot_faces:
            return False

        # check if no options have been specified
        if (
            not self.options["poptions"]
            and not self.options["woptions"]
            and not self.options["foptions"]
        ):
            return False

        return True

    def _make_data(self, face, ndiscr, byedges):
        self.data = [[], [], []]
        j = 0
        for w in face._shape.Wires:
            j = j + 1
            boundary = BluemiraWire(w)
            wplotter = WirePlotter(**self.options)
            if not self.plot_wires and not self.plot_points:
                wplotter._make_data(boundary, ndiscr, byedges)
            else:
                wplotter(
                    boundary, ax=self.ax, show=False, ndiscr=ndiscr, byedges=byedges
                )

            # Todo: it seems that discretize and discretize_by_edges produce a
            #  different output in case all the Edges of a Wire are reversed. To
            #  be checked.
            # The behaviour above would not allow the plot of a filled face
            # since the internal holes would be considered in the same direction
            # of the external one. Solved a trick, but to be adjusted.
            if j == 1:
                self.data[0] += wplotter.data[0][::-1] + [None]
                self.data[1] += wplotter.data[1][::-1] + [None]
                self.data[2] += wplotter.data[2][::-1] + [None]
            else:
                self.data[0] += wplotter.data[0] + [None]
                self.data[1] += wplotter.data[1] + [None]
                self.data[2] += wplotter.data[2] + [None]

        self.plot_data = self.data[0:2]

    def _make_plot(self):
        if self.plot_faces and self.options["foptions"]:
            plt.fill(*self.plot_data, **self.options["foptions"])

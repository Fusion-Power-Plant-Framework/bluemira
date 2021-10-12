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

from typing import Union

# matplotlib import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


class Plot3D(Axes3D):
    """
    Cheap and cheerful
    """

    def __init__(self):
        fig = plt.figure(figsize=[14, 14])
        super().__init__(fig)
        # \n to offset labels from axes
        self.set_xlabel("\n\nx [m]")
        self.set_ylabel("\n\ny [m]")
        self.set_zlabel("\n\nz [m]")


class Plottable:
    """Plottable class"""

    def __init__(self, plotter=None, object=None):
        self._plotter = plotter
        self._plottable_object = object

    def plot(self, **kwargs):
        """Plotting method"""
        self._plotter.plot(self._plottable_object, **kwargs)


class BasePlotter(ABC):
    """
    Base utility plotting class
    """

    def __init__(self, **kwargs):
        self.data = []
        if kwargs:
            for k in kwargs:
                if k in self.options:
                    if k == 'plane':
                        self.set_plane(kwargs[k])
                    else:
                        self.options[k] = kwargs[k]

    @property
    def plot_points(self):
        return self.options['plot_flag']['poptions']

    @plot_points.setter
    def plot_points(self, value):
        self.options['plot_flag']['poptions'] = value

    @property
    def plot_wires(self):
        return self.options['plot_flag']['woptions']

    @plot_points.setter
    def plot_wires(self, value):
        self.options['plot_flag']['woptions'] = value

    @property
    def plot_faces(self):
        return self.options['plot_flag']['foptions']

    @plot_points.setter
    def plot_faces(self, value):
        self.options['plot_flag']['foptions'] = value

    def set_plane(self, plane):
        """Set the plotting plane"""
        if plane == "xy":
            # Base.Placement(origin, axis, angle)
            self.options['plane'] = BluemiraPlane()
        elif plane == "xz":
            # Base.Placement(origin, axis, angle)
            self.options['plane'] = BluemiraPlane(
                axis=(1.0, 0.0, 0.0), angle=-90.0
            )
        elif plane == "yz":
            # Base.Placement(origin, axis, angle)
            self.options['plane'] = BluemiraPlane(
                axis=(0.0, 1.0, 0.0), angle=90.0
            )
        elif isinstance(plane, BluemiraPlane):
            self.options['plane'] = plane
            pass
        else:
            PlottingError(f"{plane} is not a valid plane")

    @abstractmethod
    def plot(self, data, ax=None, *argv, **kwargs):
        """Plotting method"""
        pass


class PointsPlotter(BasePlotter):
    """
    Base utility plotting class for points
    """

    def __init__(self, ax=None, **kwargs):
        self.options = DEFAULT
        super().__init__(**{**self.options, **kwargs})

    def plot(
        self,
        points,
        ax=None,
        show: bool = False,
        block: bool = False,
    ):
        """
        Main plot function

        Parameters
        ----------
        points: Iterable
            List of 3D points
        ax:
            matplotlib axes
        show: bool
            flag for plotting
        block: bool
            matplot flag in show function
        """
        if not self.options["plot_flag"]["poptions"]:
            return ax

        if not self.options["poptions"]:
            self.ax = ax
        else:
            if ax is None:
                fig = plt.figure()
                self.ax = fig.add_subplot()
            else:
                self.ax = ax

            points = points[0:2]

            self.ax.scatter(*points, **self.options["poptions"])
            self.data = points.tolist()

            if show:
                plt.gca().set_aspect("auto")
                plt.show(block=block)
        return self.ax


class WirePlotter(BasePlotter):
    """
    Base utility plotting class for bluemira wires
    """

    def __init__(self, **kwargs):
        self.options = DEFAULT
        super().__init__(**{**self.options, **kwargs})

    def plot(
        self,
        wire,
        ax=None,
        show: bool = False,
        block: bool = False,
        ndiscr=100,
        byedges=True,
    ):
        """WirePlotter plotting method"""
        if not isinstance(wire, BluemiraWire):
            raise ValueError("wire must be a BluemiraWire")

        if (
            not self.options["plot_flag"]["poptions"]
            and not self.options["plot_flag"]["woptions"]
        ):
            return ax

        if not self.options["poptions"] and not self.options["woptions"]:
            return ax
        else:
            if ax is None:
                fig = plt.figure()
                self.ax = fig.add_subplot()
            else:
                self.ax = ax

            if not isinstance(wire, BluemiraWire):
                raise ValueError("wire must be a BluemiraWire")

            new_wire = wire.deepcopy()
            new_wire.change_plane(self.options["plane"])

            pointsw = new_wire.discretize(ndiscr=ndiscr, byedges=byedges).T
            self.data = pointsw.tolist()

            # since the object have been moved in the new plane
            # only the first two coordinates have to be plotted
            data_to_plot = pointsw[0:2]

            if self.options["plot_flag"]["woptions"]:
                self.ax.plot(*data_to_plot, **self.options["woptions"])

            if self.options["plot_flag"]["poptions"]:
                pplotter = PointsPlotter(**self.options)
                self.ax = pplotter.plot(data_to_plot, self.ax, show=False)

            if show:
                plt.gca().set_aspect("equal")
                plt.show(block=block)

        return self.ax


class FacePlotter(BasePlotter):
    """
    Base utility plotting class for bluemira faces
    """

    def __init__(self, **kwargs):
        self.options = DEFAULT
        super().__init__(**{**self.options, **kwargs})

    def plot(
        self,
        face,
        ax=None,
        show: bool = False,
        block: bool = False,
        ndiscr=100,
        byedges=True,
    ):
        """FacePlotter plotting method"""
        if not isinstance(face, BluemiraFace):
            raise ValueError("wire must be a BluemiraFace")

        if (
            not self.options["plot_flag"]["poptions"]
            and not self.options["plot_flag"]["woptions"]
            and not self.options["plot_flag"]["foptions"]
        ):
            return ax

        if (
            not self.options["poptions"]
            and not self.options["woptions"]
            and not self.options["foptions"]
        ):
            return ax
        else:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot()
            else:
                ax = ax

            self.data = [[], [], []]

            j = 0
            for w in face._shape.Wires:
                j = j+1
                boundary = BluemiraWire(w)
                wplotter = WirePlotter(**self.options)
                wplotter.plot(boundary, ax=ax, show=False, ndiscr=ndiscr)
                # Todo: it seems that discretize and discretize_by_edges produce a
                #  different output in case all the Edges of a Wire are reversed. To
                #  be checked.
                # The behaviour above would not allow the plot of a filled face
                # since the internal holes would be considered in the same direction
                # of the external one. Solved a trick, but to be adjusted.
                if j==1:
                    self.data[0] += wplotter.data[0][::-1] + [None]
                    self.data[1] += wplotter.data[1][::-1] + [None]
                    self.data[2] += wplotter.data[2][::-1] + [None]
                else:
                    self.data[0] += wplotter.data[0] + [None]
                    self.data[1] += wplotter.data[1] + [None]
                    self.data[2] += wplotter.data[2] + [None]

            if self.options["plot_flag"]["foptions"] and self.options["foptions"]:
                # since the object have been moved in the new plane
                # only the first two coordinates have to be plotted
                data_to_plot = self.data[0:2]
                plt.fill(*data_to_plot, **self.options["foptions"])

            if show:
                plt.gca().set_aspect("equal")
                plt.show(block=block)

        return ax

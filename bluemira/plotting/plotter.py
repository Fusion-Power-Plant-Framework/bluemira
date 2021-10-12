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

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace


DEFAULT = {}
DEFAULT["plot_flag"] = {"poptions": True, "woptions": True, "foptions": True}
DEFAULT["poptions"] = {"s": 30, "facecolors": "blue", "edgecolors": "black"}
DEFAULT["woptions"] = {"color": "black", "linewidth": "2"}
DEFAULT["foptions"] = {"color": "red"}


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
    def __init__(self, plotter=None, object=None):
        self._plotter = plotter
        self._plottable_object = object

    def plot(self, **kwargs):
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
                    self.options[k] = kwargs[k]

    @abstractmethod
    def plot(self, data, ax=None, *argv, **kwargs):
        pass


class PointsPlotter(BasePlotter):
    """
    Base utility plotting class for points
    """

    def __init__(self, ax=None, **kwargs):
        self.options = DEFAULT
        super().__init__(**kwargs)

    def plot(self, points, ax=None, show: bool = False, block: bool = False):
        if not self.options["plot_flag"]["poptions"]:
            return ax

        if not self.options["poptions"]:
            self.ax = ax
        else:
            if ax is None:
                self.ax = Plot3D()
            else:
                self.ax = ax

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
        super().__init__(**kwargs)

    def plot(
        self,
        wire,
        ax=None,
        show: bool = False,
        block: bool = False,
        ndiscr=100,
        byedges=True,
    ):

        if (
            not self.options["plot_flag"]["poptions"]
            and not self.options["plot_flag"]["woptions"]
        ):
            return ax

        if not self.options["poptions"] and not self.options["woptions"]:
            return ax
        else:
            if ax is None:
                self.ax = Plot3D()
            else:
                self.ax = ax

            if not isinstance(wire, BluemiraWire):
                raise ValueError("wire must be a BluemiraWire")

            pointsw = wire.discretize(ndiscr=ndiscr, byedges=byedges).T
            self.data = pointsw.tolist()

            if self.options["plot_flag"]["woptions"]:
                self.ax.plot(*pointsw, **self.options["woptions"])

            if self.options["plot_flag"]["poptions"]:
                pplotter = PointsPlotter(**self.options)
                self.ax = pplotter.plot(pointsw, self.ax, show=False)

            if show:
                plt.gca().set_aspect("auto")
                plt.show(block=block)

        return self.ax


class FacePlotter(BasePlotter):
    """
    Base utility plotting class for bluemira faces
    """

    def __init__(self, **kwargs):
        self.options = DEFAULT
        super().__init__(**kwargs)

    def plot(
        self,
        face,
        ax=None,
        show: bool = False,
        block: bool = False,
        ndiscr=100,
        byedges=True,
    ):

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
                self.ax = Plot3D()
            else:
                self.ax = ax

            if not isinstance(face, BluemiraFace):
                raise ValueError("wire must be a BluemiraFace")

            for boundary in face.boundary:
                    wplotter = WirePlotter(**self.options)
                    wplotter.plot(boundary, show=False)
                    # TODO: self.data should add a None line every time a new
                    #  boundary is found.
                    self.data += wplotter.data
                    for o in self.data:
                        o = o + [None]

            for o in self.data:
                o = o[:-1]

            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            x = self.data[0]
            y = self.data[1]
            z = self.data[2]

            verts = [list(zip(x, y, z))]
            self.ax.add_collection3d(Poly3DCollection(verts))

            # plt.fill(*self.data, **self.options['foptions'])

            if show:
                plt.gca().set_aspect("auto")
                plt.show(block=block)

        return self.ax

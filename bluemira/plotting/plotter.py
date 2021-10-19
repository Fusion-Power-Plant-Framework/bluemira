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
DEFAULT["flags"] = {"points": True, "wires": True, "faces": True}
DEFAULT["poptions"] = {"s": 10, "facecolors": "blue", "edgecolors": "black"}
DEFAULT["woptions"] = {"color": "black", "linewidth": "0.5"}
DEFAULT["foptions"] = {"color": "red"}
DEFAULT["plane"] = "xy"
DEFAULT["palette"] = None

# Note: when plotting points, it can happen that markers are not centered properly as
# described in https://github.com/matplotlib/matplotlib/issues/11836


class BasePlotter(ABC):
    """
    Base utility plotting class
    """

    def __init__(self, **kwargs):
        self._data = []  # data passed to the BasePlotter
        self._data_to_plot = []  # real data that is plotted
        self.ax = None
        if kwargs:
            for k in kwargs:
                if k in self.options:
                    if k == "plane":
                        self.set_plane(kwargs[k])
                    else:
                        self.options[k] = kwargs[k]

    @property
    def plot_points(self):
        """Set the flag to plot points"""
        return self.options["flags"]["points"]

    @plot_points.setter
    def plot_points(self, value):
        self.options["flags"]["points"] = value

    @property
    def plot_wires(self):
        """Set the flag to plot wires"""
        return self.options["flags"]["wires"]

    @plot_wires.setter
    def plot_wires(self, value):
        self.options["flags"]["wires"] = value

    @property
    def plot_faces(self):
        """Set the flag to plot faces"""
        return self.options["flags"]["faces"]

    @plot_faces.setter
    def plot_faces(self, value):
        self.options["flags"]["faces"] = value

    @property
    def poptions(self):
        """Plot options for points"""
        return self.options["poptions"]

    @property
    def woptions(self):
        """Plot options for wires"""
        return self.options["woptions"]

    @property
    def foptions(self):
        """Plot options for faces"""
        return self.options["foptions"]

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
            self.options["poptions"] = value
        elif isinstance(value, tuple):
            self.options["poptions"][value[0]] = value[1]
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
            self.options["woptions"] = value
        elif isinstance(value, tuple):
            self.options["woptions"][value[0]] = value[1]
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
            self.options["foptions"] = value
        elif isinstance(value, tuple):
            self.options["foptions"][value[0]] = value[1]
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

    def __call__(
        self, obj, ax=None, show: bool = False, block: bool = False, *args, **kwargs
    ):
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


class PointsPlotter(BasePlotter):
    """
    Base utility plotting class for points
    """

    def __init__(self, ax=None, **kwargs):
        # set the plot options to DEFAULT. A copy is made in order to be able to
        # change options without modifying the DEFAULT dictionary
        self.options = DEFAULT.copy()
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

    def _make_plot(self, points, *args, **kwargs):
        self._data = points.tolist()
        self._data_to_plot = points[0:2]
        self.ax.scatter(*self._data_to_plot, **self.options["poptions"])


class WirePlotter(BasePlotter):
    """
    Base utility plotting class for bluemira wires
    """

    def __init__(self, **kwargs):
        # set the plot options to DEFAULT. A copy is made in order to be able to
        # change options without modifying the DEFAULT dictionary
        self.options = DEFAULT.copy()
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

    def _make_plot(self, wire, ndiscr, byedges):
        new_wire = wire.deepcopy()
        new_wire.change_plane(self.options["plane"])
        pointsw = new_wire.discretize(ndiscr=ndiscr, byedges=byedges).T
        self._data = pointsw.tolist()
        self._data_to_plot = pointsw[0:2]

        if self.plot_wires:
            self.ax.plot(*self._data_to_plot, **self.options["woptions"])

        if self.plot_points:
            pplotter = PointsPlotter(**self.options)
            self.ax = pplotter(self._data_to_plot, self.ax, show=False)


class FacePlotter(BasePlotter):
    """
    Base utility plotting class for bluemira faces
    """

    def __init__(self, **kwargs):
        # set the plot options to DEFAULT. A copy is made in order to be able to
        # change options without modifying the DEFAULT dictionary
        self.options = DEFAULT.copy()
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

    def _make_plot(self, face, ndiscr, byedges):
        self._data = [[], [], []]

        for w in face._shape.Wires:
            boundary = BluemiraWire(w)
            wplotter = WirePlotter(**self.options)
            if not self.plot_wires and not self.plot_points:
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

        if self.plot_faces and self.options["foptions"]:
            plt.fill(*self._data_to_plot, **self.options["foptions"])


class FaceCompoundPlotter(FacePlotter):
    """
    Base utility plotting class for shape compounds
    """

    def __init__(self, **kwargs):
        # set the plot options to DEFAULT. A copy is made in order to be able to
        # change options without modifying the DEFAULT dictionary
        self._acceptable_classes = [BluemiraFace]
        self.options = DEFAULT.copy()
        super().__init__(**{**self.options, **kwargs})

    def _check_obj(self, objs):
        """Check if objects in objs are of the correct type for this class"""
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
        if "palette" in self.options:
            import seaborn as sns

            palette = sns.color_palette(self.options["palette"], len(objs))
            print(f"palette: {palette}")
        else:
            palette = self.foptions["color"]

        for id, obj in enumerate(objs):
            temp_fplotter = FacePlotter(**self.options)
            temp_fplotter.change_foptions(("color", palette[id]))
            print(temp_fplotter.foptions)
            self.ax = temp_fplotter(
                obj, ax=self.ax, show=False, ndiscr=ndiscr, byedges=byedges
            )
            self._data += [temp_fplotter._data]
            self._data_to_plot += [temp_fplotter._data_to_plot]

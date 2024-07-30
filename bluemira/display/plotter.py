# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
api for plotting using matplotlib
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np
from matplotlib.patches import PathPatch, Polygon

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.error import DisplayError
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.display.tools import Options
from bluemira.geometry import bound_box, face, wire
from bluemira.geometry import placement as _placement
from bluemira.geometry.coordinates import (
    Coordinates,
    _parse_to_xyz_array,
    get_centroid_3d,
    rotation_matrix_v1v2,
)
from bluemira.utilities.tools import flatten_iterable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from matplotlib.axes import Axes

    from bluemira.geometry.base import BluemiraGeoT

UNIT_LABEL = "[m]"
X_LABEL = f"x {UNIT_LABEL}"
Y_LABEL = f"y {UNIT_LABEL}"
Z_LABEL = f"z {UNIT_LABEL}"


class Zorder(Enum):
    """Layer ordering of common plots"""

    POSITION_1D = 1
    POSITION_2D = 2
    PLASMACURRENT = 7
    PSI = 8
    FLUXSURFACE = 9
    SEPARATRIX = 10
    OXPOINT = 11
    FACE = 20
    WIRE = 30
    RADIATION = 40
    CONSTRAINT = 45
    TEXT = 100


class ViewDescriptor:
    """Descriptor for placements in dataclass"""

    def __init__(self):
        self._default = tuple(
            getattr(_placement.XZY, attr) for attr in ("base", "axis", "angle", "label")
        )

    def __set_name__(self, _, name: str):
        """Set the attribute name from a dataclass"""
        self._name = "_" + name

    def __get__(self, obj: Any, _) -> tuple[np.ndarray, np.ndarray, float, str]:
        """Get the view tuple"""
        if obj is None:
            return self._default

        return getattr(obj, self._name, self._default)

    def __set__(self, obj: Any, value: str | tuple | _placement.BluemiraPlacement):
        """Set the view

        Raises
        ------
        DisplayError
            View not known
        """
        if isinstance(value, str):
            if value.startswith("xy"):
                value = _placement.XYZ
            elif value.startswith("xz"):
                value = _placement.XZY
            elif value.startswith("yz"):
                value = _placement.YZX
            else:
                raise DisplayError(f"{value} is not a valid view")

        if isinstance(value, tuple):
            value = _placement.BluemiraPlacement(*value)

        setattr(
            obj,
            self._name,
            tuple(getattr(value, attr) for attr in ("base", "axis", "angle", "label")),
        )


class DictOptionsDescriptor:
    """Keep defaults for options unless explicitly overwritten

    Notes
    -----
    The default will be reinforced if value set to new dictionary.
    Otherwise as a dictionary is mutable will act in the normal way.

    .. code-block:: python

      po = PlotOptions()
      po.wire_options["linewidth"] = 0.1  # overrides default
      po.wire_options["zorder"] = 2  # adds new zorder option
      # setting a new dictionary resets the defaults with zorder overridden
      po.wire_options = {'zorder': 1}
      del po.wire_options["linewidth"]  # linewidth unset

    No checks are done on the contents of the dictionary.

    """

    def __init__(self, default_factory: Callable[[], dict[str, Any]] | None = None):
        self.default = {} if default_factory is None else default_factory()

    def __set_name__(self, _, name: str):
        """Set the attribute name from a dataclass"""
        self._name = "_" + name

    def __get__(self, obj: Any, _) -> Callable[[], dict[str, Any]] | dict[str, Any]:
        """Get the options dictionary"""
        if obj is None:
            return lambda: self.default

        return getattr(obj, self._name, self.default)

    def __set__(self, obj: Any, value: Callable[[], dict[str, Any]] | dict[str, Any]):
        """Set the options dictionary"""
        if callable(value):
            value = value()
        default = getattr(obj, self._name, self.default)
        setattr(obj, self._name, {**default, **value})


@dataclass
class DefaultPlotOptions:
    """
    The options that are available for plotting objects in 2D.

    Parameters
    ----------
    show_points:
        If True, points are plotted. By default False.
    show_wires:
        If True, wires are plotted. By default True.
    show_faces:
        If True, faces are plotted. By default True.
    point_options:
        Dictionary with matplotlib options for points. By default  {"s": 10,
        "facecolors": "blue", "edgecolors": "black"}
    wire_options:
        Dictionary with matplotlib options for wires. By default {"color": "black",
        "linewidth": "0.5"}
    face_options:
        Dictionary with matplotlib options for faces. By default {"color": "red"}
    view:
        The reference view for plotting. As string, possible
        options are "xy", "xz", "yz". By default 'xz'.
    ndiscr:
        The number of points to use when discretising a wire or face.
    byedges:
        If True then wires or faces will be discretised respecting their edges.
    """

    # flags to enable points, wires, and faces plot
    show_points: bool = False
    show_wires: bool = True
    show_faces: bool = True
    # matplotlib set of options to plot points, wires, and faces. If an empty dictionary
    # is specified, the default color plot of matplotlib is used.
    point_options: DictOptionsDescriptor = DictOptionsDescriptor(
        lambda: {
            "s": 10,
            "facecolors": "red",
            "edgecolors": "black",
            "zorder": 30,
        }
    )
    wire_options: DictOptionsDescriptor = DictOptionsDescriptor(
        lambda: {"color": "black", "linewidth": 0.5, "zorder": Zorder.WIRE.value}
    )
    face_options: DictOptionsDescriptor = DictOptionsDescriptor(
        lambda: {"color": "blue", "zorder": Zorder.FACE.value}
    )
    # discretisation properties for plotting wires (and faces)
    ndiscr: int = 100
    byedges: bool = True
    # View of object
    view: ViewDescriptor = ViewDescriptor()

    @property
    def view_placement(self) -> _placement.BluemiraPlacement:
        """Get view as BluemiraPlacement"""
        return _placement.BluemiraPlacement(*self.view)


class PlotOptions(Options):
    """
    The options that are available for plotting objects
    """

    __slots__ = ()

    def __init__(self, **kwargs):
        self._options = DefaultPlotOptions()
        super().__init__(**kwargs)


def get_default_options() -> PlotOptions:
    """
    Returns the default plot options.
    """
    return PlotOptions()


# Note: when plotting points, it can happen that markers are not centred properly as
# described in https://github.com/matplotlib/matplotlib/issues/11836
class BasePlotter(ABC):
    """
    Base utility plotting class
    """

    _CLASS_PLOT_OPTIONS: ClassVar = {}

    def __init__(self, options: PlotOptions | None = None, **kwargs):
        # discretisation points representing the shape in global coordinate system
        self._data = []
        # modified discretisation points for plotting (e.g. after view transformation)
        self._data_to_plot = []
        self.options = (
            PlotOptions(**self._CLASS_PLOT_OPTIONS) if options is None else options
        )
        self.options.modify(**kwargs)
        self.set_view(self.options._options.view)

    def set_view(self, view):
        """Set the plotting view"""
        if isinstance(view, str | _placement.BluemiraPlacement):
            self.options._options.view = view
        else:
            DisplayError(f"{view} is not a valid view")

    @property
    def ax(self) -> Axes:
        """Axes object"""
        try:
            return self._ax
        except AttributeError:
            fig = plt.figure()
            self._ax = fig.add_subplot()

        return self._ax

    @abstractmethod
    def _check_obj(self, obj):
        """Internal function that check if obj is an instance of the correct class"""

    @abstractmethod
    def _check_options(self):
        """Internal function that check if it is needed to plot something"""

    def initialise_plot_2d(self, ax=None):
        """Initialise the plot environment"""
        if ax is None:
            fig = plt.figure()
            self._ax = fig.add_subplot()
        else:
            self._ax = ax

    @staticmethod
    def show():
        """Function to show a plot"""
        plt.show(block=True)

    @abstractmethod
    def _populate_data(self, obj):
        """
        Internal function that makes the plot. It fills self._data and
        self._data_to_plot
        """

    @abstractmethod
    def _make_plot_2d(self):
        """
        Internal function that makes the plot. It should use self._data and
        self._data_to_plot, so _populate_data should be called before.
        """

    def _set_aspect_2d(self):
        self.ax.set_aspect("equal")

    def _set_label_2d(self):
        axis = np.abs(self.options.view_placement.axis)
        if np.allclose(axis, (1, 0, 0)):
            self.ax.set_xlabel(X_LABEL)
            self.ax.set_ylabel(Z_LABEL)
        elif np.allclose(axis, (0, 0, 1)):
            self.ax.set_xlabel(X_LABEL)
            self.ax.set_ylabel(Y_LABEL)
        elif np.allclose(axis, (0, 1, 0)):
            self.ax.set_xlabel(Y_LABEL)
            self.ax.set_ylabel(Z_LABEL)
        else:
            # Do not put x,y,z labels for views we do not recognise
            self.ax.set_xlabel(UNIT_LABEL)
            self.ax.set_ylabel(UNIT_LABEL)

    def _set_aspect_3d(self):
        # This was the only way I found to get 3-D plots to look right in matplotlib
        x_bb, y_bb, z_bb = bound_box.BoundingBox.from_xyz(*self._data.T).get_box_arrays()
        for x, y, z in zip(x_bb, y_bb, z_bb, strict=False):
            self.ax.plot([x], [y], [z], color="w")

    def _set_label_3d(self):
        offset = "\n\n"  # To keep labels from interfering with the axes
        self.ax.set_xlabel(offset + X_LABEL)
        self.ax.set_ylabel(offset + Y_LABEL)
        self.ax.set_zlabel(offset + Z_LABEL)

    def plot_2d(self, obj, ax=None, *, show: bool = True):
        """2D plotting method"""
        self._check_obj(obj)

        if not self._check_options():
            self._ax = ax
        else:
            self.initialise_plot_2d(ax)
            self._populate_data(obj)
            self._make_plot_2d()
            self._set_aspect_2d()
            self._set_label_2d()

            if show:
                self.show()
        return self.ax

    # # =================================================================================
    # # 3-D functions
    # # =================================================================================
    def initialise_plot_3d(self, ax=None):
        """Initialise the plot environment"""
        if ax is None:
            fig = plt.figure()
            self._ax = fig.add_subplot(projection="3d")
        else:
            self._ax = ax

    @abstractmethod
    def _make_plot_3d(self):
        """Internal function that makes the plot. It should use self._data and
        self._data_to_plot, so _populate_data should be called before.
        """

    def plot_3d(self, obj, ax=None, *, show: bool = True):
        """3D plotting method"""
        self._check_obj(obj)

        if not self._check_options():
            self._ax = ax
        else:
            self.initialise_plot_3d(ax=ax)
            # this function can be common to 2D and 3D plot
            # self._data is used for 3D plot
            # self._data_to_plot is used for 2D plot
            # TODO: probably better to rename self._data_to_plot into
            #  self._projected_data or self._data2d
            self._populate_data(obj)
            self._make_plot_3d()
            self._set_aspect_3d()
            self._set_label_3d()

            if show:
                self.show()

        return self.ax


class PointsPlotter(BasePlotter):
    """
    Base utility plotting class for points
    """

    _CLASS_PLOT_OPTIONS: ClassVar = {"show_points": True}

    @staticmethod
    def _check_obj(obj):  # noqa: ARG004
        # TODO: create a function that checks if the obj is a cloud of 3D or 2D points
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        return bool(self.options.show_points)

    def _populate_data(self, points):
        points = _parse_to_xyz_array(points).T
        self._data = points
        # apply rotation matrix given by options['view']
        rot = self.options.view_placement.to_matrix().T
        temp_data = np.c_[self._data, np.ones(len(self._data))]
        self._data_to_plot = temp_data.dot(rot).T
        self._data_to_plot = self._data_to_plot[0:2]

    def _make_plot_2d(self):
        if self.options.show_points:
            self.ax.scatter(*self._data_to_plot, **self.options.point_options)
        self._set_aspect_2d()

    def _make_plot_3d(self, *args, **kwargs):  # noqa: ARG002
        if self.options.show_points:
            self.ax.scatter(*self._data.T, **self.options.point_options)
        self._set_aspect_3d()


class WirePlotter(BasePlotter):
    """
    Base utility plotting class for bluemira wires
    """

    _CLASS_PLOT_OPTIONS: ClassVar = {"show_points": False}

    @staticmethod
    def _check_obj(obj):
        if not isinstance(obj, wire.BluemiraWire):
            raise TypeError(f"{obj} must be a BluemiraWire")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        return not (not self.options.show_points and not self.options.show_wires)

    def _populate_data(self, wire):
        self._pplotter = PointsPlotter(self.options)
        new_wire = wire.deepcopy()
        # # change of view integrated in PointsPlotter2D. Not necessary here.
        # new_wire.change_placement(self.options._options['view'])
        pointsw = new_wire.discretise(
            ndiscr=self.options._options.ndiscr,
            byedges=self.options._options.byedges,
        ).T
        self._pplotter._populate_data(pointsw)
        self._data = pointsw
        self._data_to_plot = self._pplotter._data_to_plot

    def _make_plot_2d(self):
        if self.options.show_wires:
            self.ax.plot(*self._data_to_plot, **self.options.wire_options)

        if self.options.show_points:
            self._pplotter._ax = self.ax
            self._pplotter._make_plot_2d()
        self._set_aspect_2d()

    def _make_plot_3d(self):
        if self.options.show_wires:
            self.ax.plot(*self._data.T, **self.options.wire_options)

        if self.options.show_points:
            self._pplotter._ax = self.ax
            self._pplotter._make_plot_3d()
        self._set_aspect_3d()


class FacePlotter(BasePlotter):
    """Base utility plotting class for bluemira faces"""

    _CLASS_PLOT_OPTIONS: ClassVar = {"show_points": False, "show_wires": False}

    @staticmethod
    def _check_obj(obj):
        if not isinstance(obj, face.BluemiraFace):
            raise TypeError(f"{obj} must be a BluemiraFace")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        return not (
            not self.options.show_points
            and not self.options.show_wires
            and not self.options.show_faces
        )

    def _populate_data(self, face):
        self._data = []
        self._wplotters = []
        # TODO: the for must be done using face.shape.Wires because FreeCAD
        #  re-orient the Wires in the correct way for display. Find another way to do
        #  it (maybe adding this function to the freecadapi.
        for w in face.shape.Wires:
            boundary = wire.BluemiraWire(w)
            wplotter = WirePlotter(self.options)
            self._wplotters.append(wplotter)
            wplotter._populate_data(boundary)
            self._data.extend(wplotter._data.tolist())
        self._data = np.array(self._data)

        self._data_to_plot = [[], []]
        for w in self._wplotters:
            self._data_to_plot[0] += [*w._data_to_plot[0].tolist(), None]
            self._data_to_plot[1] += [*w._data_to_plot[1].tolist(), None]

    def _make_plot_2d(self):
        if self.options.show_faces:
            face_opts = self.options.face_options
            if face_opts.get("hatch", None) is not None:
                self.ax.add_patch(
                    Polygon(
                        np.asarray(self._data_to_plot).T,
                        fill=False,
                        **face_opts,
                    )
                )
            else:
                self.ax.fill(*self._data_to_plot, **face_opts)

        for plotter in self._wplotters:
            plotter._ax = self.ax
            plotter._make_plot_2d()
        self._set_aspect_2d()

    def _make_plot_3d(self):
        if self.options.show_faces:
            poly = a3.art3d.Poly3DCollection([self._data], **self.options.face_options)
            self.ax.add_collection3d(poly)

        for plotter in self._wplotters:
            plotter._ax = self.ax
            plotter._make_plot_3d()
        self._set_aspect_3d()


class ComponentPlotter(BasePlotter):
    """Base utility plotting class for bluemira faces"""

    _CLASS_PLOT_OPTIONS: ClassVar = {"show_points": False, "show_wires": False}

    @staticmethod
    def _check_obj(obj):
        import bluemira.base.components  # noqa: PLC0415

        if not isinstance(obj, bluemira.base.components.Component):
            raise TypeError(f"{obj} must be a BluemiraComponent")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        return not (
            not self.options.show_points
            and not self.options.show_wires
            and not self.options.show_faces
        )

    def _populate_data(self, comp):
        self._cplotters = []

        def _populate_plotters(comp):
            if comp.is_leaf and getattr(comp, "shape", None) is not None:
                if comp.plot_options.face_options["color"] in flatten_iterable(
                    BLUE_PALETTE.as_hex()
                ):
                    if self.options.face_options["color"] == "blue":
                        options = comp.plot_options
                    else:
                        # not possible with ComponentPlotter only plot_2d
                        options = self.options
                else:
                    options = comp.plot_options
                plotter = _get_plotter_class(comp.shape)(options)
                plotter._populate_data(comp.shape)
                self._cplotters.append(plotter)
            else:
                for child in comp.children:
                    _populate_plotters(child)

        _populate_plotters(comp)

    def _make_plot_2d(self):
        for plotter in self._cplotters:
            plotter._ax = self.ax
            plotter._make_plot_2d()
        self._set_aspect_2d()

    def _make_plot_3d(self):
        """
        Internal function that makes the plot. It should use self._data and
        self._data_to_plot, so _populate_data should be called before.
        """
        for plotter in self._cplotters:
            plotter._ax = self.ax
            plotter._make_plot_3d()

    def _set_aspect_3d(self):
        pass


def _validate_plot_inputs(
    parts, options
) -> tuple[list[BluemiraGeoT], list[PlotOptions] | list[None]]:
    """
    Validate the lists of parts and options, applying some default options.

    Raises
    ------
    DisplayError
        Number of options not equal to number of parts
    """
    if not isinstance(parts, list):
        parts = [parts]

    if options is None:
        options = [None] * len(parts)
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

    Raises
    ------
    DisplayError
        No plotter available for type of part
    """
    import bluemira.base.components  # noqa: PLC0415

    if isinstance(part, list | np.ndarray | Coordinates):
        plot_class = PointsPlotter
    elif isinstance(part, wire.BluemiraWire):
        plot_class = WirePlotter
    elif isinstance(part, face.BluemiraFace):
        plot_class = FacePlotter
    elif isinstance(part, bluemira.base.components.Component):
        plot_class = ComponentPlotter
    else:
        raise DisplayError(
            f"{part} object cannot be plotted. No Plotter available for {type(part)}"
        )
    return plot_class


def plot_2d(
    parts: BluemiraGeoT | Iterable[BluemiraGeoT],
    options: PlotOptions | Iterable[PlotOptions] | Iterable[None] | None = None,
    ax=None,
    *,
    show: bool = True,
    **kwargs,
):
    """
    The implementation of the display API for BluemiraGeo parts.

    Parameters
    ----------
    parts: Union[Part.Shape, List[Part.Shape]]
        The parts to display.
    options: Optional[Union[PlotOptions, List[PlotOptions]]]
        The options to use to display the parts.
    ax: Optional[Axes]
        The axes onto which to plot
    show: bool
        Whether or not to show the plot immediately (default=True). Note
        that if using iPython or Jupyter, this has no effect; the plot is shown
        automatically.
    """
    parts, options = _validate_plot_inputs(parts, options)

    for part, option in zip(parts, options, strict=False):
        plotter = _get_plotter_class(part)(option, **kwargs)
        ax = plotter.plot_2d(part, ax, show=False)

    if show:
        plotter.show()

    return ax


def plot_3d(
    parts: BluemiraGeoT | Iterable[BluemiraGeoT],
    options: PlotOptions | Iterable[PlotOptions] | Iterable[None] | None = None,
    ax=None,
    *,
    show: bool = True,
    **kwargs,
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
        Whether or not to show the plot immediately in the console. (default=True). Note
        that if using iPython or Jupyter, this has no effect; the plot is shown
        automatically.
    """
    parts, options = _validate_plot_inputs(parts, options)

    for part, option in zip(parts, options, strict=False):
        plotter = _get_plotter_class(part)(option, **kwargs)
        ax = plotter.plot_3d(part, ax, show=False)

    if show:
        plotter.show()

    return ax


class Plottable:
    """
    Mixin class to make a class plottable in 2D by imparting a plot2d method and
    options.

    Notes
    -----
    The implementing class must set the _plotter2D attribute to an instance of the
    appropriate Plotter2D class.
    """

    def __init__(self):
        super().__init__()
        self._plot_options: PlotOptions = PlotOptions()
        self._plot_options.face_options["color"] = next(BLUE_PALETTE)

    @property
    def plot_options(self) -> PlotOptions:
        """
        The options that will be used to display the object.
        """
        return self._plot_options

    @plot_options.setter
    def plot_options(self, value: PlotOptions):
        if not isinstance(value, PlotOptions):
            raise DisplayError("Display options must be set to a PlotOptions instance.")
        self._plot_options = value

    @property
    def _plotter(self) -> BasePlotter:
        """
        The options that will be used to display the object.
        """
        return _get_plotter_class(self)(self._plot_options)

    def plot_2d(self, ax=None, *, show: bool = True) -> None:
        """
        Default method to call display the object by calling into the Displayer's display
        method.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        return self._plotter.plot_2d(self, ax=ax, show=show)

    def plot_3d(self, ax=None, *, show: bool = True) -> None:
        """
        Function to 3D plot a component.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        return self._plotter.plot_3d(self, ax=ax, show=show)


def _get_ndim(coords):
    count = 0
    length = coords.shape[1]
    for c in coords.xyz:
        if len(c) == length and not np.allclose(c, c[0] * np.ones(length)):
            count += 1

    return max(count, 2)


def _get_plan_dims(array):
    length = array.shape[1]
    axes = ["x", "y", "z"]
    dims = []
    for i, k in enumerate(axes):
        c = array[i]
        if not np.allclose(c[0] * np.ones(length), c):
            dims.append(k)

    if len(dims) == 1:
        # Stops error when flat lines are given (same coords in two axes)
        axes.remove(dims[0])  # remove variable axis
        temp = []
        for i, k in enumerate(axes):  # both all equal to something
            c = array[i]
            if c[0] != 0.0:
                temp.append(k)
        if len(temp) == 1:
            dims.append(temp[0])
        else:
            # This is likely due to a 3-5 long loop which is still straight
            # need to choose between one of the two constant dimensions
            # Just default to x - z, this is pretty rare..
            # usually due to an offset x - z loop
            dims = ["x", "z"]

    return sorted(dims)


def plot_coordinates(coords, ax=None, *, points=False, **kwargs):
    """
    Plot Coordinates.

    Parameters
    ----------
    coords: Coordinates
        Coordinates to plot
    ax: Axes
        Matplotlib axis on which to plot

    Other Parameters
    ----------------
    edgecolor: str
        The edgecolor to plot the Coordinates with
    facecolor: str
        The facecolor to plot the Coordinates fill with
    alpha: float
        The transparency to plot the Coordinates fill with
    """
    from bluemira.utilities.plot_tools import coordinates_to_path  # noqa: PLC0415

    ndim = _get_ndim(coords)

    fc = kwargs.get("facecolor", "royalblue")
    lw = kwargs.get("linewidth", 2)
    ls = kwargs.get("linestyle", "-")
    alpha = kwargs.get("alpha", 1)

    if coords.closed:
        fill = kwargs.get("fill", True)
        ec = kwargs.get("edgecolor", "k")
    else:
        fill = kwargs.get("fill", False)
        ec = kwargs.get("edgecolor", "r")

    if ndim == 2 and ax is None:  # noqa: PLR2004
        ax = kwargs.get("ax", plt.gca())

    if ndim == 3 or (ndim == 2 and hasattr(ax, "zaxis")):  # noqa: PLR2004
        kwargs = {
            "edgecolor": ec,
            "facecolor": fc,
            "linewidth": lw,
            "linestyle": ls,
            "alpha": alpha,
            "fill": fill,
        }
        _plot_3d(coords, ax=ax, **kwargs)

    a, b = _get_plan_dims(coords.xyz)
    x, y = (getattr(coords, c) for c in [a, b])
    marker = "o" if points else None
    ax.set_xlabel(a + " [m]")
    ax.set_ylabel(b + " [m]")
    if fill:
        poly = coordinates_to_path(x, y)
        p = PathPatch(poly, color=fc, alpha=alpha)
        ax.add_patch(p)

    ax.plot(x, y, color=ec, marker=marker, linewidth=lw, linestyle=ls)

    if points:
        for i, p in enumerate(zip(x, y, strict=False)):
            ax.annotate(i, xy=(p[0], p[1]))

    ax.set_aspect("equal")


def _plot_3d(coords, ax=None, **kwargs):
    from bluemira.utilities.plot_tools import (  # noqa: PLC0415
        BluemiraPathPatch3D,
        Plot3D,
        coordinates_to_path,
    )

    if ax is None:
        ax = Plot3D()
        # Now we re-arrange a little so that matplotlib can show us something a little
        # more correct
        x_bb, y_bb, z_bb = bound_box.BoundingBox.from_xyz(*coords.xyz).get_box_arrays()
        for x, y, z in zip(x_bb, y_bb, z_bb, strict=False):
            ax.plot([x], [y], [z], color="w")

    ax.plot(*coords.xyz, color=kwargs["edgecolor"], lw=kwargs["linewidth"])
    if kwargs["fill"]:
        if not coords.is_planar:
            bluemira_warn("Cannot fill plot of non-planar Coordinates.")
            return
        dcm = rotation_matrix_v1v2(-coords.normal_vector, np.array([0.0, 0.0, 1.0]))

        xyz = dcm.T @ coords.xyz
        center_of_mass = get_centroid_3d(*xyz)

        xyz -= center_of_mass

        dims = ["x", "y", "z"]
        a, b = _get_plan_dims(xyz)
        i = dims.index(a)
        j = dims.index(b)
        x, y = xyz[i], xyz[j]

        # To make an object which matplotlib can understand
        poly = coordinates_to_path(x, y)

        # And now re-transform the matplotlib object to 3-D
        p = BluemiraPathPatch3D(
            poly,
            -coords.normal_vector,
            coords.center_of_mass,
            color=kwargs["facecolor"],
            alpha=kwargs["alpha"],
        )
        ax.add_patch(p)

    if not hasattr(ax, "zaxis"):
        ax.set_aspect("equal")

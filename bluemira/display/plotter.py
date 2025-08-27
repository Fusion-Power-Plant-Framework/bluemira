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
from typing import TYPE_CHECKING, Any, ClassVar

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import PathPatch, Polygon
from mpl_toolkits.mplot3d import art3d

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.error import DisplayError
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.display.tools import Options
from bluemira.geometry.bound_box import BoundingBox
from bluemira.geometry.coordinates import (
    Coordinates,
    _parse_to_xyz_array,
    get_centroid_3d,
    rotation_matrix_v1v2,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.placement import XYZ, XZY, YZX, BluemiraPlacement
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.tools import flatten_iterable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    import numpy.typing as npt
    from matplotlib.axes import Axes

    from bluemira.base.components import Component
    from bluemira.geometry.base import BluemiraGeoT

UNIT_LABEL = "[m]"
X_LABEL = f"x {UNIT_LABEL}"
Y_LABEL = f"y {UNIT_LABEL}"
Z_LABEL = f"z {UNIT_LABEL}"


class Zorder(Enum):
    """Layer ordering of common plots"""

    GRID = 0.5
    POSITION_1D = 1
    POSITION_2D = 2
    PLASMACURRENT = 7
    PSI = 8
    FLUXSURFACE = 9
    SEPARATRIX = 10
    OXPOINT = 11
    FACE = 20
    WIRE = 30
    POINTS = 35
    RADIATION = 40
    CONSTRAINT = 45
    TEXT = 100


class ViewDescriptor:
    """Descriptor for placements in dataclass"""

    def __init__(self):
        self._default = tuple(
            getattr(XZY, attr) for attr in ("base", "axis", "angle", "label")
        )

    def __set_name__(self, _, name: str):
        """Set the attribute name from a dataclass"""
        self._name = "_" + name

    def __get__(self, obj: Any, _) -> tuple[np.ndarray, np.ndarray, float, str]:
        """
        Get the view tuple

        Returns
        -------
        :
            the view attributes from the instance or defaults.
        """
        if obj is None:
            return self._default

        return getattr(obj, self._name, self._default)

    def __set__(self, obj: Any, value: str | tuple | BluemiraPlacement):
        """Set the view

        Raises
        ------
        DisplayError
            View not known
        """
        if isinstance(value, str):
            if value.startswith("xy"):
                value = XYZ
            elif value.startswith("xz"):
                value = XZY
            elif value.startswith("yz"):
                value = YZX
            else:
                raise DisplayError(f"{value} is not a valid view")

        if isinstance(value, tuple):
            value = BluemiraPlacement(*value)

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
        """
        Returns
        -------
        :
            the options dictionary
        """
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
            "zorder": Zorder.POINTS.value,
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
    # use external object options or default plotter options
    _external_options: bool = False

    @property
    def view_placement(self) -> BluemiraPlacement:
        """
        Returns
        -------
        :
            the view as BluemiraPlacement
        """
        return BluemiraPlacement(*self.view)


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
    Returns
    -------
    :
        the default plot options.
    """
    return PlotOptions()


# Note: when plotting points, it can happen that markers are not centred properly as
# described in https://github.com/matplotlib/matplotlib/issues/11836
class BasePlotter(ABC):
    """
    Base utility plotting class
    """

    _CLASS_PLOT_OPTIONS: ClassVar = {}

    def __init__(
        self,
        options: PlotOptions | None = None,
        *,
        data: npt.ArrayLike | Coordinates | BluemiraGeoT | Component | None = None,
        **kwargs,
    ):
        self.options = (
            PlotOptions(**self._CLASS_PLOT_OPTIONS) if options is None else options
        )
        self.options.modify(**kwargs)
        self.set_view(self.options._options.view)

        if data is not None:
            self._populate_data(data)
        else:
            # discretisation points representing the shape in global coordinate system
            self.data = []
            # modified discretisation points for plotting (e.g. view transformation)
            self._projected_data = []

    def set_view(self, view: str | BluemiraPlacement):
        """Set the plotting view"""
        if isinstance(view, str | BluemiraPlacement):
            self.options._options.view = view
        else:
            DisplayError(f"{view} is not a valid view")

    @property
    def ax(self) -> Axes:
        """
        Returns
        -------
        :
            the axes object
        """
        try:
            return self._ax
        except AttributeError:
            _fig, self._ax = plt.subplots()

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
    def _populate_data(
        self, obj: npt.ArrayLike | Coordinates | BluemiraGeoT | Component
    ):
        """
        Internal function that makes the plot. It fills self._data and
        self._projected_data
        """

    @abstractmethod
    def _make_plot_2d(self):
        """
        Internal function that makes the plot. It should use self._data and
        self._projected_data, so _populate_data should be called before.
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
        x_bb, y_bb, z_bb = BoundingBox.from_xyz(*self._data.T).get_box_arrays()
        for x, y, z in zip(x_bb, y_bb, z_bb, strict=False):
            self.ax.plot([x], [y], [z], color="w")

    def _set_label_3d(self):
        offset = "\n\n{}"  # To keep labels from interfering with the axes
        self.ax.set_xlabel(offset.format(X_LABEL))
        self.ax.set_ylabel(offset.format(Y_LABEL))
        self.ax.set_zlabel(offset.format(Z_LABEL))

    def plot_2d(
        self,
        obj: npt.ArrayLike | Coordinates | BluemiraGeoT | Component,
        ax: Axes | None = None,
        *,
        show: bool = True,
    ) -> Axes:
        """
        2D plotting method

        Returns
        -------
        :
            The axes with the plotted data.

        """
        self._check_obj(obj)

        if self._check_options():
            self.initialise_plot_2d(ax)
            self._populate_data(obj)
            self._make_plot_2d()
            self._set_aspect_2d()
            self._set_label_2d()

            if show:
                self.show()
        else:
            self._ax = ax
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
        self._projected_data, so _populate_data should be called before.
        """

    def plot_3d(
        self,
        obj: npt.ArrayLike | Coordinates | BluemiraGeoT | Component,
        ax: Axes | None = None,
        *,
        show: bool = True,
    ) -> Axes:
        """
        3D plotting method

        Returns
        -------
        :
            The axes with the plotted data.

        """
        self._check_obj(obj)

        if not self._check_options():
            self._ax = ax
        else:
            self.initialise_plot_3d(ax=ax)
            # this function can be common to 2D and 3D plot
            # self._data is used for 3D plot
            # self._projected_data is used for 2D plot
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
        """
        Returns
        -------
        :
           Always returns True.

        Notes
        -----
        This method currently always returns True.
        """
        # TODO @DanShort12: create a function that checks if the obj is a
        # cloud of 3D or 2D points
        # 3573
        return True

    def _check_options(self):
        """
        Check if nothing has to be plotted

        Returns
        -------
        :
            True if the `show_points` option is set to True, otherwise False.
        """
        return bool(self.options.show_points)

    def _populate_data(self, obj: npt.ArrayLike):
        points = _parse_to_xyz_array(obj).T
        self._data = points
        # apply rotation matrix given by options['view']
        self._projected_data = np.dot(
            np.c_[self._data, np.ones(len(self._data))],
            self.options.view_placement.to_matrix().T,
        ).T[0:2]

    def _make_plot_2d(self):
        if self.options.show_points:
            self.ax.scatter(*self._projected_data, **self.options.point_options)
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
        """
        Returns
        -------
        :
            Always returns True if the object is a `BluemiraWire`.

        Raises
        ------
        TypeError
            If the object is not an instance of `BluemiraWire`.
        """
        if not isinstance(obj, BluemiraWire):
            raise TypeError(f"{obj} must be a BluemiraWire")
        return True

    def _check_options(self):
        """
        Check if nothing has to be plotted

        Returns
        -------
        :
            True if nothing has to be plotted, otherwise False.
        """
        return not (not self.options.show_points and not self.options.show_wires)

    def _populate_data(self, obj: BluemiraWire):
        new_wire = obj.deepcopy()
        pointsw = new_wire.discretise(
            ndiscr=self.options._options.ndiscr, byedges=self.options._options.byedges
        ).T
        self._pplotter = PointsPlotter(self.options, data=pointsw)
        self._data = pointsw
        self._projected_data = self._pplotter._projected_data

    def _make_plot_2d(self):
        if self.options.show_wires:
            self.ax.plot(*self._projected_data, **self.options.wire_options)

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
    def _check_obj(obj: BluemiraFace):
        """
        Returns
        -------
        :
            Always returns True if the object is a `BluemiraFace`.

        Raises
        ------
        TypeError
            If the object is not an instance of `BluemiraFace`.
        """
        if not isinstance(obj, BluemiraFace):
            raise TypeError(f"{obj} must be a BluemiraFace")
        return True

    def _check_options(self):
        """
        Check if nothing has to be plotted

        Returns
        -------
        :
            True if nothing has to be plotted, otherwise False.
        """
        return not (
            not self.options.show_points
            and not self.options.show_wires
            and not self.options.show_faces
        )

    def _populate_data(self, obj: BluemiraFace):
        self._data = []
        self._wplotters = []
        for boundary in obj._plotting_wires():
            wplotter = WirePlotter(self.options, data=boundary)
            self._data.extend(wplotter._data.tolist())
            self._wplotters.append(wplotter)
        self._data = np.array(self._data)

        self._projected_data = [[], []]
        for w in self._wplotters:
            self._projected_data[0] += [*w._projected_data[0].tolist(), None]
            self._projected_data[1] += [*w._projected_data[1].tolist(), None]

    def _make_plot_2d(self):
        if self.options.show_faces:
            face_opts = self.options.face_options
            if face_opts.get("hatch", None) is not None:
                self.ax.add_patch(
                    Polygon(np.asarray(self._projected_data).T, fill=False, **face_opts)
                )
            else:
                self.ax.fill(*self._projected_data, **face_opts)

        for plotter in self._wplotters:
            plotter._ax = self.ax
            plotter._make_plot_2d()
        self._set_aspect_2d()

    def _make_plot_3d(self):
        if self.options.show_faces:
            poly = art3d.Poly3DCollection([self._data], **self.options.face_options)
            self.ax.add_collection3d(poly)

        for plotter in self._wplotters:
            plotter._ax = self.ax
            plotter._make_plot_3d()
        self._set_aspect_3d()


class ComponentPlotter(BasePlotter):
    """Base utility plotting class for bluemira faces"""

    _CLASS_PLOT_OPTIONS: ClassVar = {"show_points": False, "show_wires": False}

    @staticmethod
    def _check_obj(obj: Component):
        """
        Returns
        -------
        :
            Always returns True if the object is a `BluemiraComponent`.

        Raises
        ------
        TypeError
            If the object is not an instance of `BluemiraComponent`.
        """
        import bluemira.base.components  # noqa: PLC0415

        if not isinstance(obj, bluemira.base.components.Component):
            raise TypeError(f"{obj} must be a BluemiraComponent")
        return True

    def _check_options(self):
        """
        Check if nothing has to be plotted

        Returns
        -------
        :
            True if nothing has to be plotted, otherwise False.
        """
        return not (
            not self.options.show_points
            and not self.options.show_wires
            and not self.options.show_faces
        )

    def _create_plotters(self, comp: Component) -> Iterator[BasePlotter]:
        if comp.is_leaf and getattr(comp, "shape", None) is not None:
            if comp.plot_options.face_options["color"] in flatten_iterable(
                BLUE_PALETTE.as_hex()
            ):
                if self.options._external_options:
                    options = comp.plot_options
                else:
                    options = self.options
            else:
                options = comp.plot_options
            yield _get_plotter_class(comp.shape)(options, data=comp.shape)
        else:
            for child in comp.children:
                yield from self._create_plotters(child)

    def _populate_data(self, comp: Component):
        self._cplotters = list(self._create_plotters(comp))

    def _make_plot_2d(self):
        for plotter in self._cplotters:
            plotter._ax = self.ax
            plotter._make_plot_2d()
        self._set_aspect_2d()

    def _make_plot_3d(self):
        """
        Internal function that makes the plot. It should use self._data and
        self._projected_data, so _populate_data should be called before.
        """
        for plotter in self._cplotters:
            plotter._ax = self.ax
            plotter._make_plot_3d()

    def _set_aspect_3d(self):
        pass


def _validate_plot_inputs(
    parts: BluemiraGeoT | list[BluemiraGeoT],
    options: PlotOptions | list[None] | list[PlotOptions] | None,
) -> tuple[list[BluemiraGeoT], list[PlotOptions] | list[None]]:
    """
    Validate the lists of parts and options, applying some default options.

    Raises
    ------
    DisplayError
        Number of options not equal to number of parts

    Returns
    -------
    :
        validated lists of parts and options
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


def _get_plotter_class(part: npt.ArrayLike | Coordinates | BluemiraGeoT | Component):
    """
    Returns
    -------
    :
        the plotting class for a BluemiraGeo object.

    Raises
    ------
    DisplayError
        No plotter available for type of part
    """
    import bluemira.base.components  # noqa: PLC0415

    if isinstance(part, list | np.ndarray | Coordinates):
        plot_class = PointsPlotter
    elif isinstance(part, BluemiraWire):
        plot_class = WirePlotter
    elif isinstance(part, BluemiraFace):
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
    ax: Axes | None = None,
    *,
    show: bool = True,
    **kwargs,
) -> Axes:
    """
    The implementation of the display API for BluemiraGeo parts.

    Parameters
    ----------
    parts:
        The parts to display.
    options:
        The options to use to display the parts.
    ax:
        The axes onto which to plot
    show:
        Whether or not to show the plot immediately (default=True). Note
        that if using iPython or Jupyter, this has no effect; the plot is shown
        automatically.

    Returns
    -------
    :
        The axes with the plotted data.
    """
    _external_options = options is None
    parts, options = _validate_plot_inputs(parts, options)

    for part, option in zip(parts, options, strict=False):
        plotter = _get_plotter_class(part)(option, **kwargs)
        plotter.options._external_options = _external_options
        ax = plotter.plot_2d(part, ax, show=False)

    if show:
        plotter.show()

    return ax


def plot_3d(
    parts: BluemiraGeoT | Iterable[BluemiraGeoT],
    options: PlotOptions | Iterable[PlotOptions] | Iterable[None] | None = None,
    ax: Axes | None = None,
    *,
    show: bool = True,
    **kwargs,
) -> Axes:
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

    Returns
    -------
    :
        The axes with the plotted data.
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

    def plot_2d(self, ax: Axes | None = None, *, show: bool = True) -> Axes:
        """
        Default method to call display the object by calling into the Displayer's display
        method.

        Returns
        -------
        :
            The axes with the plotted data.
        """
        return self._plotter.plot_2d(self, ax=ax, show=show)

    def plot_3d(self, ax: Axes | None = None, *, show: bool = True) -> Axes:
        """
        Function to 3D plot a component.

        Returns
        -------
        :
            The axes with the plotted data.
        """
        return self._plotter.plot_3d(self, ax=ax, show=show)


def _get_ndim(coords: Coordinates) -> int:
    """
    Returns
    -------
    :
        The number of dimensions in the coordinate data. Returns at least 2.
    """
    count = 0
    length = coords.shape[1]
    for c in coords.xyz:
        if len(c) == length and not np.allclose(c, c[0]):
            count += 1

    return max(count, 2)


def _get_plan_dims(array: npt.ArrayLike) -> list[str]:
    """
    Returns
    -------
    :
        A sorted list of axis labels ("x", "y", "z") indicating which
        dimensions have variability.
    """
    axes = ["x", "y", "z"]
    dims = [k for i, k in enumerate(axes) if not np.allclose(array[i][0], array[i])]
    if len(dims) == 1:
        # Stops error when flat lines are given (same coords in two axes)
        axes.remove(dims[0])  # remove variable axis
        # both all equal to something
        temp = [k for i, k in enumerate(axes) if array[i][0] != 0.0]
        if len(temp) == 1:
            dims.append(temp[0])
        else:
            # This is likely due to a 3-5 long loop which is still straight
            # need to choose between one of the two constant dimensions
            # Just default to x - z, this is pretty rare..
            # usually due to an offset x - z loop
            dims = ["x", "z"]
    return sorted(dims)


def plot_coordinates(
    coords: Coordinates, ax: Axes | None = None, *, points: bool = False, **kwargs
):
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

    kwargs = {
        "edgecolor": ec,
        "facecolor": fc,
        "linewidth": lw,
        "linestyle": ls,
        "alpha": alpha,
        "fill": fill,
    }

    if ndim == 3 or (ndim == 2 and hasattr(ax, "zaxis")):  # noqa: PLR2004
        _plot_3d(coords, ax=ax, **kwargs)

    else:
        _plot_2d(coords=coords, ax=ax, points=points, **kwargs)


def _plot_3d(coords: Coordinates, ax: Axes | None = None, **kwargs):
    from bluemira.utilities.plot_tools import (  # noqa: PLC0415
        BluemiraPathPatch3D,
        Plot3D,
        coordinates_to_path,
    )

    if ax is None:
        ax = Plot3D()
        # Now we re-arrange a little so that matplotlib can show us something a little
        # more correct
        x_bb, y_bb, z_bb = BoundingBox.from_xyz(*coords.xyz).get_box_arrays()
        for x, y, z in zip(x_bb, y_bb, z_bb, strict=False):
            ax.plot([x], [y], [z], color="w")

    ax.plot(*coords.xyz, color=kwargs["edgecolor"], lw=kwargs["linewidth"])
    if kwargs["fill"]:
        if not coords.is_planar:
            bluemira_warn("Cannot fill plot of non-planar Coordinates.")
            return
        dcm = rotation_matrix_v1v2(-coords.normal_vector, np.array([0.0, 0.0, 1.0])).T

        xyz = dcm @ coords.xyz
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


def _plot_2d(coords: Coordinates, ax: Axes | None = None, *, points: bool, **kwargs):
    from bluemira.utilities.plot_tools import coordinates_to_path  # noqa: PLC0415

    if ax is None:
        ax = plt.gca()

    a, b = _get_plan_dims(coords.xyz)
    x, y = (getattr(coords, c) for c in [a, b])
    marker = "o" if points else None
    ax.set_xlabel(a + " [m]")
    ax.set_ylabel(b + " [m]")
    if kwargs["fill"]:
        ax.add_patch(
            PathPatch(
                coordinates_to_path(x, y),
                color=kwargs["facecolor"],
                alpha=kwargs["alpha"],
            )
        )

    ax.plot(
        x,
        y,
        color=kwargs["edgecolor"],
        marker=marker,
        linewidth=kwargs["linewidth"],
        linestyle=kwargs["linestyle"],
    )

    if points:
        for i, p in enumerate(zip(x, y, strict=False)):
            ax.annotate(i, xy=(p[0], p[1]))

    ax.set_aspect("equal")


def plot_2d_mesh_plt(
    nodes: np.ndarray,
    faces: np.ndarray,
    face_groups: np.ndarray | None = None,
    group_colors: dict[int, str] | None = None,
    cmap: str = "tab20",
    figsize: tuple = (6, 6),
    title: str = "2D Triangular Mesh",
    ax=None,
    *,
    show: bool = False,
) -> plt.Axes:
    """
    Plots a 2D triangular mesh with optional face group coloring.

    If no face_groups and no group_colors are provided, only the triangle edges
    will be plotted without face coloring.

    Parameters
    ----------
    nodes : np.ndarray
        Array of node coordinates, shape (N, 2)
    faces : np.ndarray
        Array of triangle indices, shape (M, 3)
    face_groups : np.ndarray, optional
        Face group IDs for coloring, shape (M,)
    group_colors : dict[int, str], optional
        Custom mapping from group ID to color
    cmap : str
        Colormap used for automatic color assignment
    figsize : tuple
        Figure size (only used if ax is None)
    title : str
        Title of the plot
    ax : matplotlib.axes.Axes, optional
        An existing Axes to plot on
    show : bool
        Whether to display the plot with plt.show()

    Returns
    -------
    matplotlib.axes.Axes
        The axes object the mesh is plotted on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    triangles = [nodes[face] for face in faces]

    if face_groups is not None:
        face_groups = np.asarray(face_groups)
        unique_groups = np.unique(face_groups)
        n_groups = len(unique_groups)

        if group_colors is None:
            color_map = cm.get_cmap(cmap, n_groups)
            group_colors = {group: color_map(i) for i, group in enumerate(unique_groups)}

        face_colors = [group_colors[group] for group in face_groups]

        collection = PolyCollection(
            triangles, facecolors=face_colors, edgecolors="k", linewidths=1
        )
        ax.add_collection(collection)

    else:
        # No face colors specified → draw only triangle edges
        edges = []
        for tri in triangles:
            edges.extend([[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]])
        edge_collection = LineCollection(edges, colors="k", linewidths=1)
        ax.add_collection(edge_collection)

    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(visible=True)

    if show:
        plt.show()

    return ax


def plot_dolfinx_2d_mesh_plt(
    mesh: dolfinx.mesh.Mesh,
    face_groups: np.ndarray | None = None,
    group_colors: dict[int, str] | None = None,
    cmap: str = "tab20",
    figsize: tuple = (6, 6),
    title: str = "2D Triangular Mesh (DOLFINx)",
    ax=None,
    *,
    show: bool = False,
) -> plt.Axes:
    """
    Plot a 2D triangular mesh from a DOLFINx mesh using matplotlib by leveraging
    `plot_2d_mesh_plt`.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        A 2D DOLFINx triangular mesh.
    face_groups : np.ndarray, optional
        Group IDs for each face (element), same length as number of cells.
    group_colors : dict[int, str], optional
        Mapping from group ID to color.
    cmap : str
        Colormap for automatic coloring of groups.
    show_nodes : bool
        If True, plot red dots at node positions.
    figsize : tuple
        Size of the figure if ax is None.
    title : str
        Title of the plot.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on.
    show : bool
        Whether to show the plot immediately.

    Raises
    ------
    ValueError
        If the mesh is not a triangular mesh.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object used for plotting.
    """
    # Extract node coordinates
    nodes = mesh.geometry.x[:, :2]  # Use only x and y

    # Ensure mesh is triangular
    if mesh.topology.cell_type != dolfinx.mesh.CellType.triangle:
        raise ValueError("Only triangular meshes are supported.")

    # Ensure connectivities are computed
    mesh.topology.create_connectivity(mesh.topology.dim, 0)

    # Extract cell-to-vertex connectivity (triangles)
    faces = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(-1, 3)

    # Delegate to general-purpose plotting function
    return plot_2d_mesh_plt(
        nodes=nodes,
        faces=faces,
        face_groups=face_groups,
        group_colors=group_colors,
        cmap=cmap,
        figsize=figsize,
        title=title,
        ax=ax,
        show=show,
    )

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
Plotting utilities for ReactorSystem objects
"""
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

import bluemira.codes as codes
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.auto_config import plot_defaults
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.geometry.geomtools import qrotate
from BLUEPRINT.utilities.colortools import color_kwargs

DEFAULTS = {"linewidth": 0.3, "edgecolor": "k", "alpha": 1}


class ReactorSystemPlotter:
    """
    The root plotting class for a ReactorSystem.
    """

    def __init__(self):
        self._palette_key = None

    def _get_next_plot_config(self, plot_config, index):
        """
        Support providing a list or cycle of plotting configuration arguments
        if we're plotting more than one object.
        Return the values according to the next point in the list or cycle, or
        just the plot config if it's not a list or cycle.

        Parameters
        ----------
        plot_config
            The plot config arguments to draw values from.
        index: int
            The index of the plot config to return.

        Returns
        -------
            The next plot_config value to be used
        """
        if isinstance(plot_config, cycle):
            return next(plot_config)
        if isinstance(plot_config, list):
            try:
                return plot_config[index]
            except IndexError:
                # Default to last config value if we've run out of config
                # values
                return plot_config[-1]
        else:
            # Just return the value as-is
            return plot_config

    def _apply_default_styling(self, kwargs):
        """
        Updates the facecolor, edgecolor and linewidth kwargs to use BP
        defaults if the values are not already set.

        Parameters
        ----------
        kwargs: dict
            The keyword arguments to apply the styling to.
        """
        [kwargs.setdefault(key, value) for key, value in DEFAULTS.items()]
        if self._palette_key:
            kwargs["facecolor"] = kwargs.get("facecolor", BLUE[self._palette_key])

    def _plot(self, plot_objects, ax=None, **kwargs):
        """
        The root plotter for ReactorSystem objects.
        Must deal with different colors.

        Parameters
        ----------
        plot_objects: Iterable
            The objects to be plotted.
        ax: Axes
            The optional Axes to plot onto.
        """
        if ax is None:
            ax = kwargs.get("ax", plt.gca())

        # Ensure default styling is applied if keyword args not already set.
        self._apply_default_styling(kwargs)

        nkwargs = kwargs.copy()
        for index, plot_object in enumerate(plot_objects):
            if "facecolor" in kwargs:
                nkwargs["facecolor"] = self._get_next_plot_config(
                    kwargs["facecolor"], index
                )
            if "alpha" in kwargs:
                nkwargs["alpha"] = self._get_next_plot_config(kwargs["alpha"], index)
            plot_object.plot(ax=ax, **color_kwargs(**nkwargs))

    def plot_xz(self, plot_objects, ax=None, **kwargs):
        """
        Plots the objects in the X-Z plane.

        Parameters
        ----------
        plot_objects: Iterable
            The objects to be plotted.
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
        self._plot(plot_objects, ax=ax, **kwargs)

    def plot_xy(self, plot_objects, ax=None, **kwargs):
        """
        Plots the objects in the X-Y plane.

        Parameters
        ----------
        plot_objects: Iterable
            The objects to be plotted.
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
        self._plot(plot_objects, ax=ax, **kwargs)


class ReactorPlotter:
    """
    Plotter object for Reactors

    Parameters
    ----------
    reactor: Reactor object
        The rector to plot
    palette: dict
        The palette dicitonary with short_name keys for sub-systems
    """

    def __init__(self, reactor, palette=BLUE):
        self.reactor = reactor
        self.palette = palette
        self.axxy = None
        self.axxz = None

    @staticmethod
    def set_defaults(force=False):
        """
        Set the defaults for the Reactor plot.
        """
        plt.rcdefaults()
        plt.rcParams["legend.fontsize"] = 14
        plt.rcParams["figure.figsize"] = [18, 18]
        plt.rcParams["lines.linewidth"] = 0.1
        plot_defaults(force=force)

    def plot_1D(self, width=1.0):
        """
        Plots a 1-D vector of the radial build output from PROCESS
        """
        codes.plot_radial_build(
            self.reactor.file_manager.generated_data_dirs["systems_code"], width=width
        )

    def plot_xz(self, x=None, z=None, show_eq=False, force=False):
        """
        Plots the X-Z cross-section of the reactor through the middle of a
        sector. Colors will be ditacted by the reactor palette object.

        NOTE: TF coils are plotted in the same plane and not rotated by
        n_TF/2

        Parameters
        ----------
        x: (float, float)
            The range of x coordinates to plot
        z: (float, float)
            The range of z coordinates to plot
        show_eq: bool (default = False)
            Whether or not to overlay plot an equilibrium
        """
        if z is None:
            z = [-22, 15]
        if x is None:
            x = [0, 22]
        ReactorPlotter.set_defaults(force=force)
        failed = []
        _, self.axxz = plt.subplots(figsize=[14, 10])
        for name in [
            "PL",
            "PF",
            "TF",
            "ATEC",
            "DIV",
            "BB",
            "VV",
            "TS",
            "CR",
            "RS",
            "STBB",
        ]:
            if hasattr(self.reactor, name):
                obj = getattr(self.reactor, name)

                if obj is None:
                    # System has not been built: cannot plot
                    failed.append(name)
                    continue

            else:
                failed.append(name)
                continue
            try:
                obj.plot_xz(ax=self.axxz)
            except KeyError:
                failed.append(name)
                continue
        self.axxz.set_xlim(x)
        self.axxz.set_ylim(z)
        self.axxz.set_aspect("equal", adjustable="box")

        if len(failed) > 0:
            self._fail_print("X-Z", failed)

        if show_eq:
            self.plot_equilibrium()
        return self.axxz

    def plot_equilibrium(self):
        """
        Searches for and plots a suitable plasma equilibrium result
        """
        flag = True
        if not hasattr(self.reactor, "eqref"):
            flag = False
        else:
            eq = self.reactor.eqref
        if hasattr(self.reactor, "EQ"):
            eq = self.reactor.EQ.snapshots["SOF"]
        else:
            if not flag:
                bluemira_warn(
                    "Systems::ReactorPlotter: No reference equilibrium to " "plot!"
                )
            else:
                pass
        if self.axxz is None:
            _, self.axxz = plt.subplots(figsize=[14, 10])
        eq.plot(ax=self.axxz)

    def plot_xy(self, x=None, y=None, force=False):
        """
        Plots the midplane x-y cross-section of the reactor as seen from above
        the upper port.

        Parameters
        ----------
        x: (float, float)
            The range of x coordinates to plot
        y: (float, float)
            The range of y coordinates to plot
        """
        if y is None:
            y = [-8, 8]
        if x is None:
            x = [1, 20]
        ReactorPlotter.set_defaults(force=force)
        failed = []
        _, self.axxy = plt.subplots(figsize=[14, 10])
        for name in ["PL", "BB", "TF", "VV", "TS", "CR", "RS"]:
            if hasattr(self.reactor, name):
                obj = getattr(self.reactor, name)
            else:
                failed.append(name)
                continue
            try:
                obj.plot_xy(ax=self.axxy)
            except KeyError:
                failed.append(name)
                continue
        self._draw_cl(alpha=0.5)
        self.axxy.set_xlim(x)
        self.axxy.set_ylim(y)
        self.axxy.set_aspect("equal")
        if len(failed) > 0:
            self._fail_print("X-Y", failed)

    def _fail_print(self, dims, failed):
        """
        Prints warning if certain ReactorSystems could not be plotted

        Parameters
        ----------
        dims: str
            The string of the dimensions of the plot
        failed: List(str)
            The short names list of the ReactorSystems that failed to plot
        """
        failed = ["\t" + n for n in failed]
        fstr = "\n".join(failed)
        bluemira_warn(
            f"Systems::ReactorPlotter: Failed to {dims} plot the follow"
            "ing:\n"
            f"{fstr}"
        )

    def _draw_cl(self, color="k", alpha=0.5, linewidth=2):
        """
        Draws the centreline division between reactor sectors in the X-Y plot
        """
        beta = np.pi / self.reactor.params.n_TF  # [°] sector half angle
        betas = np.arange(0 + beta, 2 * np.pi + beta, 2 * beta)
        # Sector centreline
        p1, p2 = [0, 0, 0], [30, 0, 0]
        for beta in betas:
            p2r = qrotate(p2, theta=beta, p1=p1, p2=[0, 0, 1])[0]
            x = [p1[0], p2r[0]]
            y = [p1[1], p2r[1]]
            self.axxy.plot(
                x, y, linestyle="--", linewidth=linewidth, color=color, alpha=alpha
            )

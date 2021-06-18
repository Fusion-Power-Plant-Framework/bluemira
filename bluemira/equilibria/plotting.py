# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
Plot utilities for equilibria
"""

import numpy as np
import matplotlib.pyplot as plt
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.plot_tools import str_to_latex
from bluemira.equilibria.constants import M_PER_MN


__all__ = ["GridPlotter", "ConstraintPlotter", "LimiterPlotter"]

PLOT_DEFAULTS = {
    "grid": {
        "edgewidth": 2,
        "linewidth": 1,
        "color": "k",
    },
    "limiter": {
        "marker": "o",
        "color": "b",
    },
    "coil": {
        "facecolor": {
            "PF": "#0098D4",
            "CS": "#003688",
            "Plasma": "r",
            "Passive": "grey",
        },
        "fontsize": 6,
    },
}


class Plotter:
    """
    Utility plotter abstract object
    """

    def __init__(self, ax=None, **kwargs):

        for kwarg in kwargs:
            if kwarg not in PLOT_DEFAULTS:
                bluemira_warn(f"Unrecognised plot kwarg: {kwarg}")

        if ax is None:
            f, self.ax = plt.subplots()
        else:
            self.ax = ax
        self.ax.set_xlabel("$x$ [m]")
        self.ax.set_ylabel("$z$ [m]")


class GridPlotter(Plotter):
    """
    Utility class for plotting Grid objects
    """

    def __init__(self, grid, ax=None, edge=False, **kwargs):
        super().__init__(ax)
        self.grid = grid
        self.plot_grid(**kwargs)
        if edge:
            self.plot_edge(**kwargs)

    def plot_grid(self, **kwargs):
        """
        Plots the gridlines of the grid
        """
        lw = kwargs.get("linewidth", PLOT_DEFAULTS["grid"]["linewidth"])
        color = kwargs.get("color", PLOT_DEFAULTS["grid"]["color"])
        for i in self.grid.x_1d:
            self.ax.plot([i, i], [self.grid.z_min, self.grid.z_max], color, linewidth=lw)
        for i in self.grid.z_1d:
            self.ax.plot([self.grid.x_min, self.grid.x_max], [i, i], color, linewidth=lw)

    def plot_edge(self, **kwargs):
        """
        Plots a thicker boundary edge for the grid
        """
        lw = kwargs.get("edgewidth", PLOT_DEFAULTS["grid"]["edgewidth"])
        color = kwargs.get("color", PLOT_DEFAULTS["grid"]["color"])
        self.ax.plot(*self.grid.bounds, color, linewidth=lw)


class ConstraintPlotter(Plotter):
    """
    Utility class for Constraint plotting.
    """

    def __init__(self, constraint_set, ax=None):
        super().__init__(ax)
        self.constraint_set = constraint_set

        for constraint in self.constraint_set.constraints:
            constraint.plot(self.ax)


class LimiterPlotter(Plotter):
    """
    Utility class for plotting Limiter objects
    """

    def __init__(self, limiter, ax=None, **kwargs):
        super().__init__(ax)
        self.limiter = limiter
        self.plot_limiter(**kwargs)

    def plot_limiter(self, **kwargs):
        """
        Plot the limiter onto the Axes.
        """
        color = kwargs.get("color", PLOT_DEFAULTS["limiter"]["color"])
        marker = kwargs.get("marker", PLOT_DEFAULTS["limiter"]["marker"])
        self.ax.plot(self.limiter.x, self.limiter.z, "s", color=color, marker=marker)


def _plot_coil(ax, coil, fill=True, **kwargs):
    """
    Single coil plot utility
    """
    mask = kwargs.pop("mask", True)

    fcolor = kwargs.pop("facecolor", PLOT_DEFAULTS["coil"]["facecolor"][coil.ctype])

    kwargs["color"] = kwargs.get("color", "k")
    x, z = (
        np.append(coil.x_corner, coil.x_corner[0]),
        np.append(coil.z_corner, coil.z_corner[0]),
    )
    ax.plot(x, z, zorder=11, **kwargs)
    if fill:
        if mask:
            ax.fill(x, z, color="w", zorder=10, alpha=1)
        kwargs["alpha"] = 0.5
        del kwargs["color"]
        ax.fill(x, z, zorder=10, color=fcolor, **kwargs)


def _annotate_coil(ax, coil, force=None, centre=None):
    """
    Single coil annotation utility function
    """
    off = max(0.2, coil.dx + 0.02)
    if coil.ctype == "CS":
        drs = -1.5 * off
        ha = "right"
    else:
        drs = 2 * off
        ha = "left"
    text = "\n".join([str_to_latex(coil.name), f"{coil.current/1E6:.2f} MA"])
    if force is not None:
        text = "\n".join([text, f"{force[1]/1E6:.2f} MN"])
    x = float(coil.x) + drs
    z = float(coil.z)
    if centre is not None:
        if coil.ctype == "PF":
            v = np.array([coil.x - centre[0], coil.z - centre[1]])
            v /= np.sqrt(sum(v ** 2))
            d = 1 + np.sqrt(2) * coil.dx
            x += d * v[0] - drs * 1.5
            z += d * v[1]
    ax.text(
        x,
        z,
        text,
        fontsize=PLOT_DEFAULTS["coil"]["fontsize"],
        ha=ha,
        va="center",
        color="k",
        backgroundcolor="white",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "linewidth": 1,
            "edgecolor": "k",
        },
    )


class CoilPlotter(Plotter):
    """
    Utility class for plotting individual Coils (for checking / testing only)

    Parameters
    ----------
    coil: Coil object
        The Coil to be plotted
    ax: Matplotlib axis object
        The ax on which to plot the Coil. (plt.gca() default)
    subcoil: bool
        Whether or not to plot the Coil subcoils
    label: bool
        Whether or not to plot labels on the coils
    force: None or np.array((1, 2))
        Whether to plot force vectors, and if so the array of force vectors
    """

    def __init__(self, coil, ax=None, subcoil=True, label=False, force=None):
        super().__init__(ax)
        self.coil = coil
        self.plot_coil(subcoil=subcoil)
        if label:
            _annotate_coil(self.ax, self.coil, force=force)

        if force is not None:
            d_fx, d_fz = force / M_PER_MN
            self.ax.arrow(self.coil.x, self.coil.z, 0, d_fz, color="r", width=0.1)

    def plot_coil(self, subcoil):
        """
        Plot a coil onto the Axes.
        """
        if subcoil:
            if self.coil.sub_coils is None:
                pass
            else:
                for name, sc in self.coil.sub_coils.items():
                    _plot_coil(self.ax, sc, fill=False)
        _plot_coil(self.ax, self.coil, fill=True)

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
from itertools import cycle
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.plot_tools import str_to_latex
from bluemira.equilibria.constants import M_PER_MN, J_TOR_MIN
from bluemira.equilibria.find import Xpoint
from bluemira.equilibria.physics import get_psi


__all__ = ["GridPlotter", "ConstraintPlotter", "LimiterPlotter", "CoilSetPlotter"]

PLOT_DEFAULTS = {
    "psi": {
        "nlevels": 15,
        "cmap": "viridis",
    },
    "field": {
        "nlevels": 15,
        "cmap": "magma",
    },
    "current": {
        "nlevels": 30,
        "cmap": "plasma",
    },
    "separatrix": {
        "color": "r",
        "linewidth": 3,
    },
    "opoint": {
        "marker": "o",
        "color": "g",
    },
    "xpoint": {
        "marker": "X",
        "color": "k",
    },
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
        "edgecolor": "k",
        "linewidth": 2,
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


class CoilSetPlotter(Plotter):
    """
    Utility class for plotting CoilSets

    Parameters
    ----------
    coilset: CoilSet object
        The CoilSet to be plotted
    ax: Matplotlib axis object
        The ax on which to plot the CoilSet. (plt.gca() default)
    subcoil: bool
        Whether or not to plot subcoils
    label: bool
        Whether or not to plot labels on the coils
    force: None or np.array((n_coils, 2))
        Whether to plot force vectors, and if so the array of force vectors
    """

    def __init__(
        self, coilset, ax=None, subcoil=False, label=True, force=None, **kwargs
    ):
        super().__init__(ax)
        self.coilset = coilset
        self.colors = kwargs.pop("facecolor", None)
        self.linewidth = kwargs.pop("linewidth", PLOT_DEFAULTS["coil"]["linewidth"])
        self.edgecolor = kwargs.pop("edgecolor", PLOT_DEFAULTS["coil"]["edgecolor"])
        if "alpha" in kwargs:
            # Alpha can be provided as a list or cycle to other systems, so make sure we
            # support that here.
            alpha = kwargs["alpha"]
            if isinstance(alpha, cycle):
                kwargs["alpha"] = next(alpha)
            if isinstance(kwargs["alpha"], list):
                kwargs["alpha"] = alpha[0]

        self.plot_coils(subcoil=subcoil, label=label, force=force, **kwargs)
        if label:  # Margins and labels fighting
            self.ax.set_xlim(left=-2)
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_ylim(bottom=ymin - 1)
            self.ax.set_ylim(top=ymax + 1)

    def plot_coils(self, subcoil=False, label=True, force=None, **kwargs):
        """
        Plots all coils in CoilSet.
        """
        centre = self.get_centre()
        for i, (name, coil) in enumerate(self.coilset.coils.items()):
            if self.colors is not None:
                if coil.ctype == "PF":
                    kwargs["facecolor"] = self.colors[0]
                elif coil.ctype == "CS":
                    kwargs["facecolor"] = self.colors[1]
            if not coil.control:
                coil.plot(  # include self.linewidth here?
                    ax=self.ax, label=False, **kwargs
                )
            else:
                coil_force = None if force is None else force[i]
                coil.plot(
                    ax=self.ax,
                    subcoil=subcoil,
                    color=self.edgecolor,
                    linewidth=self.linewidth,
                    label=label,
                    force=coil_force,
                    centre=centre,
                    **kwargs,
                )

    def get_centre(self):
        """
        Get a "centre" position for the coils to arrange the labels.
        """
        x, z = self.coilset.get_positions()
        xc = (max(x) + min(x)) / 2
        zc = (max(z) + min(z)) / 2
        return xc, zc


class EquilibriumPlotter(Plotter):
    """
    Utility class for Equilibrium plotting
    """

    def __init__(
        self,
        equilibrium,
        ax=None,
        plasma=False,
        show_ox=True,
        field=False,
    ):
        super().__init__(ax)
        self.eq = equilibrium

        # Do some housework
        self.psi = self.eq.psi()
        self.o_points, self.x_points = self.eq.get_OX_points(self.psi, force_update=True)

        if self.x_points:
            self.xp_psi = self.x_points[0][2]  # Psi at separatrix
            self.op_psi = self.o_points[0][2]  # Psi at O-point
        else:
            bluemira_warn(
                "No X-point found in plotted equilibrium. Cannot normalise psi."
            )
            self.xp_psi = np.amax(self.psi)
            self.op_psi = np.amin(self.psi)

        if not field:
            self.plot_plasma_current()
            self.plot_psi()
        else:
            self.plot_Bp()

        if self.o_points and self.x_points:
            # Only plot if we can normalise psi
            self.plot_separatrix()
            self.plot_flux_surface(1.05, "pink")

        if show_ox:
            self.plot_X_points()
            self.plot_O_points()

        if plasma:
            _plot_coil(self.ax, self.eq.plasma_coil())

    def plot_Bp(self, **kwargs):
        """
        Plots the poloidal field onto the Axes.
        """
        nlevels = kwargs.pop("nlevels", PLOT_DEFAULTS["field"]["nlevels"])
        cmap = kwargs.pop("cmap", PLOT_DEFAULTS["field"]["cmap"])

        Bp = self.eq.Bp()
        levels = np.linspace(1e-36, np.amax(Bp), nlevels)
        c = self.ax.contourf(self.eq.x, self.eq.z, Bp, levels=levels, cmap=cmap)
        cbar = plt.colorbar(c)
        cbar.set_label("$B_{p}$ [T]")

    def plot_psi(self, **kwargs):
        """
        Plot flux surfaces
        """
        nlevels = kwargs.pop("nlevels", PLOT_DEFAULTS["psi"]["nlevels"])
        cmap = kwargs.pop("cmap", PLOT_DEFAULTS["psi"]["cmap"])

        levels = np.linspace(np.amin(self.psi), np.amax(self.psi), nlevels)
        self.ax.contour(
            self.eq.x, self.eq.z, self.psi, levels=levels, cmap=cmap, zorder=8
        )

    def plot_plasma_current(self, **kwargs):
        """
        Plots flux surfaces inside plasma
        """
        if self.eq._jtor is None:
            return

        nlevels = kwargs.pop("nlevels", PLOT_DEFAULTS["current"]["nlevels"])
        cmap = kwargs.pop("cmap", PLOT_DEFAULTS["current"]["cmap"])

        levels = np.linspace(J_TOR_MIN, np.amax(self.eq._jtor), nlevels)
        self.ax.contourf(
            self.eq.x, self.eq.z, self.eq._jtor, levels=levels, cmap=cmap, zorder=7
        )

    def plot_flux_surface(self, psi_norm, color="k"):
        """
        Plots a normalised flux surface relative to the separatrix with
        increasing values going outwards from plasma core.
        """
        psi = get_psi(psi_norm, self.op_psi, self.xp_psi)
        self.ax.contour(
            self.eq.x, self.eq.z, self.psi, levels=[psi], colors=color, zorder=9
        )

    def plot_separatrix(self):
        """
        Plot the separatrix.
        """
        separatrix = self.eq.get_separatrix()
        if isinstance(separatrix, MultiLoop):
            loops = separatrix.loops
        else:
            loops = [separatrix]

        for loop in loops:
            x, z = loop.d2
            self.ax.plot(
                x,
                z,
                color=PLOT_DEFAULTS["separatrix"]["color"],
                linewidth=PLOT_DEFAULTS["separatrix"]["linewidth"],
                zorder=9,
            )

    def plot_X_points(self):  # noqa (N802)
        """
        Plot X-points.
        """
        for p in self.x_points:
            if isinstance(p, Xpoint):
                self.ax.plot(
                    p.x,
                    p.z,
                    marker=PLOT_DEFAULTS["xpoint"]["marker"],
                    color=PLOT_DEFAULTS["xpoint"]["color"],
                    zorder=10,
                )

    def plot_O_points(self):  # noqa (N802)
        """
        Plot O-points.
        """
        for p in self.o_points:
            self.ax.plot(
                p.x,
                p.z,
                marker=PLOT_DEFAULTS["opoint"]["marker"],
                color=PLOT_DEFAULTS["opoint"]["color"],
                zorder=10,
            )

    def plot_plasma_coil(self):
        """
        Plot the plasma coil.
        """
        pcoil = self.eq.plasma_coil()
        for coil in pcoil.values():
            _plot_coil(self.ax, coil)

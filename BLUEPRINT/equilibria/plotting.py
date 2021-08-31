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
Plot utilities for equilibria
"""
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RectBivariateSpline
import numpy as np
from itertools import cycle
from bluemira.base.look_and_feel import bluemira_warn
from BLUEPRINT.utilities.plottools import mathify
from BLUEPRINT.geometry.geomtools import grid_2d_contour
from BLUEPRINT.geometry.loop import Loop, MultiLoop
from BLUEPRINT.equilibria.find import Xpoint, Lpoint, get_contours
from BLUEPRINT.equilibria.constants import B_BREAKDOWN, M_PER_MN, J_TOR_MIN

# TODO: Implement matplotlib plot conventions for contours
# flux    ==> viridis
# field   ==> magma
# current ==> plasma
N_LEVELS = 15
LABEL_FS = 6


class Plotter:
    """
    Utility plotter abstract object
    """

    def __init__(self, ax):
        if ax is None:
            f, self.ax = plt.subplots()
        else:
            self.ax = ax
        self.ax.set_xlabel("$x$ [m]")
        self.ax.set_ylabel("$z$ [m]")
        if isinstance(self, ProfilePlotter):
            self.ax.set_xlabel("x/a")
            self.ax.set_ylabel("Value")

        elif isinstance(self, PulsePlotter):
            self.ax.set_xlabel("$\\psi$ [V.s]")
            self.ax.set_ylabel("I [MA]")

        elif isinstance(self, CorePlotter):
            pass
        else:
            self.ax.set_aspect("equal")

    def __call__(self):
        """
        Return the completed initial Axes
        """
        return self.ax

    def _plot_contour(self, array2d, nlevels=N_LEVELS, cmap="viridis"):
        """
        Base contour plotting utility
        """
        levels = np.linspace(np.amin(array2d), np.amax(array2d), nlevels)
        self.ax.contour(self.x, self.z, array2d, levels=levels, cmap=cmap, zorder=8)

    def _plot_quiver(self, xarray2d, zarray2d, cmap="viridis"):
        """
        Base quiver plotting utility
        """
        self.ax.quiver(self.x, self.z, xarray2d, zarray2d, cmap=cmap)

    def _plot_stream(self, xarray2d, zarray2d, color=None, cmap="RdBu"):
        """
        Base streamplotting utility
        """
        self.ax.streamplot(
            self.x_1d, self.z_1d, xarray2d.T, zarray2d.T, color=color, cmap=cmap
        )


def _plot_coil(ax, coil, fill=True, **kwargs):
    """
    Single coil plot utility
    """
    mask = kwargs.get("mask", True)
    if "mask" in kwargs.keys():
        del kwargs["mask"]
    if coil.ctype == "PF":
        fcolor = "#0098D4"
    elif coil.ctype == "CS":
        fcolor = "#003688"
    elif coil.ctype == "Passive":
        fcolor = "grey"
    else:  # Plasma
        fcolor = "r"
    fcolor = kwargs.get("facecolor", fcolor)
    if "facecolor" in kwargs:
        del kwargs["facecolor"]
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
    text = "\n".join([mathify(coil.name), f"{coil.current/1E6:.2f} MA"])
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
        fontsize=LABEL_FS,
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


class PulsePlotter(Plotter):
    """
    Utility class for plotting pulses
    """

    def __init__(self, coilset, ax=None):
        super(PulsePlotter, self).__init__(ax)
        self.cs = coilset
        if not hasattr(self.cs, "Iswing"):
            raise KeyError(
                "Run pulse on CoilSet first, cannot plot without " "pulse information."
            )
        self.plot_currents()

    def plot_currents(self):
        """
        Plot the currents in the pulse
        """
        for name, coil in self.cs.Iswing.items():
            self.ax.plot(np.array(coil) / 1e6, label=name)
        self.ax.legend()


class BreakdownPlotter(Plotter):
    """
    Utility class for Breakdown plotting
    """

    def __init__(self, breakdown, ax=None, Bp=False):
        super(BreakdownPlotter, self).__init__(ax)
        self.bd = breakdown

        self.x, self.z = self.bd.x, self.bd.z
        self.psi = self.bd.psi()
        self.psi_bd = self.bd.breakdown_psi
        self.Bp = self.bd.Bp(self.x, self.z)

        self.plot_contour()
        self.plot_zone()
        if Bp:
            self.plot_Bp()

    def plot_contour(self, nlevels=N_LEVELS):
        """
        Plot flux surfaces.
        """
        levels = np.linspace(self.psi_bd - 0.1, self.psi_bd, 3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ax.contour(self.x, self.z, self.psi, levels=levels, colors="r")

    def plot_Bp(self, nlevels=N_LEVELS):
        """
        Plots the poloidal field onto the Axes.
        """
        levels = np.linspace(1e-36, np.amax(self.Bp), nlevels)
        c = self.ax.contourf(self.x, self.z, self.Bp, levels=levels, cmap="magma")
        cbar = plt.colorbar(c)
        cbar.set_label("$B_{p}$ [T]")

    def plot_zone(self):
        """
        Plot the low field zones with a dashed line.
        """
        colors = ["b"]  # , 'g', 'b']
        self.ax.contour(
            self.x,
            self.z,
            self.Bp,
            levels=[B_BREAKDOWN],  # , 2*B_BREAKDOWN, 3*B_BREAKDOWN],
            colors=colors,
            linestyles="dashed",
        )

        if self.psi_bd is not None:
            self.ax.set_title("$\\psi_{b}$ = " + f"{2*np.pi*self.psi_bd:.2f} V.s")


class EquilibriumPlotter(Plotter):
    """
    Utility class for Equilibrium plotting
    """

    def __init__(
        self,
        equilibrium,
        ax=None,
        plasma=False,
        update_ox=False,
        show_ox=True,
        field=False,
    ):
        super(EquilibriumPlotter, self).__init__(ax)
        self.eq = equilibrium
        self.housework(update_ox)

        if not field:
            self.plot_plasma_current()
            self.plot_contour()
        else:
            self.plot_Bp()

        if self.Op and self.Xp:
            # Only plot if we can normalise psi
            self.plot_separatrix()
            self.plot_flux_surface(1.05, "pink")

        if show_ox:
            self.plot_X_points()
            self.plot_O_points()

        if plasma:
            _plot_coil(self.ax, self.eq.plasma_coil())

    def housework(self, update_ox):
        """
        Handle niggly details and get relevant values
        """
        self.psi = self.eq.psi()
        self.Op, self.Xp = self.eq.get_OX_points(self.psi, force_update=update_ox)
        try:
            self._psix = self.Xp[0][2]  # Psi at separatrix
            self._psio = self.Op[0][2]  # Psi at O-point
        except IndexError:
            bluemira_warn(
                "No X-point found in plotted equilibrium. " "Cannot normalise psi."
            )
            self._psix = 100
            self._psio = 0
        self.x, self.z = self.eq.x, self.eq.z
        self.Bp = self.eq.Bp()

    def plot_Bp(self, nlevels=N_LEVELS):
        """
        Plots the poloidal field onto the Axes.
        """
        levels = np.linspace(1e-36, np.amax(self.Bp), nlevels)
        c = self.ax.contourf(self.x, self.z, self.Bp, levels=levels, cmap="magma")
        cbar = plt.colorbar(c)
        cbar.set_label("$B_{p}$ [T]")

    def plot_contour(self, nlevels=N_LEVELS):
        """
        Plot flux surfaces
        """
        self._plot_contour(self.psi, nlevels=nlevels)

    def plot_plasma_current(self, nlevels=N_LEVELS * 2):
        """
        Plots flux surfaces inside plasma
        """
        if self.eq._jtor is None:
            return
        levels = np.linspace(J_TOR_MIN, np.amax(self.eq._jtor), nlevels)
        self.ax.contourf(
            self.x, self.z, self.eq._jtor, levels=levels, cmap="plasma", zorder=7
        )

    def plot_flux_surface(self, psi_norm, color="k"):
        """
        Plots a normalised flux surface relative to the separatrix with
        increasing values going outwards from plasma core.
        """
        psi = psi_norm * (self._psix - self._psio) + self._psio
        self.ax.contour(self.x, self.z, self.psi, levels=[psi], colors=color, zorder=9)

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
            self.ax.plot(x, z, color="r", linewidth=3, zorder=9)

    def plot_X_points(self):  # noqa (N802)
        """
        Plot X-points.
        """
        for p in self.Xp:
            if isinstance(p, Xpoint):
                self.ax.plot(p.x, p.z, marker="X", color="k", zorder=10)
            elif isinstance(p, Lpoint):
                pass

    def plot_O_points(self):  # noqa (N802)
        """
        Plot O-points.
        """
        for p in self.Op:
            self.ax.plot(p.x, p.z, marker="o", color="g", zorder=10)

    def plot_plasma_coil(self):
        """
        Plot the plasma coil.
        """
        pcoil = self.eq.plasma_coil()
        for coil in pcoil.values():
            _plot_coil(self.ax, coil)


class CorePlotter(Plotter):
    """
    Utility class for plotting equilibrium normalised radius characteristic
    profiles.
    """

    def __init__(self, dictionary, ax=None):
        r, c = int((len(dictionary) - 1) / 2) + 1, 2
        gs = GridSpec(r, c)
        self.ax = [plt.subplot(gs[i]) for i in range(r * c)]
        ccycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for i, (k, v) in enumerate(dictionary.items()):
            color = next(ccycle)
            self.ax[i].plot(
                dictionary["psi_n"], dictionary[k], label=mathify(k), color=color
            )
            self.ax[i].legend()


class CorePlotter2(Plotter):
    """
    Utility class for plotting plasma equilibrium cross-core profiles.
    """

    def __init__(self, eq, ax=None):
        jfunc = RectBivariateSpline(eq.x[:, 0], eq.z[0, :], eq._jtor)
        p = eq.pressure_map()
        pfunc = RectBivariateSpline(eq.x[:, 0], eq.z[0, :], p)
        o_points, _ = eq.get_OX_points()
        xmag, zmag = o_points[0].x, o_points[0].z
        psia, psib = eq.get_OX_psis()
        n = 50
        xx = np.linspace(eq.grid.x_min, eq.grid.x_max, n)
        zz = np.linspace(zmag, zmag, n)
        gs = GridSpec(3, 1)
        ccycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        psi = []
        for x, z in zip(xx, zz):
            psi.append(eq.psi(x, z)[0])
        psi = np.array(psi) * 2 * np.pi
        self.ax = [plt.subplot(gs[i]) for i in range(3)]
        self.ax[0].plot(xx, pfunc(xx, zz, grid=False), color=next(ccycle))
        self.ax[0].annotate("$p$", xy=[0.05, 0.8], xycoords="axes fraction")
        self.ax[0].set_ylabel("[Pa]")
        self.ax[1].plot(xx, jfunc(xx, zz, grid=False), color=next(ccycle))
        self.ax[1].set_ylabel("[A/m^2]")
        self.ax[1].annotate("$J_{\\phi}$", xy=[0.05, 0.8], xycoords="axes fraction")
        self.ax[2].plot(xx, psi, color=next(ccycle))
        self.ax[2].set_ylabel("[V.s]")
        self.ax[2].annotate("$\\psi$", xy=[0.05, 0.8], xycoords="axes fraction")
        self.ax[2].axhline(psib * 2 * np.pi, color="r", linestyle="--")
        for ax in self.ax:
            ax.axvline(xmag, color="r")


class ConstraintPlotter(Plotter):
    """
    Utility class for Constraint plotting.
    """

    _color = "b"

    def __init__(self, constraint_set, ax=None, **kwargs):
        super().__init__(ax)
        self._color = kwargs.get("color", self._color)
        self.constraint_set = constraint_set

        for constraint in self.constraint_set.constraints:
            constraint.plot(self.ax)


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
    centre: None or np.array((1, 2))
        Set the central point of the coil label
    """

    def __init__(
        self, coil, ax=None, subcoil=True, label=False, force=None, centre=None, **kwargs
    ):
        super().__init__(ax)
        self.coil = coil
        self.plot_coil(
            subcoil=subcoil,
            label=label,
            force=force,
            centre=centre,
            is_coilset=kwargs.pop("is_coilset", False),
            **kwargs,
        )

    def plot_coil(
        self,
        subcoil=True,
        label=False,
        fill=True,
        force=None,
        centre=None,
        linewidth=None,
        is_coilset=False,
        **kwargs,
    ):
        """
        Plot a coil onto the Axes.
        """
        if subcoil:
            if self.coil.sub_coils is None:
                if not is_coilset:
                    bluemira_warn(
                        "No sub-coils to plot. Use coil.mesh_coil(d_coil) to create sub-coils."
                    )
            else:
                for name, sub_coil in self.coil.sub_coils.items():
                    _plot_coil(
                        self.ax, sub_coil, fill=False, linewidth=linewidth, **kwargs
                    )
        _plot_coil(self.ax, self.coil, fill=fill, linewidth=linewidth, **kwargs)
        if force is not None:
            d_fx, d_fz = force / M_PER_MN
            self.ax.arrow(self.coil.x, self.coil.z, 0, d_fz, color="r", width=0.1)
        if label:
            self.annotate_coil(force=force, centre=centre)

    def annotate_coil(self, force=None, centre=None):
        """
        Annotate a coil (name, current, force).
        """
        _annotate_coil(self.ax, self.coil, force=force, centre=centre)


class PlasmaCoilPlotter(Plotter):
    """
    Utility class for plotting PlasmaCoils

    Parameters
    ----------
    plasma_coil: PlasmaCoil
        The PlasmaCoil to be plotted
    ax: Matplotlib axis object
        The ax on which to plot the PlasmaCoil. (plt.gca() default)
    """

    def __init__(self, plasma_coil, ax=None):
        super().__init__(ax)
        self.plasma_coil = plasma_coil
        if self.plasma_coil.j_tor is None:
            # No coils to plot
            pass
        else:
            contour = get_contours(
                self.plasma_coil.grid.x,
                self.plasma_coil.grid.z,
                self.plasma_coil.j_tor,
                J_TOR_MIN,
            )
            x, z = contour[0].T
            loop = Loop(x=x, z=z)
            sq_x, sq_z = grid_2d_contour(loop)
            levels = np.linspace(
                J_TOR_MIN, np.amax(self.plasma_coil.j_tor), N_LEVELS * 2
            )
            self.ax.contourf(
                self.plasma_coil.grid.x,
                self.plasma_coil.grid.z,
                self.plasma_coil.j_tor,
                cmap="plasma",
                levels=levels,
            )
            self.ax.plot(sq_x, sq_z, linewidth=1.5, color="k")


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
        self.linewidth = kwargs.pop("linewidth", 2)
        self.edgecolor = kwargs.pop("edgecolor", "k")
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

    def plot_coils(self, subcoil=False, passive=False, label=True, force=None, **kwargs):
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
                    ax=self.ax, label=False, is_coilset=True, **kwargs
                )
            else:
                if force is None:
                    coil.plot(
                        ax=self.ax,
                        subcoil=subcoil,
                        color=self.edgecolor,
                        linewidth=self.linewidth,
                        label=label,
                        centre=centre,
                        is_coilset=True,
                        **kwargs,
                    )
                else:
                    coil.plot(
                        ax=self.ax,
                        subcoil=subcoil,
                        color=self.edgecolor,
                        linewidth=self.linewidth,
                        label=label,
                        force=force[i],
                        centre=centre,
                        is_coilset=True,
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


class LimiterPlotter(Plotter):
    """
    Utility class for plotting Limiter objects
    """

    def __init__(self, limiter, ax=None):
        super(LimiterPlotter, self).__init__(ax)
        self.limiter = limiter
        self.plot_limiter()

    def plot_limiter(self):
        """
        Plot the limiter onto the Axes.
        """
        self.ax.plot(self.limiter.x, self.limiter.z, "s", color="r", marker="o")


class ProfilePlotter(Plotter):
    """
    Utility class for plotting profile objects
    """

    def __init__(self, profiles, ax=None):
        super(ProfilePlotter, self).__init__(ax)
        self.prof = profiles
        self.plot_profiles()

    def plot_profiles(self, n=50):
        """
        Plot the plasma profiles.
        """
        x = np.linspace(0, 1, n)
        self.ax.plot(x, self.prof.shape(x), label="shape function")
        self.ax.plot(x, self.prof.fRBpol(x) / max(self.prof.fRBpol(x)), label="fRBpol")
        self.ax.plot(
            x, self.prof.ffprime(x) / max(abs(self.prof.ffprime(x))), label="FFprime"
        )
        self.ax.plot(
            x, self.prof.pprime(x) / max(abs(self.prof.pprime(x))), label="pprime"
        )
        self.ax.plot(
            x, self.prof.pressure(x) / max(abs(self.prof.pressure(x))), label="pressure"
        )
        self.ax.legend()


class GridPlotter(Plotter):
    """
    Utility class for plotting Grid objects
    """

    def __init__(self, grid, ax=None, edge=False, **kwargs):
        super(GridPlotter, self).__init__(ax)
        self.grid = grid
        self.plot_grid(**kwargs)
        if edge:
            self.plot_edge()

    def plot_grid(self, **kwargs):
        """
        Plots the gridlines of the grid
        """
        lw = kwargs.get("linewidth", 1)
        for i in self.grid.x_1d:
            self.ax.plot([i, i], [self.grid.z_min, self.grid.z_max], "k", linewidth=lw)
        for i in self.grid.z_1d:
            self.ax.plot([self.grid.x_min, self.grid.x_max], [i, i], "k", linewidth=lw)

    def plot_edge(self):
        """
        Plots a thicker boundary edge for the grid
        """
        self.ax.plot(*self.grid.bounds, color="k", linewidth=2)


class MagFieldPlotter(Plotter):
    """
    Utility class for plotting magnetic fields
    """

    def __init__(self, magfield, ax=None):
        super(MagFieldPlotter, self).__init__(ax)
        self.m = magfield
        self.x, self.z = self.m.x, self.m.z
        self.x_1d, self.z_1d = self.m.x_1d, self.m.z_1d

    def plot_Bx_contour(self):
        """
        Plots the Bx field onto the Axes.
        """
        self._plot_contour(self.m.Bx, cmap="magma")

    def plot_Bz_contour(self):
        """
        Plots the Bz field onto the Axes.
        """
        self._plot_contour(self.m.Bz, cmap="magma")

    def plot_Bp_contour(self):
        """
        Plots the poloidal field onto the Axes.
        """
        self._plot_contour(self.m.Bp, cmap="magma")

    def plot_Bp_quiver(self):
        """
        Plots the poloidal field quiver map onto the Axes.
        """
        self._plot_quiver(self.m.Bx, self.m.Bz)

    def plot_Bp_streamplot(self):
        """
        Plots the poloidal field stream map onto the Axes.
        """
        self._plot_stream(self.m.Bx, self.m.Bz, color=self.m.Bp.T)

    def plot_Btor_contour(self):  # noqa (N802)
        """
        Plots the toroidal field onto the Axes.
        """
        self._plot_contour(self.m.Bt, cmap="magma")


class XZLPlotter(Plotter):
    """
    Utility class for plotting L constraints
    """

    def __init__(self, xzl_mapper, ax=None):
        super().__init__(ax)
        self.xzl = xzl_mapper

        for loop in self.xzl.excl_zones:
            loop.plot(self.ax, fill=True, alpha=0.2, facecolor="r", edgecolor="r")

        for loop in self.xzl.excl_loops:
            loop.plot(self.ax, fill=False, edgecolor="r", zorder=1, linestyle="--")

        for loop in self.xzl.incl_loops:
            loop.plot(self.ax, fill=False, edgecolor="k", zorder=1, linestyle="--")


class RegionPlotter(Plotter):
    """
    Utility class for plotting 2D L constraints
    """

    def __init__(self, region_mapper, ax=None):
        super().__init__(ax)
        self.rmp = region_mapper

        for intpltr in self.rmp.regions.values():
            intpltr.loop.plot(
                self.ax, fill=True, alpha=0.2, zorder=1, facecolor="g", edgecolor="g"
            )


if __name__ == "__main__":
    from BLUEPRINT import test

    test()

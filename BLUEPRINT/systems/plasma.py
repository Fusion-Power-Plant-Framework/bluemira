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
Plasma system
"""
import numpy as np
from typing import Type
import matplotlib.pyplot as plt

from bluemira.base.constants import MU_0
from bluemira.base.parameter import ParameterFrame

from BLUEPRINT.geometry.geomtools import loop_volume
from BLUEPRINT.geometry.loop import make_ring
from BLUEPRINT.systems.physicstoolbox import n_DT_reactions, r_T_burn, P_LH
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.cad.plasmaCAD import PlasmaCAD
from BLUEPRINT.systems.mixins import Meshable
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class Plasma(Meshable, ReactorSystem):
    """
    Plasma reactor system
    """

    config: Type[ParameterFrame]
    profiles: dict

    # fmt: off
    default_params = [
        ['R_0', 'Major radius', 9, 'm', None, 'Input'],
        ['B_0', 'Toroidal field at R_0', 6, 'T', None, 'Input'],
        ['A', 'Aspect ratio', 3.1, 'N/A', None, 'Input'],
        ['n_TF', 'Number of TF coils', 16, 'N/A', None, 'Input'],
        ['I_p', 'Plasma current', 20, 'MA', None, 'Input'],
        ['T_e', 'Electron temperature', 10, 'keV', None, 'Input'],
        ['P_fus', 'Total fusion power', 2000, 'MW', None, 'Input'],
        ['kappa_95', '95th flux surface elongation', 1.59, 'N/A', None, 'Input'],
        ['delta_95', '95th flux surface triangularity', 0.333, 'N/A', None, 'Input'],
        ['rho', 'Plasma density', 0.8, '10**20 /m^3', None, None],
        ['f_b', 'Burnup fraction', 0.03, 'N/A', None, None],
        ['V_p', 'Plasma volume', 2400, 'm^3', None, 'Input'],
        ['beta_p', 'Ratio of plasma pressure to poloidal magnetic pressure', 1.3, 'N/A', None, 'Input'],
        ['li', 'Normalised plasma internal inductance', 0.8, 'N/A', None, 'Input'],
        ['li3', 'Normalised plasma internal inductance (ITER def)', 0.8, 'N/A', None, 'Input'],
        ['Li', 'Plasma internal inductance', 1, 'H', None, 'Input'],
        ['Wp', 'Plasma energy', 2e6, 'MJ', None, 'Input'],
        ['delta', 'Plasma triangularity', 0.3, 'N/A', None, 'Input'],
        ['kappa', 'Plasma elongation', 1.6, 'N/A', None, 'Input'],
        ['shaf_shift', 'Shafranov shift of plasma (geometric=>magnetic)', 0.5, 'm', None, 'Input'],
        ['lambda_q', 'Scrape-off layer power decay length', 0.003, 'm', None, 'Input'],
        ['plasma_mode', 'Plasma mode', 'H', 'N/A', 'The plasma mode {H, L, or A}', 'Input'],
        ['ion_density_peaking_factor', 'Ion density peaking factor', 1, 'N/A', None, 'Fausser et. al. https://doi.org/10.1016/j.fusengdes.2012.02.025'],
        ['ion_temperature_peaking_factor', 'Ion temperature peaking factor', 8.06, 'N/A', None, 'Fausser et. al. https://doi.org/10.1016/j.fusengdes.2012.02.025'],
        ['ion_temperature_beta', 'Ion temperature beta', 6, 'N/A', 'The beta_T parameter, as described in the reference', 'Fausser et. al. https://doi.org/10.1016/j.fusengdes.2012.02.025'],
        ['ion_density_pedestal', 'Pedestal ion density', 1.09e20, 'm^-3', None, 'Fausser et. al. https://doi.org/10.1016/j.fusengdes.2012.02.025'],
        ['ion_density_separatrix', 'Separatrix ion density', 3e19, 'm^-3', None, 'Fausser et. al. https://doi.org/10.1016/j.fusengdes.2012.02.025'],
        ['ion_density_origin', 'Magnetic origin ion density', 1.09e20, 'm^-3', None, 'Fausser et. al. https://doi.org/10.1016/j.fusengdes.2012.02.025'],
        ['ion_temperature_pedestal', 'Pedestal ion temperature', 6.09, 'keV', None, 'Fausser et. al. https://doi.org/10.1016/j.fusengdes.2012.02.025'],
        ['ion_temperature_separatrix', 'Separatrix ion temperature', 0.1, 'keV', None, 'Fausser et. al. https://doi.org/10.1016/j.fusengdes.2012.02.025'],
        ['ion_temperature_origin', 'Magnetic origin ion temperature', 45.9, 'keV', None, 'Fausser et. al. https://doi.org/10.1016/j.fusengdes.2012.02.025'],
        ['relative_pedestal_radius', 'Pedestal radius relative to minor radius', 0.8, 'N/A', 'The pedestal radius as a fraction of the minor radius (where minor_radius = R_0 / A)', 'Fausser et. al. https://doi.org/10.1016/j.fusengdes.2012.02.025'],
    ]
    # fmt: on
    CADConstructor = PlasmaCAD

    def __init__(self, config, profiles, method):
        self.config = config
        self.profiles = profiles
        self.method = method
        self._plotter = PlasmaPlotter()

        self._init_params(self.config)

        if len(self.profiles) != 0:
            self.profiles = self.profiles.copy()
            self._adjust_psi()
            self._get_current()
        self.n_TF = self.params.n_TF
        self.derive_fuelling_requirements()

    def _adjust_psi(self):
        self.profiles["psi"] = 1 * self.profiles["psi"]
        self.Mpsi = self.profiles["psi"][0]
        self.Xpsi = self.profiles["psi"][-1]
        self.norm = 1
        self.b_scale = 1

    def _get_current(self):
        """
        Sums current profiles contributions
        """
        i = np.sum([self.profiles[v] for v in ["j_parallel", "j_bs", "j_cd"]], axis=0)
        self.profiles["j_tot"] = i

    def update_separatrix(self, separatrix):
        """
        Ajuste o objeto Plasma com a informação addicional do GS solver
        """
        self.geom["Separatrix"] = separatrix

    def update_LCFS(self, lcfs):
        """
        Ajuste o objeto Plasma com a informação addicional do GS solver
        """
        self.geom["LCFS"] = lcfs
        self._get_params()

    def export_neutron_source(self):
        """
        Creates a dictionary of source term parameters for neutronics models

        Returns
        -------
        d: dict
            The dictionary of parameters for the neutron source term
        """
        minor_radius = (self.params.R_0 / self.params.A) * 100.0
        pedestal_radius = self.params.relative_pedestal_radius * minor_radius
        d = {
            "elongation": self.params.kappa_95,
            "triangularity": self.params.delta_95,
            "minor_radius": minor_radius,  # [cm]
            "major_radius": self.params.R_0 * 100.0,  # [cm]
            "ion_density_peaking_factor": self.params.ion_density_peaking_factor,
            "plasma_mode": self.params.plasma_mode,
            "shafranov_shift": self.params.shaf_shift * 100.0,  # [cm]
            "ion_density_pedestal": self.params.ion_density_pedestal,
            "ion_density_separatrix": self.params.ion_density_separatrix,
            "ion_density_origin": self.params.ion_density_origin,
            "ion_temperature_pedestal": self.params.ion_temperature_pedestal,
            "ion_temperature_separatrix": self.params.ion_temperature_separatrix,
            "ion_temperature_origin": self.params.ion_temperature_origin,
            "pedestal_radius": pedestal_radius,
            "ion_temperature_peaking_factor": self.params.ion_temperature_peaking_factor,
            "ion_temperature_beta": self.params.ion_temperature_beta,
        }
        return d

    def derive_fuelling_requirements(self):
        """
        Calculate the plasma fuelling requirements for the specified fusion power.
        """
        n = n_DT_reactions(self.params.P_fus)
        burn_rate = r_T_burn(self.params.P_fus)
        # fmt: off
        p = [
            ["n_DT_reactions", "Fusion reaction rate", n, "1/s", "At full power", "Derived input"],
            ["T_b", "Tritium burn rate", burn_rate, "g/s", None, "Derived input"],
        ]
        # fmt: on
        self.add_parameters(p)

    def get_boundary(self):
        """
        Extracts plasma LCFS
        """
        xz = np.zeros([2, len(self.X.T)])
        for i, (x, z) in enumerate(zip(self.X.T, self.Z.T)):
            xz[:, i] = [x[-1], z[-1]]
        self.xbdry, self.zbdry = xz
        self.nbdry = len(self.xbdry)

    def get_sep(self):
        """
        Get the separatrix profile.

        Returns
        -------
        separatrix: Loop
            The loop for the separatrix.
        """
        return self.geom["Separatrix"]

    def get_LCFS(self):  # noqa (N802)
        """
        Get the last closed flux surface profile.

        Returns
        -------
        lcfs: Loop
            The loop for the LCFS.
        """
        return self.geom["LCFS"]

    def plot_profiles(self, titles=False):
        """
        Plot the plasma profiles.
        """
        # TODO FIX
        # p = self.profiles
        f, ax = plt.subplots(2, 2, figsize=[13, 9])
        # Temperature profiles
        self._plot_temp(ax[0, 0], titles=titles)
        # Current profiles
        self._plot_current(ax[0, 1], titles=titles)
        # Density profile
        self._plot_dens(ax[1, 0], titles=titles)
        # Flux profiles
        self._plot_q_k(ax[1, 1], titles)
        # self._plot_flux(ax[1, 1], titles)
        # ax[1, 1].plot(p['x'], p['q'], label='$q$')

    def _plot_q_k(self, ax, titles=False):
        ax1 = ax
        ax2 = ax1.twinx()
        self._plot_p(ax1, "q", "$q$", "")
        self._plot_p(
            ax2,
            "kappa",
            "${\\kappa}$",
            "",
            color=next(ax1._get_lines.prop_cycler)["color"],
        )

        if titles:
            ax2.set_title("$q$ and ${\\kappa}$ profiles")
        ax2.set_ylim([0.95 * min(self.profiles["kappa"]), 2])
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right", bbox_to_anchor=[0.95, 1])

    def _plot_flux(self, ax, titles=False):
        unit = "${\\psi}$ [$Wb$]"
        self._plot_p(ax, "psi", "$\\psi$", unit)
        if titles:
            ax.set_title("Flux profile")
        ax.legend()

    def _plot_dens(self, ax, titles=False):
        unit = "$n$ [$10^{19}/m^{3}$]"
        self._plot_p(ax, "n_e", "$n_{e}$", unit)
        self._plot_p(ax, "n_ions", "$n_{i}$", unit)
        if titles:
            ax.set_title("Density profiles")
        ax.legend()

    def _plot_temp(self, ax, titles):
        unit = "$T$ [keV]"
        for k, lab in zip(["T_e", "T_i"], ["$T_{e}$", "$T_{i}$"]):
            self._plot_p(ax, k, lab, unit)
        if titles:
            ax.set_title("Temperature profiles")
        ax.legend()

    def _plot_current(self, ax, titles=False):
        unit = "$j$ [$MA/m^{2}$]"
        for k, lab in zip(
            ["j_parallel", "j_bs", "j_cd"], ["$j_{//}$", "$j_{bs}$", "$j_{CD}$"]
        ):
            self._plot_p(ax, k, lab, unit)
        if titles:
            ax.set_title("Current profiles")
        ax.legend()

    def _plot_p(self, ax, key, label, ylabel, **kwargs):
        ax.plot(self.profiles["x"], self.profiles[key], label=label, **kwargs)
        # 'r' is the BLUEPRINT coordinate system for the
        # plasma-centric quasi-toroidal coordinate system
        ax.set_xlabel("r/a")
        ax.set_ylabel(ylabel)
        ax.set_xlim([-0.025, 1.025])
        ax.set_xticks(np.arange(0, 1.2, 0.2))

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        return ["LCFS"]

    @property
    def xy_plot_loop_names(self):
        """
        The x-y loop names to plot.
        """
        return ["LCFS X-Y"]

    def _generate_xy_plot_loops(self):
        shell = make_ring(min(self.geom["LCFS"].x), max(self.geom["LCFS"].x))
        self.geom["LCFS X-Y"] = shell
        return super()._generate_xy_plot_loops()

    def _get_params(self):
        # fmt: off
        p = [
            ["A_p", "Plasma cross-sectional area", self.geom["LCFS"].area, "m^2", None, None],
            ["V_p", "Plasma volume", loop_volume(*self.geom["LCFS"].d2), "m^3", None, None],
            ["res_plasma", "Plasma resistance", self.plasma_resistance, "Ohm", None, "Uckan et al."],
            ["P_LH", "L-H transition power", self.calc_p_lh(), "MW", "Martin scaling", None]
        ]
        # fmt: on
        self.add_parameters(p)

    def calc_p_lh(self):
        """
        Calculate the L-H transition power for the plasma.

        Returns
        -------
        P_LH: float
            The L-H transition power [MW]
        """
        a = self.params.R_0 / self.params.A
        return P_LH(self.params.rho * 1e20, self.params.B_0, a, self.params.R_0)

    @property
    def plasma_resistance(self):
        # TODO UPDATE:  EF says this is wrong
        """
        Plasma resistance, from loop voltage calculation in IPDG89 (??)

        Returns
        -------
        R_plasma: float
            The plasma resistance [Ohm]

        Notes
        -----
        Taken from PROCESS.
        """
        a = self.params.R_0 / self.params.A
        kappa = self.params.kappa
        r_plasma = (
            2.15e-9
            * self.config["Z_eff"]
            * self.params.R_0
            / (kappa * a ** 2 * (self.params.T_e / 10) ** 1.5)
        )  # [Ohms]

        rpfac = 4.3 - 0.6 * self.params.R_0 / a
        return r_plasma * rpfac

    def burn_voltage(self, f_ohm, f_sawtooth_control=1):
        """
        Calculate the loop voltage during flat-top.

        Parameters
        ----------
        f_ohm: float
            The Ohmic fraction of current drive

        f_sawtooth_control: float
            The "enhancement factor" in flattop V.s requirement to account for
            MHD sawtooth effects. Defaults to 1.

        Returns
        -------
        v_burn: float
            The flat-top burn loop voltage [V]
        """
        v_burn = (
            self.params.I_p
            * self.params.res_plasma
            * f_ohm
            * f_sawtooth_control
            * 10 ** 6
        )
        return v_burn

    def resistive_losses(self, ejima_gamma=0.3):
        """
        Calculate the start-up resistive losses [V.s/Wb].

        Parameters
        ----------
        Ejima coefficient: float
            The Ejima coefficient to use

        Returns
        -------
        resistive_flux_swing_loss: float
            The resistive flux swing loss [V.s/Wb]

        Notes
        -----
        ITER formula without 10 V.s buffer
        """
        resistive_flux_swing_loss = (
            ejima_gamma * MU_0 * self.params.R_0 * self.params.I_p * 1e6
        )
        return resistive_flux_swing_loss


class PlasmaPlotter(ReactorSystemPlotter):
    """
    The plotter for a Plasma.
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "PL"

    def plot_xz(self, plot_objects, ax=None, **kwargs):
        """
        Plot the Plasma in the x-z plane.
        """
        kwargs["alpha"] = kwargs.get("alpha", 0.5)
        super().plot_xz(plot_objects, ax=ax, **kwargs)

    def plot_xy(self, plot_objects, ax=None, **kwargs):
        """
        Plot the Plasma in the x-y plane.
        """
        kwargs["alpha"] = kwargs.get("alpha", 0.15)
        super().plot_xy(plot_objects, ax=ax, **kwargs)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()

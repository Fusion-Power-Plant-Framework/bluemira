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
Main interface for building and loading equilibria and coilset designs
"""

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, concat
import tabulate

from bluemira.equilibria.constants import (
    NB3SN_B_MAX,
    NB3SN_J_MAX,
    NBTI_B_MAX,
    NBTI_J_MAX,
    B_BREAKDOWN,
)
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.base.file import try_get_bluemira_path
from bluemira.utilities.plot_tools import make_gif
from bluemira.utilities.tools import abs_rel_difference
from bluemira.equilibria.positioner import CoilPositioner
from bluemira.equilibria.equilibrium import Equilibrium, Breakdown
from bluemira.equilibria.profiles import (
    BetaIpProfile,
    LaoPolynomialFunc,
    DoublePowerFunc,
    ShapeFunction,
    CustomProfile,
)
from bluemira.equilibria.physics import calc_psib, calc_li, calc_betap
from bluemira.equilibria.limiter import Limiter
from bluemira.equilibria.constraints import (
    AutoConstraints,
    EUDEMOSingleNullConstraints,
    EUDEMODoubleNullConstraints,
)
from bluemira.equilibria.optimiser import (
    Norm2Tikhonov,
    BreakdownOptimiser,
    PositionOptimiser,
)
from bluemira.equilibria.solve import (
    PicardLiDeltaIterator,
    PicardLiAbsIterator,
    PicardDeltaIterator,
    PicardAbsIterator,
    EquilibriumConverger,
)
from bluemira.equilibria.grid import Grid


class Snapshot:
    """
    Abstract object for grouping of equilibria objects in a given state.

    Parameters
    ----------
    eq: Equilibrium object
        The equilibrium at the snapshot
    coilset: CoilSet
        The coilset at the snapshot
    constraints: Constraints object
        The constraints at the snapshot
    profiles: Profile object
        The profile at the snapshot
    optimiser: EquilibriumOptimiser object
        The optimiser for the snapshot
    limiter: Limiter object
        The limiter for the snapshot
    tfcoil: Loop object
        The PF coil placement boundary
    """

    def __init__(
        self,
        eq,
        coilset,
        constraints,
        profiles,
        optimiser=None,
        limiter=None,
        tfcoil=None,
    ):
        self.eq = eq.copy()
        self.coilset = coilset.copy()
        if constraints is not None:
            self.constraints = constraints.copy()
        else:
            self.constraints = None
        if profiles is not None:
            self.profiles = profiles.copy()
        else:
            self.profiles = None
        if limiter is not None:
            self.limiter = limiter.copy()
        else:
            self.limiter = None
        if optimiser is not None:
            self.optimiser = optimiser.copy()
        else:
            self.optimiser = None
        self.tf = tfcoil

    def plot(self, ax=None):
        """
        Plots a Snapshot object, including:
            equilibrium
            constraints
            limiters
            coilset
            TF coil boundary

        Parameters
        ----------
        ax: Axes object
            The matplotlib axes on which to plot
        """
        if ax is None:
            ax = plt.gca()
        self.eq.plot(ax)
        if self.limiter is not None:
            self.limiter.plot(ax)
        if self.tf is not None:
            self.tf.plot(ax, fill=False)
        if self.constraints is not None:
            self.constraints.plot(ax)
        self.coilset.plot(ax)


class EquilibriumProblem:
    """
    Abstract base class for equilibrium problems

    Subclasses require:
        .coilset
        .eq
        .profiles
        .constraints
        .optimiser
        .lim
    """

    def __init__(self):
        self.n_swing = 2  # Default number of swing snapshots
        self.tfcoil = None
        self.eq = None
        self.profiles = None
        self.constraints = None
        self.optimiser = None
        self.p_optimiser = None
        self.coilset = None
        self.lim = None
        self.eqtype = None

        self.nx = None
        self.nz = None

        self.R_0 = None
        self.kappa = None
        self.delta = None
        self.A = None
        self.betap = None
        self.li = None
        self.Ip = None
        self.B_0 = None
        self.q95min = None
        self.eqtype = None
        self.c_ejima = None
        self.psi_bd = None

        self.old = None
        self._refpsi = None
        self.snapshots = {}

    def solve(self, constrained=False, plot=True, gif=False):
        """
        Iteratively solves the non-linear Grad-Shafranov equation over the
        starting grid, to produce an Equilibrium

        Parameters
        ----------
        constrained: bool
            Whether or a not a constrained optimiser has been specified. In
            practice, whether dI vs. I is being optimised in the Picard
            iterations.
        plot: bool
            Plots iterative solution as it progresses
        gif: bool
            Creates a GIF of the iteration solution progression

        Returns
        -------
        self.eq: Equilibrium object
            A deep-copied Equilibrium object representing the state of affairs
            from the solve operation.
        """
        args = (self.eq, self.profiles, self.constraints, self.optimiser)
        kwargs = {"relaxation": 0.1, "plot": plot, "gif": gif}
        if not constrained:
            if self.li is not None:
                iterator = PicardLiDeltaIterator(*args, **kwargs)
            else:
                iterator = PicardDeltaIterator(*args, **kwargs)
        else:
            if self.li is not None:
                iterator = PicardLiAbsIterator(*args, **kwargs)
            else:
                iterator = PicardAbsIterator(*args, **kwargs)
        iterator()
        self.coilset.adjust_sizes()
        self.eq._remap_greens()
        if self.li is None:
            self.li = self.eq.calc_li()
        return self.eq.copy()

    def update_psi(self):
        """
        Ronseal function
        """
        self._refpsi = self.eq.psi()

    def plot_summary(self):
        """
        Plots a summary of the flux swing pulse, showing:
            SOF: Start of flat-top
            MOF: Middle of flat-top
            EOF: End of flat-top
        """
        if self.snapshots is None:
            raise ValueError("Please run a flux swing before plotting " "a summary.")
        f, ax = plt.subplots(1, len(self.snapshots))
        for i, (name, snap) in enumerate(self.snapshots.items()):
            snap.plot(ax[i])
            if name == "Breakdown":
                psi_b = snap.eq.breakdown_psi
            else:
                psi_b = snap.eq.get_OX_psis()[1]
            ax[i].set_title(f"{name}" + "\n$\\psi_{b}$ = " f"{2*np.pi*psi_b:.2f} V.s")

    def take_snapshot(self, key, equilibrium, **kwargs):
        """
        Parameters
        ----------
        key: str
            Snapshot key name
        equilibrium: Equilibrium Object
            The equilibrium to be stored
        """
        coilset = kwargs.get("coilset", self.coilset)
        constraints = kwargs.get("constraints", self.constraints)
        profiles = kwargs.get("profiles", self.profiles)
        optimiser = kwargs.get("optimiser", self.optimiser)
        limiter = kwargs.get("limiter", self.lim)
        tfcoil = kwargs.get("tfcoil", self.tfcoil)
        self.snapshots[key] = Snapshot(
            equilibrium, coilset, constraints, profiles, optimiser, limiter, tfcoil
        )

    def breakdown(
        self,
        PF_Fz_max,
        CS_Fz_sum,
        CS_Fz_sep,
        x_zone=None,
        z_zone=None,
        r_zone=None,
        b_zone_max=B_BREAKDOWN,
    ):
        """
        Initialises a plasma breakdown to determine peak magnetic flux prior
        to ramp-up.

        Parameters
        ----------
        PF_Fz_max: float
            The maximum absolute vertical on a PF coil [N]
        CS_Fz_sum: float
            The maximum absolute vertical on all CS coils [N]
        CS_Fz_sep: float
            The maximum Central Solenoid vertical separation force [N]
        r_zone: float
            The radius of the zone in which to generate a very low poloidal
            magnetic field [m]
        b_zone_max: float
            The maximum poloidal magnetic field in the zone [T]
        """
        if x_zone is None:
            x_zone = 1.09 * self.R_0
        if z_zone is None:
            z_zone = 0.0
        if r_zone is None:
            r_zone = 0.5 * self.R_0 / self.A  # Augenapfel

        bluemira_print("EQUILIBRIA: Calculating plasma breakdown flux")
        max_currents = self.coilset.get_max_currents(self.Ip * 1.4)
        max_fields = self.coilset.get_max_fields()
        optimiser = BreakdownOptimiser(
            x_zone,
            z_zone,
            r_zone=r_zone,
            b_zone_max=b_zone_max,
            max_currents=max_currents,
            max_fields=max_fields,
            PF_Fz_max=PF_Fz_max,
            CS_Fz_sum=CS_Fz_sum,
            CS_Fz_sep=CS_Fz_sep,
        )

        grid = Grid(0.1, self.R_0 * 2, -1.5 * self.R_0, 1.5 * self.R_0, 100, 100)
        bd = Breakdown(self.coilset, grid, psi=None, R_0=self.R_0)
        bd.set_breakdown_point(x_zone, z_zone)
        self.coilset.reset()
        cs_names = self.coilset.get_CS_names()
        for name in cs_names:
            max_currents = self.coilset.coils[name].get_max_current()
            self.coilset.coils[name].current = max_currents
            self.coilset.coils[name].mesh_coil(0.3)
        pf_names = self.coilset.get_PF_names()
        for name in pf_names:
            self.coilset.coils[name].current = self.Ip
            self.coilset.coils[name].make_size()
            self.coilset.coils[name].fix_size()
            self.coilset.coils[name].mesh_coil(0.4)
        bd._remap_greens()
        currents = optimiser(bd)
        self.coilset.set_control_currents(currents)

        for name in pf_names:
            self.coilset.coils[name].flag_sizefix = False

        bd._remap_greens()
        self.psi_bd = bd.breakdown_psi * 2 * np.pi
        bluemira_print(f"EQUILIBRIA: breakdown psi = {self.psi_bd:.2f} V.s")
        self.take_snapshot(
            "Breakdown",
            bd,
            limiter=None,
            constraints=None,
            tfcoil=None,
            optimiser=optimiser,
        )

    def calculate_flux_swing(
        self,
        tau_flattop,
        v_burn,
        psi_bd=None,
        PF_Fz_max=None,
        CS_Fz_sum=None,
        CS_Fz_sep=None,
    ):
        """
        Parameters
        ----------
        tau_flattop: float
            Flat-top duration [s]
        v_burn: float
            Plasma loop voltage during burn [V]

        Returns
        -------
        psi_sof, psi_eof: float, float
            Plasma boundary flux values at start of and end of flat-top [V.s]
        """
        if psi_bd is None:
            if self.psi_bd is None:
                self.breakdown(PF_Fz_max, CS_Fz_sum, CS_Fz_sep)
            psi_bd = self.psi_bd
        psi_sof = calc_psib(psi_bd, self.R_0, self.Ip, self.li, self.c_ejima)
        psi_eof = psi_sof - tau_flattop * v_burn
        return psi_sof, psi_eof

    def optimise_positions(
        self,
        max_PF_current,
        PF_Fz_max,
        CS_Fz_sum,
        CS_Fz_sep,
        tau_flattop,
        v_burn,
        psi_bd=None,
        pfcoiltrack=None,
        pf_exclusions=None,
        pf_coilregions=None,
        CS=False,
        plot=True,
        gif=False,
        figure_folder=None,
    ):
        """
        Optimises the positions of the PF coils for the EquilibriumProblem.

        Parameters
        ----------
        max_PF_current: float
            Maximum PF coil current [A]
        PF_Fz_max: float
            The maximum absolute vertical on a PF coil [N]
        CS_Fz_sum: float
            The maximum absolute vertical on all CS coils [N]
        CS_Fz_sep: float
            The maximum Central Solenoid vertical separation force [N]
        tau_flattop: float
            The desired flat-top length [s]
        v_burn: float
            The plasma loop voltage during burn [V]
        psi_bd: float (default = None)
            The plasma boundary magnetic flux value at breakdown [V.s]
        pfcoiltrack: Loop
            The track along which the PF coil positions are optimised
        pf_exclusions: list(Loop, Loop, ..)
            Set of exclusion zones to apply to the PF coil track
        pf_coilregions: dict(coil_name:Loop, coil_name:Loop, ...)
            Regions in which each PF coil resides. The loop object must be 2d.
        CS: bool (default = False)
            Whether or not to optimise the CS module positions as well
        plot: bool (default = False)
            Plot progress
        gif: bool (default = False)
            Save figures and make gif
        figure_folder: str (default = None)
            The path where figures will be saved. If the input value is None (e.g.
            default) then this will be reinterpreted as the path data/plots/equilibria
            under the bluemira root folder, if that path is available.

        Note
        ----
        Modifies:
            .coilset
            .eq

        """
        self.tfcoil = pfcoiltrack  # Only necessary from AbExtra
        swing = self.calculate_flux_swing(
            tau_flattop,
            v_burn,
            psi_bd=psi_bd,
            PF_Fz_max=PF_Fz_max,
            CS_Fz_sum=CS_Fz_sum,
            CS_Fz_sep=CS_Fz_sum,
        )
        psis = np.linspace(*swing, self.n_swing)
        psis /= 2 * np.pi  # Per rad für die Optimierung

        if CS is True:
            solenoid = self.coilset.get_solenoid()
            CS_x = solenoid.radius
            CS_zmin = solenoid.z_min
            CS_zmax = solenoid.z_max
            CS_gap = solenoid.gap
        else:  # CS divisions will not be optimised
            CS_x = None
            CS_zmin = None
            CS_zmax = None
            CS_gap = None

        if pfcoiltrack is None:
            pfcoiltrack = self.tfcoil

        if figure_folder is None:
            figure_folder = try_get_bluemira_path(
                "plots/equilibria", subfolder="data", allow_missing=not gif
            )

        self.p_optimiser = PositionOptimiser(
            max_PF_current,
            self.coilset.get_max_fields(),
            PF_Fz_max,
            CS_Fz_sum,
            CS_Fz_sep,
            psi_values=psis,
            pfcoiltrack=pfcoiltrack,
            CS_x=CS_x,
            CS_zmin=CS_zmin,
            CS_zmax=CS_zmax,
            CS_gap=CS_gap,
            pf_coilregions=pf_coilregions,
            pf_exclusions=pf_exclusions,
            CS=CS,
            plot=plot,
            gif=gif,
        )

        self.p_optimiser(self.eq, self.constraints)

        if gif:
            make_gif(figure_folder, "pos_opt")

        self._consolidate_coilset(
            self.p_optimiser.eq.copy(), self.p_optimiser.swing, plot=plot
        )

        if "Breakdown" in self.snapshots:
            self._consolidate_breakdown()

    def _consolidate_coilset(self, eqbase, swing, plot=False):
        """
        Finalises the problem
            * Sizes PF coils correctly (max size over swing)
            * Converges the equilibria and stores equilibrium objects

        Parameters
        ----------
        eqbase: Equilibrium object
            Base equilibrium (with fixed current source)
        swing: dict
            Dictionary of optimal coil current vectors for each snapshot
        """
        # TODO: Check update of PF coil current limits is being correctly handled!
        currents = np.array(list(swing.values()))
        max_currents = np.max(np.abs(currents), axis=0)
        self.coilset.adjust_sizes(max_currents)
        self.coilset.fix_sizes()
        self.coilset.mesh_coils(0.3)
        max_currents = self.coilset.get_max_currents(0)  # Sizes should all be fixed
        eqbase.coilset = self.coilset
        eqbase._remap_greens()
        # Make new equilibria objects for snapshots
        optimiser = self.p_optimiser.current_optimiser.copy()
        # relaxation
        max_currents = np.append(
            1.0 * max_currents[: self.coilset.n_PF], max_currents[self.coilset.n_PF :]
        )
        optimiser.update_current_constraint(max_currents)

        for i, (k, v) in enumerate(swing.items()):
            converger = EquilibriumConverger(
                eqbase, self.profiles, self.constraints, optimiser, plot=plot
            )
            eq, profiles = converger(k)
            if i == 0:
                name = "SOF"
            elif i == self.n_swing - 1:
                name = "EOF"
            else:
                name = f"MOF_{i}"

            self.take_snapshot(
                name,
                eq,
                profiles=profiles,
                coilset=eq.coilset.copy(),
                optimiser=optimiser,
            )

    def _consolidate_breakdown(self):
        """
        Recalculates the breakdown phase with optimal coil positions and fixed
        PF coil positions
        """
        bd = self.snapshots["Breakdown"].eq
        bd.coilset = self.coilset.copy()
        optimiser = self.snapshots["Breakdown"].optimiser

        max_currents = self.coilset.get_max_currents(0)  # Sizes should all be fixed
        optimiser.update_current_constraint(max_currents)

        currents = optimiser(bd)
        bd.coilset.set_control_currents(currents)
        incr = 5  # m
        z = incr * (1.5 * self.kappa * self.R_0 / self.A // incr + 1)
        grid = Grid(0, incr * (self.R_0 * 2.2 // incr + 1), -z, z, 100, 100)
        bd.reset_grid(grid)

        psi_bd_new = bd.breakdown_psi * 2 * np.pi

        if abs_rel_difference(psi_bd_new, self.psi_bd) > 0.01:
            bluemira_warn(
                "It appears there is a problem with pre-magnetisation. The new breakdown flux is "
                "quite different from the old one:\n"
                f"\t{psi_bd_new:.2f} !~= {self.psi_bd:.2f} [V.s]"
            )

        self.take_snapshot(
            "Breakdown",
            bd,
            limiter=None,
            constraints=None,
            tfcoil=None,
            optimiser=optimiser,
            coilset=bd.coilset,
        )
        self.coilset.reset()
        self.coilset.mesh_coils(0.4)

    def report(self):
        """
        Prints a summary of the plasma shape and properties for the
        solution of the EquilibriumProblem
        """
        table = {
            "Target": {
                "R_0": self.R_0,
                "A": self.A,
                "kappa": self.kappa,
                "delta": self.delta,
                "I_p": self.Ip,
                "l_i": self.li,
                "beta_p": self.betap,
                "q_95_min": self.q95min,
            }
        }
        if len(self.snapshots) != 0:
            for name, snap in self.snapshots.items():
                if name == "Breakdown":
                    continue
                res = snap.eq.analyse_plasma()
                table[name] = {
                    "R_0": res["R_0"],
                    "A": res["A"],
                    "kappa": res["kappa"],
                    "delta": res["delta"],
                    "I_p": res["Ip"],
                    "l_i": res["li(3)"],
                    "beta_p": res["beta_p"],
                    "q_95_min": res["q_95"],
                }
        else:
            res = self.eq.analyse_plasma()
            table["ref"] = {
                "R_0": res["R_0"],
                "A": res["A"],
                "kappa": res["kappa"],
                "delta": res["delta"],
                "I_p": res["Ip"],
                "l_i": res["li(3)"],
                "beta_p": res["beta_p"],
                "q_95_min": res["q_95"],
            }
        df = DataFrame(list(table.values()), index=list(table.keys())).transpose()
        df = df.applymap(lambda x: f"{x:.2f}")
        print(tabulate.tabulate(df, headers=df.columns))
        return df

    def opt_report(self):
        """
        Prints a summary of the optimisers for the EquilibriumProblem
        """
        dd = {}
        for k, snap in self.snapshots.items():
            dd[k] = snap.optimiser.report(verbose=False)
        # BUILD MEGATABLE
        cols = [dd["SOF"]["Coil / Constraint"], dd["Breakdown"]["I_max (abs) [MA]"]]
        cols.extend([dd[k]["I [MA]"] for k in dd.keys()])
        cols.append(dd["Breakdown"]["B_max [T]"])
        cols.extend([dd[k]["B [T]"] for k in dd.keys()])
        cols.append(dd["Breakdown"]["F_z_max [MN]"])
        cols.extend([dd[k]["F_z [MN]"] for k in dd.keys()])
        ddd = concat(cols, axis=1)
        ddd = ddd[ddd["Coil / Constraint"] != "F_sep_max"]
        ddd = ddd[ddd["Coil / Constraint"] != "F_z_CS_tot"]
        print(tabulate.tabulate(ddd, headers=ddd.columns, showindex=False))
        return ddd

    def copy(self):
        """
        Deepcopies an EquilibriumProblem object, returning a fully independent
        copy, with independent values.
        """
        return deepcopy(self)


class AbInitioEquilibriumProblem(EquilibriumProblem):
    """
    Class for defining and solving ab initio equilibria and generating
    reference geometries and .eqdsk files

    Parameters
    ----------
    R_0: float
        Major radius [m]
    B_0: float
        Toroidal field at major radius [T]
    A: float
        Plasma aspect ratio
    Ip: float
        Plasma current [A]
    betap: float
        Ratio of plasma pressure to poloidal magnetic pressure
    li: float
        Normalised plasma internal inductance
    kappa: float
        Plasma elongation
    delta: float
        Plasma triangularity
    r_cs: float
        Central Solenoid mean radius [m]
    tk_cs: float
        Central Solenoid thickness either side [m]
    tfbnd: Loop object
        TF coil boundary (PF coil positioning "track")
    n_PF: int
        Number of PF coils
    n_CS: int
        Number of CS modules
    q95min: float
        Minimum q_95 value. NOTE: not optimised, just checked
    eqtype: str from ['SN', 'DN'] (optional) default = 'SN'
        Type of equilibrium shape
    rtype: str (optional) default = 'SN'
        Type of reactor
    profile: None or 1-D np.array
        Shape profile to use for equilibrium. Defaults to a PowerLaw (2, 4)
    coilset: None or CoilSet object
        Coilset from which to set up the equilibrium problem. If not specified,
        other input parameters will be used to generate a default coilset.
    """

    def __init__(
        self,
        R_0,
        B_0,
        A,
        Ip,
        betap,
        li,
        kappa_u,
        kappa_l,
        delta_u,
        delta_l,
        psi_u_neg,
        psi_u_pos,
        psi_l_neg,
        psi_l_pos,
        div_l_ib,
        div_l_ob,
        r_cs,
        tk_cs,
        tfbnd,
        n_PF,
        n_CS,
        q95min=3,
        c_ejima=0.4,
        eqtype="SN",
        rtype="Normal",
        profile=None,
        psi=None,
        coilset=None,
    ):
        super().__init__()
        # Make FD grid for G-S solver
        sx, sz = 1.6, 1.7  # grid scales from plasma
        self.nx, self.nz = 65, 65
        kappa = max(kappa_l, kappa_u)
        delta = max(delta_l, delta_u)
        x_min, x_max = R_0 - sx * (R_0 / A), R_0 + sx * (R_0 / A)
        z_min, z_max = -sz * (kappa * R_0 / A), sz * (kappa * R_0 / A)

        if rtype == "ST":
            x_min = 0.01
            x_max += 3.0
            z_min -= 0
            z_max += 0

        grid = Grid(x_min, x_max, z_min, z_max, self.nx, self.nz)

        # Set up plasma position constraints
        if rtype == "Normal":
            if eqtype == "SN":
                self.constraints = EUDEMOSingleNullConstraints(
                    R_0,
                    0,
                    A,
                    kappa_u,
                    kappa_l,
                    delta_u,
                    delta_l,
                    psi_u_neg,
                    psi_u_pos,
                    psi_l_neg,
                    psi_l_pos,
                    div_l_ib,
                    div_l_ob,
                    0,
                )
            elif eqtype == "DN":
                self.constraints = EUDEMODoubleNullConstraints(
                    R_0,
                    0,
                    A,
                    kappa_u,
                    delta_u,
                    psi_u_neg,
                    psi_u_pos,
                    div_l_ib,
                    div_l_ob,
                    0,
                )
        elif rtype == "ST":
            raise ValueError(
                "Spherical reactors not yet supported through this interface."
            )
        else:
            raise ValueError("Alternatives not yet supported through this interface.")

        # Specify poloidal field system coils
        self.tfcoil = tfbnd
        if coilset is None:  # CoilPositioner determines locations
            if self.tfcoil is None:
                raise ValueError(
                    "You need to specify a boundary along which to position the"
                    "PF coils."
                )

            positioner = CoilPositioner(
                R_0, A, delta, kappa, tfbnd, r_cs, tk_cs, n_PF, n_CS, rtype=rtype
            )
            self.positioner = positioner
            self.coilset = positioner.make_coilset()
        else:  # Input coilset object
            self.coilset = coilset.copy()
        self.coilset.assign_coil_materials("PF", j_max=NBTI_J_MAX, b_max=NBTI_B_MAX)
        self.coilset.assign_coil_materials("CS", j_max=NB3SN_J_MAX, b_max=NB3SN_B_MAX)
        # Limiter for mathematical stability (occasionally)
        self.lim = Limiter(x=[x_min + 0.3, x_max - 0.3], z=[0, 0])
        # Equilibrium object
        self.eq = Equilibrium(
            self.coilset,
            grid,
            vcontrol=None,  # 'virtual',
            limiter=self.lim,
            Ip=Ip,
            psi=psi,
            li=li,
        )
        # Specify plasma profiles and integral plasma constraints
        if isinstance(profile, np.ndarray):
            profile = LaoPolynomialFunc.from_datafit(profile, order=3)
        elif profile is None:
            if rtype == "Normal":
                profile = DoublePowerFunc([2, 2])
            elif rtype == "ST":
                profile = DoublePowerFunc([2, 1])
        elif isinstance(profile, ShapeFunction):
            profile = profile
        else:
            raise ValueError(
                f"Could not make a ShapeFunction from profile of type {type(profile)}"
            )
        self.profiles = BetaIpProfile(betap, Ip, R_0, B_0, shape=profile)

        # Default optimiser
        # Performs an initial Tikhonov unconstrained optimisation routine, to
        # reach a reference equilibrium state, ignoring flux swing
        self.optimiser = Norm2Tikhonov(gamma=1e-7)
        # Store target values
        self.R_0 = R_0
        self.kappa = kappa
        self.delta = delta
        self.A = A
        self.betap = betap
        self.li = li
        self.Ip = Ip
        self.B_0 = B_0
        self.q95min = q95min
        self.c_ejima = c_ejima
        self.eqtype = eqtype
        self.snapshots = {}


class AbExtraEquilibriumProblem(EquilibriumProblem):
    """
    Class for defining and re-solving ab extra equilibria

    Parameters
    ----------
    filename: str
        Filename of an .eqdsk file
    """

    def __init__(self, filename, load_large_file=False):
        super().__init__()
        self.eq = Equilibrium.from_eqdsk(filename, load_large_file=load_large_file)
        self.coilset = self.eq.coilset
        self.lim = None  # self.eq.limiter  # bloody CREATE
        self.tfcoil = None
        self.profiles = CustomProfile.from_eqdsk(filename)
        self.eq._reassign_profiles(self.profiles)
        self.Ip = self.profiles.Ip
        self.B_0 = self.profiles._B_0
        self.betap = calc_betap(self.eq)
        self.li = calc_li(self.eq)
        self.optimiser = Norm2Tikhonov(gamma=1e-12)
        self.constraints = AutoConstraints(self.eq)
        self.set_shape_characteristics()
        self.snapshots = {}

    def set_shape_characteristics(self):
        """
        Analyse the separatrix to get shape characteristics.
        """
        d = self.eq.analyse_separatrix()
        kappa = "kappa_u"  # TODO: Handle eq type reocvery
        delta = "delta_l"
        self.eqtype = "SN"
        d["kappa"] = d[kappa]
        d["delta"] = d[delta]
        for a in ["A", "R_0", "kappa", "delta"]:
            setattr(self, a, d[a])

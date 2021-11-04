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
Constrained and unconstrained optimisation tools for coilset design
"""
import numpy as np
import nlopt
from typing import Type
from copy import deepcopy
import matplotlib.pyplot as plt
from pandas import DataFrame
import tabulate

from bluemira.base.look_and_feel import bluemira_warn, bluemira_print_flush
from bluemira.base.file import try_get_bluemira_path
from bluemira.equilibria.error import EquilibriaError
from bluemira.geometry._deprecated_tools import make_circle_arc
from bluemira.utilities.plot_tools import save_figure
from bluemira.utilities.opt_tools import (
    least_squares,
    tikhonov,
)
from bluemira.utilities.optimiser import approx_derivative
from bluemira.utilities._nlopt_api import process_NLOPT_result
from bluemira.equilibria.positioner import XZLMapper, RegionMapper
from bluemira.equilibria.coils import CS_COIL_NAME
from bluemira.equilibria.constants import DPI_GIF, PLT_PAUSE
from bluemira.equilibria.equilibrium import Equilibrium

__all__ = [
    "FBIOptimiser",
    "BoundedCurrentOptimiser",
    "BreakdownOptimiser",
    "PositionOptimiser",
    "Norm2Tikhonov",
]


class EquilibriumOptimiser:
    """
    Base class for optimisers used on equilibria, which provides a __call__
    method, handling Equilibrium and Constraints objects

    Child classes must provide the following methods:
        .optimise(self) ==> return: x*

    Child classses must provide the following attributes:
        .rms_error
    """

    def __call__(self, eq, constraints, psi_bndry=None):
        """
        Parameters
        ----------
        eq: Equilibrium object
            The Equilibrium to be optimised
        constraints: Constraints object
            The Constraints to apply to the equilibrium. NOTE: these only
            include linearised constraints. Quadratic and/or non-linear
            constraints must be provided in the sub-classes

        Attributes
        ----------
        A: np.array(N, M)
            Response matrix
        b: np.array(N)
            Constraint vector

        \t:math:`\\mathbf{A}\\mathbf{x}-\\mathbf{b}=\\mathbf{b_{plasma}}`

        Notes
        -----
        The weight vector is used to scale the response matrix and
        constraint vector. The weights are assumed to be uncorrelated, such that the
        weight matrix W_ij used to define (for example) the least-squares objective
        function (Ax - b)ᵀ W (Ax - b), is diagonal, such that
        weights[i] = w[i] = sqrt(W[i,i]).
        """
        self.eq = eq
        self.constraints = constraints
        self.n = len(eq.coilset._ccoils)

        if psi_bndry is not None:
            constraints.update_psi_boundary(psi_bndry)

        self.A = constraints.A
        self.b = constraints.b
        self.w = constraints.w

        # Scale the control matrix and constraint vector by weights
        self.A = self.w[:, np.newaxis] * self.A
        self.b *= self.w

        self.n_PF, self.n_CS = eq.coilset.n_PF, eq.coilset.n_CS
        self.n_C = eq.coilset.n_coils
        return self.optimise()

    def copy(self):
        """
        Get a deep copy of the EquilibriumOptimiser.
        """
        return deepcopy(self)


class Norm2Tikhonov(EquilibriumOptimiser):
    """
    Unconstrained norm-2 optimisation with Tikhonov regularisation

    Returns x*
    """

    def __init__(self, gamma=1e-12):
        self.gamma = gamma

    def optimise(self):
        """
        Optimise the prescribed problem.
        """
        self.x = tikhonov(self.A, self.b, self.gamma)
        self.calc_error()
        return self.x

    def calc_error(self):
        """
        Calculate the RSS and RMS errors.
        """
        x = self.x.reshape(-1, 1)
        b = self.b.reshape(len(self.b), 1)
        err = np.dot(self.A, x) - b
        self.rms_error = np.sqrt(np.mean(err ** 2 + (self.gamma * self.x) ** 2))
        self.rss_error = np.sum(err ** 2) + np.sum((self.gamma * self.x) ** 2)


class LeastSquares(EquilibriumOptimiser):
    """
    Unconstrained least squares optimisation

    Returns x*
    """

    def __init__(self):
        pass

    def optimise(self):
        """
        Optimise the prescribed problem.
        """
        self.x = least_squares(self.A, self.b)
        self.calc_error()
        return self.x

    def calc_error(self):
        """
        Calculate the RSS and RMS errors.
        """
        x = self.x.reshape(-1, 1)
        b = self.b.reshape(len(self.b), 1)
        err = np.dot(self.A, x) - b
        self.rms_error = np.sqrt(np.mean(err ** 2))
        self.rss_error = np.sum(err ** 2)


class PositionOptimiser:
    """
    Coil position optimiser a la McIntosh

    Parameters
    ----------
    max_PF_current: float
        Maximum PF coil current [A]
    max_fields: np.array(n_C)
        Maximum fields at coils [T]
    PF_Fz_max: float
        The maximum absolute vertical on a PF coil [N]
    CS_Fz_sum: float
        The maximum absolute vertical on all CS coils [N]
    CS_Fz_sep: float
        The maximum Central Solenoid vertical separation force [N]
    psi_values: list(float, float) (will default to 0 flux)
        Plasma boundary fluxes to optimiser over (equilibrium snapshots)
        [V.s/rad]
    pfcoiltrack: Geometry::Loop
        The track along which the PF coil positions are optimised
    CS_x: float
        Solenoid geometric centre axis location (radius from machine axis)
    CS_zmin: float
        Minimum CS z position (lowest edge)
    CS_zmax: float
        Maximum CS z position (highest edge)
    CS_gap: float
        Gap between CS modules
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
        The path where figures will be saved. If the input value is None (e.g. default)
        then this will be reinterpreted as the path data/plots/equilibria under the
        bluemira root folder, if that path is available.

    """

    def __init__(
        self,
        max_PF_current,
        max_fields,
        PF_Fz_max,
        CS_Fz_sum,
        CS_Fz_sep,
        psi_values=None,
        pfcoiltrack=None,
        CS_x=None,
        CS_zmin=None,
        CS_zmax=None,
        CS_gap=None,
        pf_exclusions=None,
        pf_coilregions=None,
        CS=False,
        plot=False,
        gif=False,
        figure_folder=None,
    ):

        self._sanity_checks(pf_coilregions, pfcoiltrack)

        self.IPF_max = max_PF_current
        self.current_optimiser = FBIOptimiser(
            max_fields, PF_Fz_max, CS_Fz_sum, CS_Fz_sep
        )
        if pfcoiltrack is not None:
            self.flag_PFT = True
            self.XLmap = XZLMapper(
                pfcoiltrack, CS_x, CS_zmin, CS_zmax, cs_gap=CS_gap, CS=CS
            )
            if pf_exclusions is not None:
                self.XLmap.add_exclusion_zones(pf_exclusions)
        else:
            self.flag_PFT = False

        if pf_coilregions is not None:
            self.Rmap = RegionMapper(pf_coilregions)
            self.flag_PFR = True
            self.region_coils = set(pf_coilregions.keys())
        else:
            self.flag_PFR = False
            self.region_coils = set()

        self.psi_vals = psi_values if psi_values is not None else [0]
        self.flag_CS = CS
        self.iter = 0
        self.swing = {}  # Flux swing dict for currents at snapshots

        # Plot utilities
        self.flag_plot = plot
        self.flag_gif = gif
        if figure_folder is None:
            figure_folder = try_get_bluemira_path(
                "plots/equilibria", subfolder="data", allow_missing=not self.flag_gif
            )
        self.figure_folder = figure_folder
        self.plot_iter = 0

    @staticmethod
    def _sanity_checks(pf_coilregions, pfcoiltrack):
        """
        Check required inputs exists.
        """
        if pfcoiltrack is None and pf_coilregions is None:
            raise EquilibriaError("No coil track or coil region specified")

    def __call__(self, eq, constraints):
        """
        Optimise the coil positions in an equilibrium, subject to constraints.

        Parameters
        ----------
        eq: Equilibrium
            The equilibrium to optimise the positions for
        constraints: Constraints
            The plasma shaping constraints to apply

        Returns
        -------
        opt_currents: np.array(n_coils)
            The optimal currents for the controlled coils.
        """
        self.eq = eq
        self.constraints = constraints
        self.n_PF = self.eq.coilset.n_PF

        if self.flag_PFR:
            self.n_PFR = self.Rmap.no_regions
            if self.n_PFR <= self.n_PF:
                self.n_PF -= self.n_PFR
            else:
                raise ValueError("More regions than PF coils")
        else:
            self.n_PFR = 0

        self.track_pf_coils = sorted(
            {*self.eq.coilset.get_PF_names()} - self.region_coils
        )
        # DO NOT DELETE THIS COMMENT
        # this line is very important sets are unordered therefore
        # need to be converted to a list
        # WARNING if CS coils are ever here this may be a source of bugs
        self.region_coils = sorted(self.region_coils)

        if self.flag_CS:
            self.n_CS = self.eq.coilset.n_CS
            self.n_L = self.n_PF + self.n_CS + 2 * self.n_PFR  # CS edges not centres
        else:
            self.n_CS = -1
            self.n_L = self.n_PF + 2 * self.n_PFR

        self.mx_c_ind = self.n_L - 2 * self.n_PFR

        return self.optimise()

    def optimise(self):
        """
        The main optimiser object function. Returned upon __call__
        Returns
        -------
        I_star: np.array(n_C)
            The array of optimal coil currents
        """
        l_0, lb, ub = np.empty((3, self.n_L))

        if self.flag_PFT:
            (
                l_0[: self.mx_c_ind],
                lb[: self.mx_c_ind],
                ub[: self.mx_c_ind],
            ) = self.XLmap.get_Lmap(self.eq.coilset, self.track_pf_coils)
            l_0[: self.n_PF] = l_0[: self.n_PF][::-1]
            lb[: self.n_PF] = lb[: self.n_PF][::-1]
            ub[: self.n_PF] = ub[: self.n_PF][::-1]
            # L0 = self.normalise_L(L0, lb, ub)
            # self.lb, self.ub = lb, ub

        if self.flag_PFR:
            (
                l_0[self.mx_c_ind :],
                lb[self.mx_c_ind :],
                ub[self.mx_c_ind :],
            ) = self.Rmap.get_Lmap(self.eq.coilset)

        opt = nlopt.opt(nlopt.LN_COBYLA, self.n_L)
        # TODO: Improve termination conditions...
        # Stop conditions difficult to generalise with mixed objective func
        # opt.set_ftol_abs(1)  # Doesn't seem to ever trigger? Not great
        # opt.set_ftol_rel(0.1)  # Doesn't seem to ever trigger? Bad for global
        # opt.set_stopval(1279)  # Very problem specific...
        opt.set_maxeval(20)  # Pretty generic
        # opt.set_maxtime(200)  # Porque no.. :)
        opt.set_min_objective(self.f_min_rms)
        opt.set_lower_bounds([0 for _ in range(self.n_L)])
        opt.set_upper_bounds([1 for _ in range(self.n_L)])
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)

        self.bounds = np.array(
            [np.zeros(self.n_L, dtype=np.int), np.ones(self.n_L, dtype=np.int)]
        )

        if self.flag_CS:
            tol = 1e-3 * np.ones(self.n_CS - 1)
            opt.add_inequality_mconstraint(self.constrain_L_CS, tol)
        try:
            positions = opt.optimize(l_0)
        except nlopt.RoundoffLimited:  # Dodgy SLSQP. Normalerweise gut genug
            positions = self._store
            bluemira_warn("NLopt RoundoffLimited!")

        self.current_optimiser.sanity()
        self.rms = opt.last_optimum_value()
        self.update_positions(positions)
        process_NLOPT_result(opt)
        return self.I_star  # Need current vector for min_solve

    def f_min_rms(self, pos_vector, grad):
        """
        Minimisation objective function for NLOPT optimiser object

        Parameters
        ----------
        pos_vector: np.array(n_L)
            The initial positions of the coils
        grad: np.array
            Gradient used by NLOPT algorithms. Approximated for L opt
        """
        self.iter += 1
        self.rms_error = self.update_positions(pos_vector)
        if grad.size > 0:
            grad[:] = approx_derivative(
                self.update_positions,
                pos_vector,
                bounds=self.bounds,
                f0=self.rms_error,
                rel_step=1e-3,
            )
        bluemira_print_flush(
            f"EQUILIBRIA position optimisation iteration {self.iter}: "
            f"f_obj = {self.rms_error:.2f}"
        )
        return self.rms_error

    def _get_PF_pos(self, pos_vector):
        for position, name in zip(pos_vector[: self.n_PF][::-1], self.track_pf_coils):
            self._positions[name] = [*self.XLmap.L_to_xz(position)]

    def _get_CS_pos(self, pos_vector):
        cs_pos = pos_vector[self.n_PF : self.n_CS + self.n_PF]
        xcs, zcs, dzz = self.XLmap.L_to_zdz(cs_pos)
        for no, (x, z, dz) in enumerate(zip(xcs, zcs, dzz), start=1):
            name = CS_COIL_NAME.format(abs(no))
            self._positions[name] = [x, z, dz]

    def _get_PFR_pos(self, pos_vector):
        pv = pos_vector[self.mx_c_ind :].reshape(-1, 2)
        for name, position in zip(self.region_coils, pv):
            self._positions[name] = [*self.Rmap.L_to_xz(name, position)]

    def _get_i_max(self):
        i_max = self.eq.coilset.get_max_currents(self.IPF_max)

        if self.flag_PFR:
            # set max currents based on coil positions in region
            PFR = self.mx_c_ind + self.n_PFR
            i_max = i_max[
                np.r_[
                    slice(0, self.mx_c_ind),
                    slice(PFR, i_max.size),
                    slice(self.mx_c_ind, PFR),
                ]
            ]
            i_max[i_max.size - self.n_PFR :] = self.Rmap.get_size_current_limit()
        return i_max

    def _get_current_rms_error(self):
        error = []
        for psi in self.psi_vals:
            i_star = self.current_optimiser(
                self.eq,
                self.constraints,
                psi_bndry=psi,
            )
            self.swing[psi] = i_star
            self.I_star = i_star
            self.eq.coilset.set_control_currents(i_star)

            if self.flag_plot:
                ax = plt.gca()
                ax.clear()
                self.eq.plot(ax=ax, update_ox=True)
                self.eq.coilset.plot(ax=ax)
                ax.figure.canvas.draw()
                if self.flag_gif:
                    self.plot_iter += 1
                    save_figure(
                        plt.gcf(),
                        "pos_opt_snapshot_" + str(self.plot_iter),
                        folder=self.figure_folder,
                        save=True,
                        dpi=DPI_GIF,
                    )
                plt.pause(PLT_PAUSE)

            error.append(self.current_optimiser.rms_error)

        return max(error)

    def _optimise_currents(self):
        self.constraints(self.eq, I_not_dI=True)

        i_max = self._get_i_max()

        try:  # For shifting CS coils.. need to update I constraint vector
            self.current_optimiser.update_current_constraint(i_max)
        except AttributeError:
            bluemira_warn("No update_current_constraint implemented in sub-optimiser")
            pass

        return self._get_current_rms_error()

    def update_positions(self, pos_vector):
        """
        Updates coilset with new PF and CS coil positions.

        Parameters
        ----------
        pos_vector: np.array(n_L)
            The normalised coil positions along the PF and CS parameterised
            tracks. Handled by XLmap to pass new positions onto CoilSet

        """
        self._store = pos_vector
        self._positions = {}

        if self.flag_PFT:
            self._get_PF_pos(pos_vector)

        if self.flag_CS:
            self._get_CS_pos(pos_vector)

        if self.flag_PFR:
            self._get_PFR_pos(pos_vector)

        self.eq.coilset.set_positions(self._positions)
        self.eq.coilset.mesh_coils()

        if self.flag_plot:
            self.eq._remap_greens()
        else:
            self.eq.set_forcefield()

        return self._optimise_currents()

    def constrain_positions(self, constraint, pos_vector, grad):
        """
        Positional constraints on coils
        """
        # Semi-deprecated, but nice formulation of jacobian for positional
        # constraints
        pf_dl = 1e-4  # minimum PF inter-coil spacing
        cs_dl = 1e-4  # minimum CS coil height
        if grad.size > 0:
            if self.flag_CS:
                grad[:] = np.zeros((self.n_PF - 1, len(pos_vector)))
                for i in range(self.n_PF - 1):  # PF
                    grad[i, i] = 1
                    grad[i, i + 1] = -1
                for i in range(self.n_CS):  # CS   # -1
                    grad[i + self.n_PF - 1, i + self.n_PF] = 1
                    grad[i + self.n_PF - 1, i + self.n_PF + 1] = -1
            else:
                grad[:] = np.zeros((self.n_PF - 1, len(pos_vector)))
                for i in range(self.n_PF - 1):  # PF
                    grad[i, i] = 1
                    grad[i, i + 1] = -1
        constraint[: self.n_PF - 1] = (
            pos_vector[: self.n_PF - 1] - pos_vector[1 : self.n_PF] + pf_dl
        )  # PF
        if self.flag_CS:  # CS
            constraint[self.n_PF - 1 :] = (
                pos_vector[self.n_PF : -1] - pos_vector[self.n_PF + 1 :] + cs_dl
            )

    def constrain_L_CS(self, constraint, pos_vector, grad):
        """
        Positional constraints on coils
        """
        # Semi-deprecated, but nice formulation of jacobian for positional
        # constraints
        cs_dl = 1e-4  # minimum CS coil height
        if grad.size > 0:
            grad[:] = np.zeros((self.n_CS, len(pos_vector)))  # initalise
            for i in range(self.n_CS):  # CS
                grad[i, i] = 1
                grad[i, i + 1] = -1
        constraint[:] = (
            pos_vector[self.n_PF : -1] - pos_vector[self.n_PF + 1 :] + cs_dl
        )  # CS


class SanityReporter:
    """
    FBI constraint optimiser reporting mixin class. Provides optimiser
    sanity checking and reporting functionality.
    """

    eq: Type[Equilibrium]
    n_PF: int
    n_CS: int
    scale: float
    PF_Fz_max: float
    CS_Fz_sep: float
    CS_Fz_sum: float
    constraint_tol: float
    I_max: Type[np.array]
    B_max: Type[np.array]
    _I_star: Type[np.array]

    def _get_force_field(self):
        F = self.eq.force_field.calc_force(self._I_star)[0] / 1e6  # MN
        B = self.eq.force_field.calc_field(self._I_star)[0]
        f_pf = F[: self.n_PF, 1]
        fz_cs = F[self.n_PF :, 1]
        f_cs_tot = np.sum(fz_cs)
        f_cs_sep = [
            np.sum(fz_cs[: i + 1]) - np.sum(fz_cs[i + 1 :]) for i in range(self.n_CS - 1)
        ]
        return F, B, f_pf, fz_cs, f_cs_tot, f_cs_sep

    def sanity(self):
        """
        Man kann nie zu vorsichtig sein
        """
        F, B, f_pf, fz_cs, f_cs_tot, f_cs_sep = self._get_force_field()

        pf_max = self.PF_Fz_max * np.ones(self.n_PF)
        cs_sep = self.CS_Fz_sep * np.ones(self.n_CS - 1)
        cons = [self.I_max, self.B_max, pf_max, [self.CS_Fz_sum], cs_sep]
        names = [
            "Max current",
            "Max field",
            "Max PF vertical force",
            "Max CS stack vertical force",
            "Max CS separation force",
        ]
        vals = [np.abs(self._I_star / self.scale), B, np.abs(f_pf), [f_cs_tot], f_cs_sep]
        for con, val, name in zip(cons, vals, names):
            for c, v in zip(con, val):
                if v > (1 + self.constraint_tol) * c:
                    bluemira_warn(f"FBI {name} constraint violated: |{v:.2f}| > {c:.2f}")

    def report(self, verbose=True):
        """
        Generates a report on the optimiser's solution including constraint
        information
        """
        table = {}
        names = self.eq.coilset.get_control_names()
        F, B, f_pf, fz_cs, f_cs_tot, f_cs_sep = self._get_force_field()
        f_sep_max = max(f_cs_sep)

        table["Coil / Constraint"] = names
        table["I [MA]"] = self._I_star / self.scale
        table["I_max (abs) [MA]"] = self.I_max
        table["B [T]"] = B
        table["B_max [T]"] = self.eq.coilset.get_max_fields()
        table["F_z [MN]"] = F[:, 1]
        table["F_z_max [MN]"] = [self.PF_Fz_max] * self.n_PF + ["N/A"] * self.n_CS
        keys = list(table.keys())
        df = DataFrame(list(table.values()), index=keys).transpose()
        row = {k: "" for k in keys}
        fseprow = row.copy()
        fseprow["Coil / Constraint"] = "F_sep_max"
        fseprow["F_z [MN]"] = f_sep_max
        fseprow["F_z_max [MN]"] = self.CS_Fz_sep
        c_stotrow = row.copy()
        c_stotrow["Coil / Constraint"] = "F_z_CS_tot"
        c_stotrow["F_z [MN]"] = f_cs_tot
        c_stotrow["F_z_max [MN]"] = self.CS_Fz_sum
        df = df.append(fseprow, ignore_index=True)
        df = df.append(c_stotrow, ignore_index=True)
        df = df.applymap(lambda x: x if type(x) is str else f"{x:.2f}")
        if verbose:
            print(tabulate.tabulate(df, headers=df.columns))
        return df


class ForceFieldConstrainer:
    """
    Mixin utility class for calculation force and field constraint matrices.
    """

    eq: object
    scale: float
    n_PF: int
    n_CS: int
    PF_Fz_max: float
    CS_Fz_sum: float
    CS_Fz_sep: float
    B_max: float

    def constrain_forces(self, constraint, vector, grad):
        """
        Current optimisation force constraints
        vector: I

        Note
        ----
        Modifies (in place):
            constraint
            grad
        """
        # get coil force and jacobian
        F, dF = self.eq.force_field.calc_force(vector * self.scale)  # noqa (N803)
        F /= self.scale  # Scale down to MN
        # dF /= self.scale

        # calculate constraint jacobian
        if grad.size > 0:
            # PFz lower bound
            grad[: self.n_PF] = -dF[: self.n_PF, :, 1]
            # PFz upper bound
            grad[self.n_PF : 2 * self.n_PF] = dF[: self.n_PF, :, 1]

            if self.n_CS != 0:
                # CS sum lower
                grad[2 * self.n_PF] = -np.sum(dF[self.n_PF :, :, 1], axis=0)
                # CS sum upper
                grad[2 * self.n_PF + 1] = np.sum(dF[self.n_PF :, :, 1], axis=0)
                for j in range(self.n_CS - 1):  # evaluate each gap in CS stack
                    # CS separation constraint Jacobians
                    f_up = np.sum(dF[self.n_PF : self.n_PF + j + 1, :, 1], axis=0)
                    f_down = np.sum(dF[self.n_PF + j + 1 :, :, 1], axis=0)
                    grad[2 * self.n_PF + 2 + j] = f_up - f_down

        # vertical force on PF coils
        pf_fz = F[: self.n_PF, 1]
        # PFz lower bound
        constraint[: self.n_PF] = -self.PF_Fz_max - pf_fz
        # PFz upper bound
        constraint[self.n_PF : 2 * self.n_PF] = pf_fz - self.PF_Fz_max

        if self.n_CS != 0:
            # vertical force on CS coils
            cs_fz = F[self.n_PF :, 1]
            # vertical force on CS stack
            cs_z_sum = np.sum(cs_fz)
            # CSsum lower bound
            constraint[2 * self.n_PF] = -self.CS_Fz_sum - cs_z_sum
            # CSsum upper bound
            constraint[2 * self.n_PF + 1] = cs_z_sum - self.CS_Fz_sum
            for j in range(self.n_CS - 1):  # evaluate each gap in CS stack
                # CS seperation constraints
                f_sep = np.sum(cs_fz[: j + 1]) - np.sum(cs_fz[j + 1 :])
                constraint[2 * self.n_PF + 2 + j] = f_sep - self.CS_Fz_sep
        return constraint

    def constrain_fields(self, constraint, vector, grad):
        """
        Current optimisation field constraints
        vector: I

        Note
        ----
        Modifies (in-place):
            constraint
            grad
        """
        B, dB = self.eq.force_field.calc_field(vector * self.scale)  # noqa (N803)
        dB /= self.scale ** 2
        if grad.size > 0:
            grad[:] = dB
        constraint[:] = B - self.B_max
        return constraint


class FBIOptimiser(SanityReporter, ForceFieldConstrainer, EquilibriumOptimiser):
    """
    Force Field and Current constrained McIntoshian optimiser class.
    Freeze punk!

    Parameters
    ----------
    max_fields: np.array(n_coils)
        The array of maximum poloidal field [T]
    PF_Fz_max: float
        The maximum absolute vertical on a PF coil [N]
    CS_Fz_sum: float
        The maximum absolute vertical on all CS coils [N]
    CS_Fz_sep: float
        The maximum Central Solenoid vertical separation force [N]
    """

    def __init__(
        self, max_fields, PF_Fz_max, CS_Fz_sum, CS_Fz_sep, **kwargs
    ):  # noqa (N803)
        # Used scale for optimiser RoundoffLimited Error prevention
        self.scale = 1e6  # Scale for currents and forces (MA and MN)
        self.gamma = kwargs.get("gamma", 1e-14)  # 1e-7  # 0
        self.constraint_tol = kwargs.get("constraint_tol", 1e-3)
        # self.gamma /= self.scale
        self.B_max = max_fields
        self.PF_Fz_max = PF_Fz_max / self.scale
        self.CS_Fz_sum = CS_Fz_sum / self.scale
        self.CS_Fz_sep = CS_Fz_sep / self.scale
        self.rms = None
        self.rms_error = None
        self.I_max = None

    def update_current_constraint(self, max_current):
        """
        Updates the current vector bounds. Must be called prior to optimise.

        Parameters
        ----------
        max_current: float or np.array(self.n_C)
            Maximum magnitude of currents in each coil [A] permitted during optimisation.
            If max_current is supplied as a float, the float will be set as the
            maximum allowed current magnitude for all coils.
        """
        self.I_max = max_current / self.scale

    def optimise(self):
        """
        Optimiser handle. Used in __call__

        Returns np.array(self.n_C) of optimised currents in each coil [A].
        """
        opt = nlopt.opt(nlopt.LD_SLSQP, self.n_C)
        opt.set_min_objective(self.f_min_rms)
        opt.set_xtol_abs(1e-5)
        opt.set_xtol_rel(1e-5)
        opt.set_ftol_abs(1e-12)
        opt.set_ftol_rel(1e-10)
        # opt.set_maxtime(3)
        opt.set_maxeval(1000)
        opt.set_lower_bounds(-self.I_max)
        opt.set_upper_bounds(self.I_max)

        if self.n_CS == 0:
            n_f_constraints = 2 * self.n_PF
        else:
            n_f_constraints = 2 * self.n_PF + self.n_CS + 1
        tol = self.constraint_tol * np.ones(n_f_constraints)
        opt.add_inequality_mconstraint(self.constrain_forces, tol)
        tol = self.constraint_tol * np.ones(self.n_C)
        opt.add_inequality_mconstraint(self.constrain_fields, tol)

        x0 = np.clip(tikhonov(self.A, self.b, self.gamma), -self.I_max, self.I_max)
        currents = opt.optimize(x0)
        self.rms = opt.last_optimum_value()
        process_NLOPT_result(opt)
        self._I_star = currents * self.scale
        # self.sanity()
        return currents * self.scale

    def f_min_rms(self, vector, grad):
        """
        Objective function for nlopt optimisation (minimisation),
        consisting of a least-squares objective with Tikhonov
        regularisation term, which updates the gradient in-place.

        Parameters
        ----------
        vector: np.array(n_C)
            State vector of the array of coil currents.
        grad: np.array
            Local gradient of objective function used by LD NLOPT algorithms.
            Updated in-place.

        Returns
        -------
        rss: Value of objective function (figure of merit).
        """
        vector = vector * self.scale
        rss, err = self.get_rss(vector)
        if grad.size > 0:
            jac = 2 * self.A.T @ self.A @ vector
            jac -= 2 * self.A.T @ self.b
            jac += 2 * self.gamma * vector
            grad[:] = self.scale * jac
        if not rss > 0:
            raise EquilibriaError(
                "FBIOptimiser least-squares objective function less than zero."
            )
        return rss

    def get_rss(self, vector):
        """
        Calculates the value and residual of the least-squares objective
        function with Tikhonov regularisation term:

        ||(Ax - b)||² + Γ||x||²

        for the state vector x.

        Parameters
        ----------
        vector: np.array(n_C)
            State vector of the array of coil currents.

        Returns
        -------
        rss: Value of objective function (figure of merit).
        err: Residual (Ax - b) corresponding to the state vector x.
        """
        err = np.dot(self.A, vector) - self.b
        rss = err.T @ err + self.gamma * vector.T @ vector
        self.rms_error = rss
        return rss, err


class BreakdownOptimiser(SanityReporter, ForceFieldConstrainer):
    """
    Optimiser for the premagnetisation phase of the plasma. The sum of the
    PF coil currents is minimised (operating at maximum CS module voltages).
    Constraints are applied directly within the optimiser:
    - the maximum absolute current value per PF coil
    - the maxmimum poloidal magnetic field inside the breakdown zone
    - peak field inside the conductors

    Parameters
    ----------
    x_zone: float
        The X coordinate of the centre of the circular breakdown zone [m]
    z_zone: float
        The Z coordinate of the centre of the circular breakdown zone [m]
    r_zone: float
        The radius of the circular breakdown zone [m]
    b_zone_max: float
        The maximum field constraint inside the breakdown zone [T]
    max_currents: np.array(coils)
        The array of maximum coil currents [A]
    max_fields: np.array(n_coils)
        The array of maximum poloidal field [T]
    PF_Fz_max: float
        The maximum absolute vertical on a PF coil [N]
    CS_Fz_sum: float
        The maximum absolute vertical on all CS coils [N]
    CS_Fz_sep: float
        The maximum Central Solenoid vertical separation force [N]
    """

    def __init__(
        self,
        x_zone,
        z_zone,
        r_zone,
        b_zone_max,
        max_currents,
        max_fields,
        PF_Fz_max,
        CS_Fz_sum,
        CS_Fz_sep,
        **kwargs,
    ):
        self.scale = 1e6
        self.x_zone = x_zone
        self.z_zone = z_zone
        self.r_zone = r_zone
        self.B_zone_max = b_zone_max
        self._I_max = max_currents
        self.B_max = max_fields
        self.PF_Fz_max = PF_Fz_max / self.scale
        self.CS_Fz_sum = CS_Fz_sum / self.scale
        self.CS_Fz_sep = CS_Fz_sep / self.scale
        self.I_max = None
        self.rms = None
        self.constraint_tol = kwargs.get("constraint_tol", 1e-3)

    def __call__(self, eq):
        """
        Optimise the coil currents in an breakdown.

        Parameters
        ----------
        eq: Breakdown
            The breakdown to optimise the positions for

        Returns
        -------
        opt_currents: np.array(n_coils)
            The optimal currents for the controlled coils.
        """
        self.n_PF, self.n_CS = eq.coilset.n_PF, eq.coilset.n_CS
        self.n_C = eq.coilset.n_coils
        self.eq = eq

        # Set up stray field region
        self.zone = np.array(
            make_circle_arc(self.r_zone, self.x_zone, self.z_zone, n_points=20)
        )
        # Add centre and overwrite duplicate point
        self.zone[0][-1] = self.x_zone
        self.zone[1][-1] = self.z_zone

        # Build response matrices for optimisation
        self.eq.set_forcefield()
        self._build_matrices()

        self._I = eq.coilset.get_control_currents() / self.scale

        if self.I_max is None:
            self.I_max = eq.coilset.get_max_currents(self._I_max) / self.scale

        return self.optimise()

    def _build_matrices(self):
        cBx = np.zeros((len(self.zone.T), self.n_C))
        cBz = np.zeros((len(self.zone.T), self.n_C))
        for i, point in enumerate(self.zone.T):
            for j, coil in enumerate(self.eq.coilset.coils.values()):
                cBx[i, j] = coil.control_Bx(*point)
                cBz[i, j] = coil.control_Bz(*point)
        self.cBx = cBx
        self.cBz = cBz

        cpsi = np.zeros(self.n_C)
        for i, coil in enumerate(self.eq.coilset.coils.values()):
            cpsi[i] = coil.control_psi(self.x_zone, self.z_zone)
        self.cpsi = cpsi

    def optimise(self):
        """
        Optimiser handle for the BreakdownOptimiser object. Called on __call__
        """
        opt = nlopt.opt(nlopt.LN_COBYLA, self.n_C)
        opt.set_max_objective(self.f_maxflux)
        opt.set_ftol_abs(1e-6)
        opt.set_ftol_rel(1e-6)
        opt.set_maxtime(45)
        opt.set_lower_bounds(-self.I_max)
        opt.set_upper_bounds(self.I_max)

        if self.n_CS == 0:
            n_f_constraints = 2 * self.n_PF
        else:
            n_f_constraints = 2 * self.n_PF + self.n_CS + 1

        tol = self.constraint_tol * np.ones(n_f_constraints)
        opt.add_inequality_mconstraint(self.constrain_forces, tol)

        tol = self.constraint_tol * np.ones(self.n_C)
        opt.add_inequality_mconstraint(self.constrain_fields, tol)

        tol = 1e-6 * np.ones(len(self.zone.T))
        opt.add_inequality_mconstraint(self.constrain_breakdown, tol)

        # A vector of zeros would cause division by zero, so we instead used
        # a vector of 1 A per coil
        x0 = 1e-6 * np.ones(self.n_C)
        # Assist the optimiser: we know high currents in the CS are best
        x0[self.n_PF :] = self.I_max[self.n_PF :]

        currents = opt.optimize(x0)
        self.rms = opt.last_optimum_value()
        process_NLOPT_result(opt)
        self._I_star = currents * self.scale
        # self.sanity()
        return currents * self.scale

    def f_maxflux(self, x, grad):
        """
        Objective function for total psi maximisation minimisation
        """
        if grad.size > 0:
            grad[:] = self.scale * self.cpsi
            return self.scale * self.cpsi @ x

        return self.scale * self.cpsi @ x

    def f_maxcurrent(self, x, grad):
        """
        Objective function for total current sum minimisation

        \t:math:`\\sum_i^{n_C} \\lvert I_i\\rvert`
        """
        if grad.size > 0:
            grad[: self.n_PF] = 0
            grad[self.n_PF :] = 1

        return np.sum(x[self.n_PF :])

    def constrain_breakdown(self, constraint, vector, grad):
        """
        Constraint on the maximum field value insize the breakdown zone

        \t:math:`\\text{max}(B_p(\\mathbf{p})) \\forall \\mathbf{p}`
        \t:math:`\\in \\delta\\Omega \\leq B_{max}`
        """
        B = self.scale * np.hypot((self.cBx @ vector), (self.cBz @ vector))
        constraint[:] = B - self.B_zone_max
        if grad.size > 0:
            for i in range(len(self.zone.T)):
                for j in range(self.n_C):
                    grad[i, j] = (
                        self.scale
                        * (
                            self.cBx[i, j] * self.cBx[i, :] @ vector * self.scale
                            + self.cBz[i, j] * self.cBz[i, :] @ vector * self.scale
                        )
                        / B[i]
                    )
        return constraint

    def update_current_constraint(self, max_currents):
        """
        Update the current vector bounds. Must be called prior to optimise
        """
        self.I_max = max_currents / self.scale

    def copy(self):
        """
        Get a deep copy of the BreakdownOptimiser.
        """
        return deepcopy(self)


class BoundedCurrentOptimiser(EquilibriumOptimiser):
    """
    NLOpt based optimiser for coil currents subject to maximum current bounds.

    Parameters
    ----------
    coilset: CoilSet
        Coilset used to get coil current limits and number of coils.
    max_currents float or np.array(len(coilset._ccoils)) (default = None)
        Maximum allowed current for each independent coil current in coilset [A].
        If specified as a float, the float will set the maximum allowed current
        for all coils.
    gamma: float (default = 1e-7)
        Tikhonov regularisation parameter.
    opt_conditions: dict
    opt_conditions: dict
        (default {"xtol_rel": 1e-4, "xtol_abs": 1e-4,"ftol_rel": 1e-4, "ftol_abs": 1e-4})
        Termination conditions to pass to the optimiser.
    """

    def __init__(
        self,
        coilset,
        max_currents=None,
        gamma=1e-7,
        opt_conditions={
            "xtol_rel": 1e-4,
            "xtol_abs": 1e-4,
            "ftol_rel": 1e-4,
            "ftol_abs": 1e-4,
        },
    ):
        # noqa (N803)

        # Used scale for optimiser RoundoffLimited Error prevention
        self.scale = 1e6  # Scale for currents and forces (MA and MN)
        self.rms = None
        self.rms_error = None

        self.coilset = coilset

        if max_currents is not None:
            self.I_max = self.update_current_constraint(max_currents)
        else:
            self.I_max = np.inf
        self.gamma = gamma
        self.opt_conditions = opt_conditions

        # Set up optimiser
        self.opt = self.set_up_optimiser(len(self.coilset._ccoils))

    def update_current_constraint(self, max_currents):
        """
        Updates the current vector bounds. Must be called prior to optimise.

        Parameters
        ----------
        max_currents: float or np.array(len(self.coilset._ccoils))
            Maximum magnitude of currents in each coil [A] permitted during optimisation.
            If max_current is supplied as a float, the float will be set as the
            maximum allowed current magnitude for all coils.

        Returns
        -------
        i_max: float or np.array(len(self.coilset._ccoils))
            Maximum magnitude(s) of currents allowed in each coil.
        """
        i_max = max_currents / self.scale
        return i_max

    def set_up_optimiser(self, n_currents):
        """
        Set up NLOpt-based optimiser with algorithm,  bounds, tolerances, and
        constraint & objective functions.

        Parameters
        ----------
        n_currents: int
            Number of independent coil currents to optimise.
            Should be equal to eq.coilset._ccoils when called.

        Returns
        -------
        opt: nlopt.opt
            NLOpt optimiser to be used for optimisation.
        """
        # Initialise NLOpt optimiser, with optimisation strategy and length
        # of state vector
        opt = nlopt.opt(nlopt.LD_SLSQP, n_currents)
        # Set up objective function for optimiser
        opt.set_min_objective(self.f_min_objective)

        # Set tolerances for convergence of state vector and objective function
        opt.set_xtol_abs(self.opt_conditions["xtol_abs"])
        opt.set_xtol_rel(self.opt_conditions["xtol_rel"])
        opt.set_ftol_abs(self.opt_conditions["ftol_abs"])
        opt.set_ftol_rel(self.opt_conditions["ftol_rel"])

        # Set state vector bounds (current limits)
        opt.set_lower_bounds(-self.I_max)
        opt.set_upper_bounds(self.I_max)

        return opt

    def optimise(self):
        """
        Optimiser handle. Used in __call__

        Returns np.array(len(self.coilset._ccoils)) of optimised currents
        in each coil [A].
        """
        # Get initial currents, and trim to within current bounds.
        initial_currents = self.eq.coilset.get_control_currents() / self.scale
        initial_currents = np.clip(initial_currents, -self.I_max, self.I_max)

        # Optimise
        currents = self.opt.optimize(initial_currents)

        # Store found optimum of objective function and currents at optimum
        self.rms = self.opt.last_optimum_value()
        self._I_star = currents * self.scale
        process_NLOPT_result(self.opt)
        return currents * self.scale

    def f_min_objective(self, vector, grad):
        """
        Objective function for nlopt optimisation (minimisation),
        consisting of a least-squares objective with Tikhonov
        regularisation term, which updates the gradient in-place.

        Parameters
        ----------
        vector: np.array(n_C)
            State vector of the array of coil currents.
        grad: np.array
            Local gradient of objective function used by LD NLOPT algorithms.
            Updated in-place.

        Returns
        -------
        rss: Value of objective function (figure of merit).
        """
        vector = vector * self.scale
        rss, err = self.get_rss(vector)
        if grad.size > 0:
            jac = 2 * self.A.T @ self.A @ vector
            jac -= 2 * self.A.T @ self.b
            jac += 2 * self.gamma * self.gamma * vector
            grad[:] = self.scale * jac
        if not rss > 0:
            raise EquilibriaError(
                "Optimiser least-squares objective function less than zero or nan."
            )
        return rss

    def get_rss(self, vector):
        """
        Calculates the value and residual of the least-squares objective
        function with Tikhonov regularisation term:

        ||(Ax - b)||² + ||Γx||²

        for the state vector x.

        Parameters
        ----------
        vector: np.array(n_C)
            State vector of the array of coil currents.

        Returns
        -------
        rss: Value of objective function (figure of merit).
        err: Residual (Ax - b) corresponding to the state vector x.
        """
        err = np.dot(self.A, vector) - self.b
        rss = err.T @ err + self.gamma * self.gamma * vector.T @ vector
        self.rms_error = rss
        return rss, err

# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Interface for building and loading equilibria and coilset designs
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Type

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.limiter import Limiter
from bluemira.equilibria.opt_constraints import (
    MagneticConstraintSet,
    PsiBoundaryConstraint,
    PsiConstraint,
)
from bluemira.equilibria.opt_problems import (
    BreakdownCOP,
    BreakdownZoneStrategy,
    CoilsetOptimisationProblem,
    MinimalCurrentCOP,
    PulsedNestedPositionCOP,
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.physics import calc_psib
from bluemira.equilibria.profiles import Profile
from bluemira.equilibria.solve import (
    ConvergenceCriterion,
    DudsonConvergence,
    PicardIterator,
)
from bluemira.utilities.opt_problems import OptimisationConstraint
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.positioning import PositionMapper


@dataclass
class Snapshot:
    """
    Abstract object for grouping of equilibria objects in a given state.

    Parameters
    ----------
    eq
        The equilibrium at the snapshot
    coilset
        The coilset at the snapshot
    opt_problem
        The constraints at the snapshot
    profiles
        The profile at the snapshot
    optimiser
        The optimiser for the snapshot
    limiter
        The limiter for the snapshot
    tfcoil
        The PF coil placement boundary
    """

    eq: Equilibrium
    coilset: CoilSet
    constraints: Optional[CoilsetOptimisationProblem] = None
    profiles: Optional[Profile] = None
    optimiser: Optional[EquilibriumOptimiser] = None  # noqa: F821
    limiter: Optional[Limiter] = None
    tfcoil: Optional[Coordinates] = None  # noqa: F821

    def __post_init__(self):
        """Copy some variables on initialisation"""
        for field in fields(type(self)):
            if (val := getattr(self, field.name)) is not None and field.name not in (
                "constraints",
                "tfcoil",
            ):
                setattr(self, field.name, deepcopy(val))


class PulsedCoilsetDesign:
    """
    Abstract base class for the procedural design of a pulsed tokamak poloidal field
    coilset.
    """

    BREAKDOWN = "Breakdown"
    EQ_REF = "Reference"
    SOF = "SOF"
    EOF = "EOF"

    def __init__(self):
        self.snapshots = {}

    def take_snapshot(self, name, eq, coilset, problem, profiles=None):
        """
        Take a snapshot of the pulse.
        """
        if name in self.snapshots:
            bluemira_warn(f"Over-writing snapshot {name}!")

        self.snapshots[name] = Snapshot(
            eq, coilset, problem, profiles, limiter=self.limiter
        )

    def run_premagnetisation(self):
        """
        Run the breakdown optimisation problem
        """
        R_0 = self.params.R_0.value
        strategy = self._bd_strat_cls(
            R_0, self.params.A.value, self.params.tk_sol_ib.value
        )

        i_max = 30
        relaxed = all(self.coilset._flag_sizefix)
        for i in range(i_max):
            coilset = deepcopy(self.coilset)
            breakdown = Breakdown(coilset, self.grid)
            constraints = deepcopy(self._coil_cons)

            if relaxed:
                max_currents = self.coilset.get_max_current(0)
            else:
                max_currents = self.coilset.get_max_current(self.params.I_p.value)
                coilset.get_control_coils().current = max_currents
                coilset.discretisation = self._eq_settings["coil_mesh_size"]

            problem = self._bd_prob_cls(
                breakdown.coilset,
                breakdown,
                strategy,
                B_stray_max=self.params.B_premag_stray_max.value,
                B_stray_con_tol=self._bd_settings["B_stray_con_tol"],
                n_B_stray_points=self._bd_settings["n_B_stray_points"],
                optimiser=self._bd_opt,
                max_currents=max_currents,
                constraints=constraints,
            )
            coilset = problem.optimise(x0=max_currents, fixed_coils=False)
            breakdown.set_breakdown_point(*strategy.breakdown_point)
            psi_premag = breakdown.breakdown_psi

            if i == 0:
                psi_1 = psi_premag
            if relaxed or np.isclose(psi_premag, psi_1, rtol=1e-2):
                break

        else:
            raise EquilibriaError(
                "Unable to relax the breakdown optimisation for coil sizes."
            )

        bluemira_print(f"Premagnetisation flux = {2*np.pi * psi_premag:.2f} V.s")

        self._psi_premag = 2 * np.pi * psi_premag
        self.take_snapshot(self.BREAKDOWN, breakdown, coilset, problem)

    def run_reference_equilibrium(self):
        """
        Run a reference equilibrium.
        """
        coilset = deepcopy(self.coilset)
        eq = Equilibrium(
            coilset,
            self.grid,
            self.profiles,
        )
        opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
            coilset,
            eq,
            MagneticConstraintSet(self.eq_constraints),
            gamma=self._eq_settings["gamma"],
        )
        program = PicardIterator(
            eq,
            opt_problem,
            convergence=self._eq_convergence,
            relaxation=self._eq_settings["relaxation"],
            fixed_coils=True,
            plot=False,
        )
        program()

        eq_constraints = deepcopy(self.eq_constraints)
        for con in eq_constraints:
            if isinstance(con, (PsiConstraint, PsiBoundaryConstraint)):
                eq_constraints.remove(con)
        opt_problem = self._make_opt_problem(
            eq,
            deepcopy(self._eq_opt),
            self._get_max_currents(eq.coilset),
            current_constraints=None,
            eq_constraints=eq_constraints,
        )

        program = PicardIterator(
            eq,
            opt_problem,
            convergence=self._eq_convergence,
            relaxation=self._eq_settings["relaxation"],
            fixed_coils=True,
            plot=False,
        )
        program()

        self.take_snapshot(self.EQ_REF, eq, coilset, opt_problem, self.profiles)

    def calculate_sof_eof_fluxes(self, psi_premag: Optional[float] = None):
        """
        Calculate the SOF and EOF plasma boundary fluxes.
        """
        if psi_premag is None:
            if self.BREAKDOWN not in self.snapshots:
                self.run_premagnetisation()
            psi_premag = self.snapshots[self.BREAKDOWN].eq.breakdown_psi

        psi_sof = calc_psib(
            2 * np.pi * psi_premag,
            self.params.R_0.value,
            self.params.I_p.value,
            self.params.l_i.value,
            self.params.C_Ejima.value,
        )
        psi_eof = psi_sof - self.params.tau_flattop.value * self.params.v_burn.value
        return psi_sof, psi_eof

    def _get_max_currents(self, coilset):
        factor = self._eq_settings["peak_PF_current_factor"]
        return coilset.get_max_current(factor * self.params.I_p.value)

    def _get_sof_eof_opt_problems(self, psi_sof, psi_eof):
        eq_ref = self.snapshots[self.EQ_REF].eq
        max_currents_pf = self._get_max_currents(self.coilset.get_coiltype("PF"))
        max_currents = self._get_max_currents(self.coilset)

        opt_problems = []
        for psi_boundary in [psi_sof, psi_eof]:
            eq = deepcopy(eq_ref)
            eq.coilset.get_coiltype("PF").resize(max_currents_pf)
            optimiser = deepcopy(self._eq_opt)

            current_constraints = []
            if self._current_opt_cons:
                current_constraints += deepcopy(self._current_opt_cons)
            if self._coil_cons:
                current_constraints += deepcopy(self._coil_cons)

            eq_constraints = deepcopy(self.eq_constraints)
            for con in eq_constraints:
                if isinstance(con, (PsiBoundaryConstraint, PsiConstraint)):
                    con.target_value = psi_boundary / (2 * np.pi)
            for con in current_constraints:
                if isinstance(con, (PsiBoundaryConstraint, PsiConstraint)):
                    con.target_value = psi_boundary / (2 * np.pi)

            problem = self._make_opt_problem(
                eq, optimiser, max_currents, current_constraints, eq_constraints
            )

            opt_problems.append(problem)

        return opt_problems

    def _make_opt_problem(
        self, eq, optimiser, max_currents, current_constraints, eq_constraints
    ):
        if self._eq_prob_cls == MinimalCurrentCOP:
            constraints = eq_constraints
            if current_constraints:
                constraints += current_constraints

            problem = self._eq_prob_cls(
                eq.coilset,
                eq,
                optimiser,
                max_currents=max_currents,
                constraints=constraints,
            )
        elif self._eq_prob_cls == TikhonovCurrentCOP:
            problem = self._eq_prob_cls(
                eq.coilset,
                eq,
                MagneticConstraintSet(eq_constraints),
                gamma=self._eq_settings["gamma"],
                optimiser=optimiser,
                max_currents=max_currents,
                constraints=current_constraints,
            )
        return problem

    def converge_equilibrium(self, eq, problem):
        """
        Converge an equilibrium problem from a 'frozen' plasma optimised state.
        """
        program = PicardIterator(
            eq,
            problem,
            fixed_coils=True,
            convergence=self._eq_convergence,
            relaxation=self._eq_settings["relaxation"],
            plot=False,
        )
        program()

    def plot(self):
        """
        Plot the pulsed equilibrium problem.
        """
        n_snapshots = len(self.snapshots)
        if n_snapshots == 0:
            return

        f, ax = plt.subplots(1, n_snapshots)
        for i, (k, snap) in enumerate(self.snapshots.items()):
            axi = ax[i]
            snap.eq.plot(ax=axi)
            snap.coilset.plot(ax=axi)
            if k == "Breakdown":
                title = (
                    k + " $\\Psi_{bd}$: " + f"{2*np.pi * snap.eq.breakdown_psi:.2f} V.s"
                )
            else:
                title = (
                    k
                    + " $\\Psi_{b}$: "
                    + f"{2*np.pi * snap.eq.get_OX_points()[1][0].psi:.2f} V.s"
                )
            axi.set_title(title)
        return f


class FixedPulsedCoilsetDesign(PulsedCoilsetDesign):
    """
    Procedural design for a pulsed tokamak with a known, fixed PF coilset.

    Parameters
    ----------
    params: ParameterFrame
        Parameter frame with which to perform the problem
    coilset: CoilSet
        PF coilset to use in the equilibrium design
    grid: Grid
        Grid to use in the equilibrium design
    coil_constraints:
        pass
    """

    def __init__(
        self,
        params,
        coilset: CoilSet,
        grid: Grid,
        current_opt_constraints: Optional[List[OptimisationConstraint]],
        equilibrium_constraints: MagneticConstraintSet,
        profiles: Profile,
        breakdown_strategy_cls: Type[BreakdownZoneStrategy],
        breakdown_problem_cls: Type[BreakdownCOP],
        breakdown_optimiser: Optimiser = Optimiser(
            "COBYLA", opt_conditions={"max_eval": 5000, "ftol_rel": 1e-10}
        ),
        breakdown_settings: Optional[Dict] = None,
        equilibrium_problem_cls: Type[CoilsetOptimisationProblem] = MinimalCurrentCOP,
        equilibrium_optimiser: Optimiser = Optimiser(
            "SLSQP", opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6}
        ),
        equilibrium_convergence: ConvergenceCriterion = DudsonConvergence(1e-2),
        equilibrium_settings: Optional[Dict] = None,
        limiter: Optional[Limiter] = None,
    ):
        self.params = params
        self.coilset = coilset
        self.grid = grid
        self.profiles = profiles
        self.limiter = limiter
        self.eq_constraints = equilibrium_constraints

        self._bd_strat_cls = breakdown_strategy_cls
        self._bd_prob_cls = breakdown_problem_cls
        self._bd_opt = breakdown_optimiser

        self._bd_settings = {"B_stray_con_tol": 1e-8, "n_B_stray_points": 20}
        if breakdown_settings:
            self._bd_settings = {**self._bd_settings, **breakdown_settings}

        self._eq_prob_cls = equilibrium_problem_cls
        self._eq_opt = equilibrium_optimiser
        self._eq_convergence = equilibrium_convergence

        self._eq_settings = {
            "gamma": 1e-8,
            "relaxation": 0.1,
            "coil_mesh_size": 0.3,
            "peak_PF_current_factor": 1.5,
        }
        if equilibrium_settings:
            self._eq_settings = {**self._eq_settings, **equilibrium_settings}

        self._current_opt_cons = current_opt_constraints

        super().__init__()

    def optimise_currents(self):
        """
        Optimise the coil currents at the start and end of the current flat-top.
        """
        psi_sof, psi_eof = self.calculate_sof_eof_fluxes()
        if self.EQ_REF not in self.snapshots:
            self.run_reference_equilibrium()

        sof_opt_problem, eof_opt_problem = self._get_sof_eof_opt_problems(
            psi_sof, psi_eof
        )

        for snap, problem in zip(
            [self.SOF, self.EOF], [sof_opt_problem, eof_opt_problem]
        ):
            eq = problem.eq
            self.converge_equilibrium(eq, problem)
            self.take_snapshot(snap, eq, eq.coilset, problem, eq.profiles)


class OptimisedPulsedCoilsetDesign(PulsedCoilsetDesign):
    """
    Procedural design for a pulsed tokamak with no prescribed PF coil positions.

    Parameters
    ----------
    params: ParameterFrame
        Parameter frame with which to perform the problem
    coilset: CoilSet
        PF coilset to use in the equilibrium design
    grid: Grid
        Grid to use in the equilibrium design
    current_opt_constraints: Optional[List[OptimisationConstraint]]
        List of current optimisation constraints for equilibria
    coil_constraints: Optional[List[OptimisationConstraint]]
        List of coil current optimisation constraints for all snapshots (including
        breakdown)
    equilibrium_constraints: List[OptimisationConstraint]
        List of magnetic constraints to use for equilibria. Depending on the optimisation
        problem, these may be used in the objective function or constraints
    profiles: Profile
        Plasma profile object to use when solving equilibria
    breakdown_strategy_cls: Type[BreakdownZoneStrategy]
        BreakdownZoneStrategy class to use when determining breakdown constraints
    breakdown_problem_cls: Type[BreakdownCOP]
        Coilset optimisation problem class for the breakdown phase
    breakdown_optimiser: Optimiser
        Optimiser for the breakdown
    equilibrium_problem_cls: Type[CoilsetOptimisationProblem]
        Coilset optimisation problem class for the equilibria and current vector
    equilibrium_optimiser: Optimiser
        Optimiser for the equilibria and current vector
    equilibrium_convergence: ConvergenceCriterion
        Convergence criteria to use when solving equilibria
    equilibrium_settings: Optional[Dict]
        Settings for the solution of equilibria
    position_problem_cls: Type[PulsedNestedPositionCOP]
        Coilset optimisation problem class for the coil positions
    position_optimiser: Optimiser
        Optimiser for the coil positions
    limiter: Optional[Limiter]
        Limiter to use when solving equilibria
    """

    def __init__(
        self,
        params,
        coilset: CoilSet,
        position_mapper: PositionMapper,
        grid: Grid,
        current_opt_constraints: Optional[List[OptimisationConstraint]],
        coil_constraints: Optional[List[OptimisationConstraint]],
        equilibrium_constraints: List[OptimisationConstraint],
        profiles: Profile,
        breakdown_strategy_cls: Type[BreakdownZoneStrategy],
        breakdown_problem_cls: Type[BreakdownCOP],
        breakdown_optimiser: Optimiser = Optimiser(
            "COBYLA", opt_conditions={"max_eval": 5000, "ftol_rel": 1e-10}
        ),
        breakdown_settings: Optional[Dict] = None,
        equilibrium_problem_cls: Type[CoilsetOptimisationProblem] = MinimalCurrentCOP,
        equilibrium_optimiser: Optimiser = Optimiser(
            "SLSQP", opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6}
        ),
        equilibrium_convergence: ConvergenceCriterion = DudsonConvergence(1e-2),
        equilibrium_settings: Optional[Dict] = None,
        position_problem_cls: Type[PulsedNestedPositionCOP] = PulsedNestedPositionCOP,
        position_optimiser: Optimiser = Optimiser(
            "COBYLA", opt_conditions={"max_eval": 100, "ftol_rel": 1e-4}
        ),
        limiter: Optional[Limiter] = None,
    ):
        self.params = params
        self._eq_settings = {
            "gamma": 1e-8,
            "relaxation": 0.1,
            "coil_mesh_size": 0.3,
            "peak_PF_current_factor": 1.5,
        }
        if equilibrium_settings:
            self._eq_settings = {**self._eq_settings, **equilibrium_settings}

        self.coilset = self._prepare_coilset(coilset)
        self.position_mapper = position_mapper
        self.grid = grid
        self.profiles = profiles
        self.limiter = limiter
        self.eq_constraints = equilibrium_constraints

        self._bd_strat_cls = breakdown_strategy_cls
        self._bd_prob_cls = breakdown_problem_cls
        self._bd_settings = breakdown_settings or {
            "B_stray_con_tol": 1e-8,
            "n_B_stray_points": 20,
        }
        self._bd_opt = breakdown_optimiser

        self._eq_prob_cls = equilibrium_problem_cls
        self._eq_opt = equilibrium_optimiser
        self._eq_convergence = equilibrium_convergence

        self._pos_prob_cls = position_problem_cls
        self._pos_opt = position_optimiser

        self._current_opt_cons = current_opt_constraints
        self._coil_cons = coil_constraints

        super().__init__()

    def _prepare_coilset(self, coilset):
        coilset = deepcopy(coilset)
        coilset.discretisation = np.where(
            coilset._flag_sizefix,
            self._eq_settings["coil_mesh_size"],
            coilset.discretisation,
        )
        return coilset

    def optimise_positions(self, verbose=False):
        """
        Optimise the coil positions for the start and end of the current flat-top.
        """
        psi_sof, psi_eof = self.calculate_sof_eof_fluxes()
        if self.EQ_REF not in self.snapshots:
            self.run_reference_equilibrium()

        sub_opt_problems = self._get_sof_eof_opt_problems(psi_sof, psi_eof)

        pos_opt_problem = self._pos_prob_cls(
            sub_opt_problems[0].eq.coilset,
            self.position_mapper,
            sub_opt_problems,
            self._pos_opt,
            constraints=None,
        )
        optimised_coilset = pos_opt_problem.optimise(verbose=verbose)

        optimised_coilset = self._consolidate_coilset(
            optimised_coilset, sub_opt_problems
        )

        for snap, problem in zip([self.SOF, self.EOF], sub_opt_problems):
            eq = problem.eq
            self.converge_equilibrium(eq, problem)
            self.take_snapshot(snap, eq, eq.coilset, problem, eq.profiles)

        # Re-run breakdown
        psi_bd_orig = self._psi_premag
        self.coilset = optimised_coilset
        self.run_premagnetisation()
        if self._psi_premag < psi_bd_orig - 2.0:
            bluemira_warn(
                f"Breakdown flux significantly lower with optimised coil positions: {self._psi_premag:.2f} < {psi_bd_orig:.2f}"
            )
        return optimised_coilset

    def _consolidate_coilset(self, coilset, sub_opt_problems):
        """
        Set the current bounds on the current optimisation problems, fix coil sizes, and
        mesh.
        """
        max_cs_currents = coilset.get_coiltype("CS").get_max_current(0.0)

        for problem in sub_opt_problems:
            pf_coils = problem.eq.coilset.get_coiltype("PF").get_control_coils()
            pf_current = pf_coils.current
            max_pf_current = np.max(np.abs(pf_current))
            pf_coils.resize(max_pf_current)
            pf_coils.fix_sizes()
            pf_coils.discretisation = self._eq_settings.coil_mesh_size
            problem.set_current_bounds(
                np.concatenate(
                    [np.full(pf_current.size, max_pf_current), max_cs_currents]
                )
            )

        consolidated_coilset = deepcopy(problem.eq.coilset)
        consolidated_coilset.fix_sizes()
        consolidated_coilset.get_control_coils().current = 0
        return consolidated_coilset

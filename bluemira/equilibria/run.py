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

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Dict, Iterable, List, Optional, Type

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.limiter import Limiter
from bluemira.equilibria.opt_constraints import (
    MagneticConstraint,
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
    eq:
        The equilibrium at the snapshot
    coilset:
        The coilset at the snapshot
    opt_problem:
        The constraints at the snapshot
    profiles:
        The profile at the snapshot
    optimiser:
        The optimiser for the snapshot
    limiter:
        The limiter for the snapshot
    tfcoil:
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


@dataclass
class BreakdownCOPSettings:
    """Breakdown settings for PulsedCoilsetDesign"""

    B_stray_con_tol: float
    n_B_stray_points: int


@dataclass
class EQSettings:
    """Equilibrium settings for PulsedCoilsetDesign"""

    coil_mesh_size: float
    gamma: float
    relaxation: float
    peak_PF_current_factor: float


@dataclass
class PulsedCoilsetDesignFrame(ParameterFrame):
    """PulsedCoilsetDesign Parameters"""

    A: Parameter[float]
    B_premag_stray_max: Parameter[float]
    C_Ejima: Parameter[float]
    I_p: Parameter[float]
    l_i: Parameter[float]
    R_0: Parameter[float]
    tau_flattop: Parameter[float]
    tk_sol_ib: Parameter[float]
    v_burn: Parameter[float]


class PulsedCoilsetDesign(ABC):
    """
    Abstract base class for the procedural design of a pulsed tokamak poloidal field
    coilset.

    Parameters
    ----------
    params:
        Parameter frame with which to perform the problem
    coilset:
        PF coilset to use in the equilibrium design
    grid:
        Grid to use in the equilibrium design
    equilibrium_constraints:
        List of magnetic constraints to use for equilibria. Depending on the optimisation
        problem, these may be used in the objective function or constraints
    profiles:
        Plasma profile object to use when solving equilibria
    breakdown_strategy_cls:
        BreakdownZoneStrategy class to use when determining breakdown constraints
    breakdown_problem_cls:
        Coilset optimisation problem class for the breakdown phase
    breakdown_optimiser:
        Optimiser for the breakdown,
        default is COBYLA with ftol_rel=1e-10 and max_eval=5000
    breakdown_settings:
        Breakdown optimiser settings
    equilibrium_problem_cls:
        Coilset optimisation problem class for the equilibria and current vector
    equilibrium_optimiser:
        Optimiser for the equilibria and current vector
        default is SLSQP with ftol_rel=1e-6 and max_eval=1000
    equilibrium_convergence:
        Convergence criteria to use when solving equilibria
        default is 1e-2 DudsonConvergence
    equilibrium_settings:
        Settings for the solution of equilibria
    current_opt_constraints:
        List of current optimisation constraints for equilibria
    coil_constraints:
        List of coil current optimisation constraints for all snapshots (including
        breakdown)
    limiter:
        Limiter to use when solving equilibria
    """

    BREAKDOWN = "Breakdown"
    EQ_REF = "Reference"
    SOF = "SOF"
    EOF = "EOF"

    def __init__(
        self,
        params: ParameterFrame,
        coilset: CoilSet,
        grid: Grid,
        equilibrium_constraints: List[MagneticConstraint],
        profiles: Profile,
        breakdown_strategy_cls: Type[BreakdownZoneStrategy],
        breakdown_problem_cls: Type[BreakdownCOP],
        breakdown_optimiser: Optional[Optimiser] = None,
        breakdown_settings: Optional[Dict] = None,
        equilibrium_problem_cls: Type[CoilsetOptimisationProblem] = MinimalCurrentCOP,
        equilibrium_optimiser: Optional[Optimiser] = None,
        equilibrium_convergence: Optional[ConvergenceCriterion] = None,
        equilibrium_settings: Optional[Dict] = None,
        current_opt_constraints: Optional[List[OptimisationConstraint]] = None,
        coil_constraints: Optional[List[OptimisationConstraint]] = None,
        limiter: Optional[Limiter] = None,
    ):
        self.snapshots = {}
        self.params = PulsedCoilsetDesignFrame.from_frame(params)
        self.coilset = coilset
        self.grid = grid

        self._current_opt_cons = current_opt_constraints
        self.eq_constraints = equilibrium_constraints
        self.profiles = profiles

        self._bd_strat_cls = breakdown_strategy_cls
        self._bd_prob_cls = breakdown_problem_cls
        self._bd_opt = breakdown_optimiser or Optimiser(
            "COBYLA", opt_conditions={"max_eval": 5000, "ftol_rel": 1e-10}
        )
        self._bd_settings = breakdown_settings

        self._eq_settings = equilibrium_settings
        self._eq_convergence = equilibrium_convergence or DudsonConvergence(1e-2)
        self._eq_prob_cls = equilibrium_problem_cls
        self._eq_opt = equilibrium_optimiser or Optimiser(
            "SLSQP", opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6}
        )

        self._coil_cons = [] if coil_constraints is None else coil_constraints
        self.limiter = limiter

    @abstractmethod
    def optimise(self, *args, **kwargs) -> CoilSet:
        """
        Run pulsed coilset design optimisation
        """
        pass

    @property
    def _bd_settings(self) -> BreakdownCOPSettings:
        return self.__bd_settings

    @_bd_settings.setter
    def _bd_settings(self, value: Optional[Dict] = None):
        self.__bd_settings = BreakdownCOPSettings(
            **{
                **{"B_stray_con_tol": 1e-8, "n_B_stray_points": 20},
                **({} if value is None else value),
            }
        )

    @property
    def _eq_settings(self) -> EQSettings:
        return self.__eq_settings

    @_eq_settings.setter
    def _eq_settings(self, value: Optional[Dict] = None):
        self.__eq_settings = EQSettings(
            **{
                **{
                    "gamma": 1e-8,
                    "relaxation": 0.1,
                    "coil_mesh_size": 0.3,
                    "peak_PF_current_factor": 1.5,
                },
                **({} if value is None else value),
            }
        )

    def take_snapshot(
        self,
        name: str,
        eq: Equilibrium,
        coilset: CoilSet,
        problem: CoilsetOptimisationProblem,
        profiles: Profile = None,
    ):
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
                coilset.discretisation = self._eq_settings.coil_mesh_size

            problem = self._bd_prob_cls(
                breakdown.coilset,
                breakdown,
                strategy,
                B_stray_max=self.params.B_premag_stray_max.value,
                B_stray_con_tol=self._bd_settings.B_stray_con_tol,
                n_B_stray_points=self._bd_settings.n_B_stray_points,
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
            gamma=self._eq_settings.gamma,
        )
        program = PicardIterator(
            eq,
            opt_problem,
            convergence=self._eq_convergence,
            relaxation=self._eq_settings.relaxation,
            fixed_coils=True,
            plot=False,
        )
        program()

        opt_problem = self._make_opt_problem(
            eq,
            deepcopy(self._eq_opt),
            self._get_max_currents(eq.coilset),
            current_constraints=None,
            eq_constraints=[
                deepcopy(con)
                for con in self.eq_constraints
                if not isinstance(con, (PsiConstraint, PsiBoundaryConstraint))
            ],
        )

        program = PicardIterator(
            eq,
            opt_problem,
            convergence=self._eq_convergence,
            relaxation=self._eq_settings.relaxation,
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

    def _get_max_currents(self, coilset: CoilSet) -> np.ndarray:
        return coilset.get_max_current(
            self._eq_settings.peak_PF_current_factor * self.params.I_p.value
        )

    def get_sof_eof_opt_problems(
        self, psi_sof: float, psi_eof: float
    ) -> List[CoilsetOptimisationProblem]:
        """Get start of flat top and end of flat top optimisation problems"""
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
            for constraints in (eq_constraints, current_constraints):
                for con in constraints:
                    if isinstance(con, (PsiBoundaryConstraint, PsiConstraint)):
                        con.target_value = psi_boundary / (2 * np.pi)

            opt_problems.append(
                self._make_opt_problem(
                    eq, optimiser, max_currents, current_constraints, eq_constraints
                )
            )

        return opt_problems

    def _make_opt_problem(
        self,
        eq: Equilibrium,
        optimiser: Optimiser,
        max_currents: np.ndarray,
        current_constraints: List[OptimisationConstraint],
        eq_constraints: List[MagneticConstraint],
    ) -> CoilsetOptimisationProblem:
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
                gamma=self._eq_settings.gamma,
                optimiser=optimiser,
                max_currents=max_currents,
                constraints=current_constraints,
            )
        else:
            raise EquilibriaError(
                "Only MinimalCurrentCOP and TikhonovCurrentCOP"
                " equilibrium problems supported"
            )
        return problem

    def converge_equilibrium(self, eq: Equilibrium, problem: CoilsetOptimisationProblem):
        """
        Converge an equilibrium problem from a 'frozen' plasma optimised state.
        """
        program = PicardIterator(
            eq,
            problem,
            fixed_coils=True,
            convergence=self._eq_convergence,
            relaxation=self._eq_settings.relaxation,
            plot=False,
        )
        program()

    def converge_and_snapshot(
        self,
        sub_opt_problems: Iterable[CoilsetOptimisationProblem],
        problem_names: Iterable[str] = (SOF, EOF),
    ):
        """Converge equilibrium optimisation problems and take snapshots"""
        for snap, problem in zip(problem_names, sub_opt_problems):
            eq = problem.eq
            self.converge_equilibrium(eq, problem)
            self.take_snapshot(snap, eq, eq.coilset, problem, eq.profiles)

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
            if k == self.BREAKDOWN:
                psi_name, psi_val = r"$\Psi_{bd}$", snap.eq.breakdown_psi
            else:
                psi_name, psi_val = r"$\Psi_{b}$", snap.eq.get_OX_points()[1][0].psi

            axi.set_title(f"{k} {psi_name}: {2* np.pi * psi_val:.2f} V.s")
        return f


class FixedPulsedCoilsetDesign(PulsedCoilsetDesign):
    """
    Procedural design for a pulsed tokamak with a known, fixed PF coilset.
    """

    def optimise(self) -> CoilSet:
        """
        Run pulsed coilset design optimisation
        """
        self.optimise_currents()
        return self.coilset

    def optimise_currents(self):
        """
        Optimise the coil currents at the start and end of the current flat-top.
        """
        psi_sof, psi_eof = self.calculate_sof_eof_fluxes()
        if self.EQ_REF not in self.snapshots:
            self.run_reference_equilibrium()

        self.converge_and_snapshot(self.get_sof_eof_opt_problems(psi_sof, psi_eof))


class OptimisedPulsedCoilsetDesign(PulsedCoilsetDesign):
    """
    Procedural design for a pulsed tokamak with no prescribed PF coil positions.

    Parameters
    ----------
    params:
        Parameter frame with which to perform the problem
    coilset:
        PF coilset to use in the equilibrium design
    position_mapper:
        Normalised coil position mapping
    grid:
        Grid to use in the equilibrium design
    equilibrium_constraints:
        List of magnetic constraints to use for equilibria. Depending on the optimisation
        problem, these may be used in the objective function or constraints
    profiles:
        Plasma profile object to use when solving equilibria
    breakdown_strategy_cls:
        BreakdownZoneStrategy class to use when determining breakdown constraints
    breakdown_problem_cls:
        Coilset optimisation problem class for the breakdown phase
    breakdown_optimiser:
        Optimiser for the breakdown,
        default is COBYLA with ftol_rel=1e-10 and max_eval=5000
    breakdown_settings:
        Breakdown optimiser settings
    equilibrium_problem_cls:
        Coilset optimisation problem class for the equilibria and current vector
    equilibrium_optimiser:
        Optimiser for the equilibria and current vector
        default is SLSQP with ftol_rel=1e-6 and max_eval=1000
    equilibrium_convergence:
        Convergence criteria to use when solving equilibria
        default is 1e-2 DudsonConvergence
    equilibrium_settings:
        Settings for the solution of equilibria
    current_opt_constraints:
        List of current optimisation constraints for equilibria
    coil_constraints:
        List of coil current optimisation constraints for all snapshots (including
        breakdown)
    limiter:
        Limiter to use when solving equilibria
    position_problem_cls:
        Coilset optimisation problem class for the coil positions
    position_optimiser:
        Optimiser for the coil positions
        default is COBYLA with ftol_rel=1e-4 and max_eval=100
    """

    def __init__(
        self,
        params: ParameterFrame,
        coilset: CoilSet,
        position_mapper: PositionMapper,
        grid: Grid,
        equilibrium_constraints: List[OptimisationConstraint],
        profiles: Profile,
        breakdown_strategy_cls: Type[BreakdownZoneStrategy],
        breakdown_problem_cls: Type[BreakdownCOP],
        breakdown_optimiser: Optional[Optimiser] = None,
        breakdown_settings: Optional[Dict] = None,
        equilibrium_problem_cls: Type[CoilsetOptimisationProblem] = MinimalCurrentCOP,
        equilibrium_optimiser: Optional[Optimiser] = None,
        equilibrium_convergence: ConvergenceCriterion = None,
        equilibrium_settings: Optional[Dict] = None,
        current_opt_constraints: Optional[List[OptimisationConstraint]] = None,
        coil_constraints: Optional[List[OptimisationConstraint]] = None,
        limiter: Optional[Limiter] = None,
        position_problem_cls: Type[PulsedNestedPositionCOP] = PulsedNestedPositionCOP,
        position_optimiser: Optional[Optimiser] = None,
    ):
        super().__init__(
            params,
            coilset,
            grid,
            equilibrium_constraints,
            profiles,
            breakdown_strategy_cls,
            breakdown_problem_cls,
            breakdown_optimiser,
            breakdown_settings,
            equilibrium_problem_cls,
            equilibrium_optimiser,
            equilibrium_convergence,
            equilibrium_settings,
            current_opt_constraints,
            coil_constraints,
            limiter,
        )
        self.coilset = self._prepare_coilset(self.coilset)
        self.position_mapper = position_mapper

        self._pos_prob_cls = position_problem_cls
        self._pos_opt = position_optimiser or Optimiser(
            "COBYLA", opt_conditions={"max_eval": 100, "ftol_rel": 1e-4}
        )

    def _prepare_coilset(self, coilset: CoilSet) -> CoilSet:
        coilset = deepcopy(coilset)
        coilset.discretisation = np.where(
            coilset._flag_sizefix,
            self._eq_settings.coil_mesh_size,
            coilset.discretisation,
        )
        return coilset

    def optimise(self, verbose: bool = False) -> CoilSet:
        """
        Optimise the coil positions for the start and end of the current flat-top.
        """
        psi_sof, psi_eof = self.calculate_sof_eof_fluxes()
        if self.EQ_REF not in self.snapshots:
            self.run_reference_equilibrium()

        sub_opt_problems = self.get_sof_eof_opt_problems(psi_sof, psi_eof)

        pos_opt_problem = self._pos_prob_cls(
            sub_opt_problems[0].eq.coilset,
            self.position_mapper,
            sub_opt_problems,
            self._pos_opt,
            constraints=None,
        )

        optimised_coilset = self._consolidate_coilset(
            pos_opt_problem.optimise(verbose=verbose), sub_opt_problems
        )

        self.converge_and_snapshot(sub_opt_problems)

        # Re-run breakdown
        psi_bd_orig = self._psi_premag
        self.coilset = optimised_coilset
        self.run_premagnetisation()
        if self._psi_premag < psi_bd_orig - 2.0:
            bluemira_warn(
                f"Breakdown flux significantly lower with optimised coil positions: {self._psi_premag:.2f} < {psi_bd_orig:.2f}"
            )
        return optimised_coilset

    def _consolidate_coilset(
        self, coilset: CoilSet, sub_opt_problems: Iterable[CoilsetOptimisationProblem]
    ) -> CoilSet:
        """
        Set the current bounds on the current optimisation problems, fix coil sizes, and
        mesh.
        """
        max_cs_currents = coilset.get_coiltype("CS").get_max_current(0.0)
        pf_currents = []
        for problem in sub_opt_problems:
            pf_coils = problem.eq.coilset.get_coiltype("PF").get_control_coils()
            pf_currents.append(np.abs(pf_coils.current))

        max_pf_currents = np.max(pf_currents, axis=0)
        # Relax the max currents a bit to avoid oscillation
        max_pf_current = self._eq_settings.peak_PF_current_factor * self.params.I_p.value
        max_pf_currents = np.clip(1.1 * max_pf_currents, 0, max_pf_current)

        for problem in sub_opt_problems:
            pf_coils = problem.eq.coilset.get_coiltype("PF").get_control_coils()
            pf_coils.resize(max_pf_currents)
            pf_coils.fix_sizes()
            pf_coils.discretisation = self._eq_settings.coil_mesh_size
            problem.set_current_bounds(
                np.concatenate([max_pf_currents, max_cs_currents])
            )

        consolidated_coilset = deepcopy(problem.eq.coilset)
        consolidated_coilset.fix_sizes()
        consolidated_coilset.get_control_coils().current = 0
        return consolidated_coilset

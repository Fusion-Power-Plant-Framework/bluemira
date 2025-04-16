# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Interface for building and loading equilibria and coilset designs
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.equilibria.diagnostics import PicardDiagnosticOptions
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium, MHDState
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.optimisation.constraints import (
    MagneticConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
    PsiConstraint,
    UpdateableConstraint,
)
from bluemira.equilibria.optimisation.problem import (
    BreakdownCOP,
    BreakdownZoneStrategy,
    CircularZoneStrategy,
    CoilsetOptimisationProblem,
    MinimalCurrentCOP,
    PulsedNestedPositionCOP,
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.physics import calc_psib
from bluemira.equilibria.solve import (
    ConvergenceCriterion,
    DudsonConvergence,
    PicardIterator,
)
from bluemira.optimisation import Algorithm, AlgorithmType

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from bluemira.equilibria.coils import CoilSet
    from bluemira.equilibria.grid import Grid
    from bluemira.equilibria.limiter import Limiter
    from bluemira.equilibria.optimisation.problem.base import CoilsetOptimiserResult
    from bluemira.equilibria.profiles import Profile
    from bluemira.geometry.coordinates import Coordinates
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
    optimisation_result:
        The optimisation result
    limiter:
        The limiter for the snapshot
    tfcoil:
        The PF coil placement boundary
    """

    eq: MHDState
    coilset: CoilSet
    constraints: CoilsetOptimisationProblem | None = None
    profiles: Profile | None = None
    optimisation_result: CoilsetOptimiserResult | None = None
    iterator: PicardIterator | None = None
    limiter: Limiter | None = None
    tfcoil: Coordinates | None = None

    def __post_init__(self):
        """Copy some variables on initialisation"""
        for fld in fields(type(self)):
            if (val := getattr(self, fld.name)) is not None and fld.name not in {
                "constraints",
                "tfcoil",
            }:
                setattr(self, fld.name, deepcopy(val))


@dataclass
class BreakdownCOPConfig:
    """Breakdown settings for PulsedCoilsetDesign"""

    problem: type[BreakdownCOP] = BreakdownCOP
    strategy: type[BreakdownZoneStrategy] = CircularZoneStrategy
    algorithm: AlgorithmType = Algorithm.COBYLA
    opt_conditions: dict[str, float | int] = field(
        default_factory=lambda: {"max_eval": 5000, "ftol_rel": 1e-10}
    )
    B_stray_con_tol: float = 1e-8
    n_B_stray_points: int = 20
    iter_max: int = 30

    def make_opt_problem(
        self, breakdown, strategy, max_currents, constraints, B_stray_max
    ):
        """Make breakdown optimisation problem

        Returns
        -------
        :
            The breakdown problem
        """
        return self.problem(
            breakdown.coilset,
            breakdown,
            strategy,
            B_stray_max=B_stray_max,
            B_stray_con_tol=self.B_stray_con_tol,
            n_B_stray_points=self.n_B_stray_points,
            max_currents=max_currents,
            opt_algorithm=self.algorithm,
            opt_conditions=self.opt_conditions,
            constraints=constraints,
        )


@dataclass
class EQConfig:
    """Equilibrium settings for PulsedCoilsetDesign"""

    problem: type[CoilsetOptimisationProblem] = MinimalCurrentCOP
    convergence: ConvergenceCriterion = field(
        default_factory=lambda: DudsonConvergence(1e-2)
    )
    algorithm: AlgorithmType = Algorithm.SLSQP
    opt_conditions: dict[str, float | int] = field(
        default_factory=lambda: {"max_eval": 1000, "ftol_rel": 1e-6}
    )
    opt_parameters: dict[str, Any] = field(
        default_factory=lambda: {"initial_step": 0.03}
    )
    coil_mesh_size: float = 0.3
    gamma: float = 1e-8
    relaxation: float = 0.1
    peak_PF_current_factor: float = 1.5
    diagnostic_plotting: PicardDiagnosticOptions = field(
        default_factory=PicardDiagnosticOptions
    )

    def make_opt_problem(
        self,
        eq: Equilibrium,
        max_currents: npt.NDArray[np.float64],
        current_constraints: list[UpdateableConstraint] | None,
        eq_constraints: list[MagneticConstraint],
    ) -> CoilsetOptimisationProblem:
        """Make equilibria optimisation problem

        Returns
        -------
        :
            The equilibria problem

        Raises
        ------
        EquilibriaError
            Unimplemented setup for equilibria problem
        """
        if self.problem == MinimalCurrentCOP:
            constraints = eq_constraints
            if current_constraints:
                constraints += current_constraints

            problem = self.problem(
                eq.coilset,
                eq,
                max_currents=max_currents,
                opt_conditions=self.opt_conditions,
                opt_algorithm=self.algorithm,
                constraints=constraints,
            )
        elif self.problem == TikhonovCurrentCOP:
            problem = self.problem(
                eq.coilset,
                eq,
                max_currents=max_currents,
                opt_conditions=self.opt_conditions,
                opt_algorithm=self.algorithm,
                opt_parameters=self.opt_parameters,
                constraints=current_constraints,
                targets=MagneticConstraintSet(eq_constraints),
                gamma=self.gamma,
            )
        else:
            raise EquilibriaError(
                "Only MinimalCurrentCOP and TikhonovCurrentCOP"
                " equilibrium problems supported"
            )
        return problem


@dataclass
class PositionConfig:
    """Position optimiser settings"""

    problem: type[PulsedNestedPositionCOP] = PulsedNestedPositionCOP
    algorithm: AlgorithmType = Algorithm.COBYLA
    opt_conditions: dict[str, float | int] = field(
        default_factory=lambda: {"max_eval": 100, "ftol_rel": 1e-4}
    )

    def make_opt_problem(self, position_mapper, sub_opt_problems):
        """
        Make outer position optimisation problem

        Returns
        -------
        :
            The position problem
        """
        return self.problem(
            sub_opt_problems[0].eq.coilset,
            position_mapper,
            sub_opt_problems,
            self.algorithm,
            self.opt_conditions,
            constraints=None,
        )


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
    breakdown_settings:
        Breakdown optimiser settings
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
        equilibrium_constraints: list[MagneticConstraint],
        profiles: Profile,
        breakdown_settings: dict | BreakdownCOPConfig | None = None,
        equilibrium_settings: dict | EQConfig | None = None,
        current_opt_constraints: list[UpdateableConstraint] | None = None,
        coil_constraints: list[UpdateableConstraint] | None = None,
        limiter: Limiter | None = None,
    ):
        self.snapshots = {}
        self.params = PulsedCoilsetDesignFrame.from_frame(params)
        self.coilset = coilset
        self.grid = grid

        self._current_opt_cons = current_opt_constraints
        self.eq_constraints = equilibrium_constraints
        self.profiles = profiles
        self.bd_config = breakdown_settings
        self.eq_config = equilibrium_settings

        self._coil_cons = [] if coil_constraints is None else coil_constraints
        self.limiter = limiter

    @abstractmethod
    def optimise(self, *args, **kwargs) -> CoilSet:
        """Run pulsed coilset design optimisation."""

    @property
    def bd_config(self) -> BreakdownCOPConfig:
        """Breakdown COP settings."""
        return self._bd_config

    @bd_config.setter
    def bd_config(self, value: dict | BreakdownCOPConfig | None = None):
        """Breakdown COP settings."""
        if isinstance(value, BreakdownCOPConfig):
            self._bd_config = value
        else:
            self._bd_config = BreakdownCOPConfig(**(value or {}))

    @property
    def eq_config(self) -> EQConfig:
        """Equilibrium COP settings."""
        return self._eq_config

    @eq_config.setter
    def eq_config(self, value: EQConfig | dict | None = None):
        """Equilibrium COP settings."""
        if isinstance(value, EQConfig):
            self._eq_config = value
        else:
            self._eq_config = EQConfig(**(value or {}))

    def take_snapshot(
        self,
        name: str,
        eq: MHDState,
        coilset: CoilSet,
        problem: CoilsetOptimisationProblem,
        profiles: Profile | None = None,
        iterator: PicardIterator | None = None,
    ):
        """Take a snapshot of the pulse."""
        if name in self.snapshots:
            bluemira_warn(f"Over-writing snapshot {name}!")

        self.snapshots[name] = Snapshot(
            eq, coilset, problem, profiles, iterator=iterator, limiter=self.limiter
        )

    def _get_psi_premag(self):
        if bd_snap := self.snapshots.get(self.BREAKDOWN):
            return 2 * np.pi * bd_snap.eq.breakdown_psi
        bluemira_warn("Premagnetisation not calculated")
        return np.inf

    def run_premagnetisation(self):
        """Run the breakdown optimisation problem.

        Raises
        ------
        EquilibriaError
            Unable to relax breakdown for given coil sizes
        """
        strategy = self.bd_config.strategy(
            self.params.R_0.value, self.params.A.value, self.params.tk_sol_ib.value
        )

        relaxed = all(self.coilset.get_control_coils()._flag_sizefix)
        for i in range(self.bd_config.iter_max):
            coilset = deepcopy(self.coilset)
            breakdown = Breakdown(coilset, self.grid)
            constraints = deepcopy(self._coil_cons)

            cc = coilset.get_control_coils()

            if relaxed:
                max_currents = cc.get_max_current(0)
            else:
                max_currents = cc.get_max_current(self.params.I_p.value)
                cc.current = max_currents
                cc.discretisation = self.eq_config.coil_mesh_size

            problem = self.bd_config.make_opt_problem(
                breakdown,
                strategy,
                max_currents=max_currents,
                constraints=constraints,
                B_stray_max=self.params.B_premag_stray_max.value,
            )
            result = problem.optimise(fixed_coils=False)
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

        self.take_snapshot(self.BREAKDOWN, breakdown, result.coilset, problem)

        bluemira_print(f"Premagnetisation flux = {self._get_psi_premag():.2f} V.s")

    def run_reference_equilibrium(self):
        """Run a reference equilibrium."""
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
            gamma=self.eq_config.gamma,
        )
        program = PicardIterator(
            eq,
            opt_problem,
            convergence=deepcopy(self.eq_config.convergence),
            relaxation=self.eq_config.relaxation,
            fixed_coils=True,
            diagnostic_plotting=self.eq_config.diagnostic_plotting,
        )
        program()

        opt_problem = self.eq_config.make_opt_problem(
            eq,
            self._get_max_currents(eq.coilset),
            current_constraints=None,
            eq_constraints=[
                deepcopy(con)
                for con in self.eq_constraints
                if not isinstance(con, PsiConstraint | PsiBoundaryConstraint)
            ],
        )

        program = PicardIterator(
            eq,
            opt_problem,
            convergence=deepcopy(self.eq_config.convergence),
            relaxation=self.eq_config.relaxation,
            fixed_coils=True,
            diagnostic_plotting=self.eq_config.diagnostic_plotting,
        )
        program()

        self.take_snapshot(
            self.EQ_REF, eq, coilset, opt_problem, self.profiles, iterator=program
        )

    def calculate_sof_eof_fluxes(
        self, psi_premag: float | None = None
    ) -> tuple[float, float]:
        """Calculate the SOF and EOF plasma boundary fluxes.

        Returns
        -------
        :
            SOF psi
        :
            EOF psi
        """
        if psi_premag is None and self.BREAKDOWN not in self.snapshots:
            self.run_premagnetisation()
        elif psi_premag is not None:
            psi_premag *= 2 * np.pi

        psi_sof = calc_psib(
            psi_premag or self._get_psi_premag(),
            self.params.R_0.value,
            self.params.I_p.value,
            self.params.l_i.value,
            self.params.C_Ejima.value,
        )
        psi_eof = psi_sof - self.params.tau_flattop.value * self.params.v_burn.value
        return psi_sof, psi_eof

    def _get_max_currents(self, coilset: CoilSet) -> npt.NDArray[np.float64]:
        cc = coilset.get_control_coils()
        return cc.get_max_current(
            self.eq_config.peak_PF_current_factor * self.params.I_p.value
        )

    def get_sof_eof_opt_problems(
        self, psi_sof: float, psi_eof: float
    ) -> list[CoilsetOptimisationProblem]:
        """
        Returns
        -------
        :
            Start of flat top and end of flat top optimisation problems.
        """
        eq_ref = self.snapshots[self.EQ_REF].eq
        max_currents_pf = self._get_max_currents(self.coilset.get_coiltype("PF"))
        max_currents = self._get_max_currents(self.coilset)

        opt_problems = []
        for psi_boundary in [psi_sof, psi_eof]:
            eq = deepcopy(eq_ref)
            eq.coilset.get_coiltype("PF").resize(max_currents_pf)

            current_constraints = []
            if self._current_opt_cons:
                current_constraints += deepcopy(self._current_opt_cons)
            if self._coil_cons:
                current_constraints += deepcopy(self._coil_cons)

            eq_constraints = deepcopy(self.eq_constraints)
            for constraint in (*eq_constraints, *current_constraints):
                if isinstance(constraint, PsiBoundaryConstraint | PsiConstraint):
                    constraint.target_value = psi_boundary / (2 * np.pi)

            opt_problems.append(
                self.eq_config.make_opt_problem(
                    eq, max_currents, current_constraints, eq_constraints
                )
            )

        return opt_problems

    def converge_equilibrium(self, eq: Equilibrium, problem: CoilsetOptimisationProblem):
        """Converge an equilibrium problem from a 'frozen' plasma optimised state.

        Returns
        -------
        :
            The iterator
        """
        program = PicardIterator(
            eq,
            problem,
            fixed_coils=True,
            convergence=deepcopy(self.eq_config.convergence),
            relaxation=self.eq_config.relaxation,
            diagnostic_plotting=self.eq_config.diagnostic_plotting,
        )
        program()
        return program

    def converge_and_snapshot(
        self,
        sub_opt_problems: Iterable[CoilsetOptimisationProblem],
        problem_names: Iterable[str] = (SOF, EOF),
    ):
        """Converge equilibrium optimisation problems and take snapshots."""
        for snap, problem in zip(problem_names, sub_opt_problems, strict=False):
            eq = problem.eq
            program = self.converge_equilibrium(eq, problem)
            self.take_snapshot(
                snap, eq, eq.coilset, problem, eq.profiles, iterator=program
            )

    def plot(self):
        """Plot the pulsed equilibrium problem.

        Returns
        -------
        :
            plot figure
        """
        n_snapshots = len(self.snapshots)
        if n_snapshots == 0:
            return None

        f, ax = plt.subplots(1, n_snapshots)
        for i, (k, snap) in enumerate(self.snapshots.items()):
            axi = ax[i]
            snap.eq.plot(ax=axi)
            snap.coilset.plot(ax=axi)
            if k == self.BREAKDOWN:
                psi_name, psi_val = r"$\Psi_{bd}$", snap.eq.breakdown_psi
            else:
                psi_name, psi_val = r"$\Psi_{b}$", snap.eq.get_OX_points()[1][0].psi

            axi.set_title(f"{k} {psi_name}: {2 * np.pi * psi_val:.2f} V.s")
        return f


class FixedPulsedCoilsetDesign(PulsedCoilsetDesign):
    """Procedural design for a pulsed tokamak with a known, fixed PF coilset."""

    def optimise(self) -> CoilSet:
        """Run pulsed coilset design optimisation."""  # noqa: DOC201
        self.optimise_currents()
        return self.coilset

    def optimise_currents(self):
        """Optimise the coil currents at the start and end of the current flat-top."""
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
    breakdown_settings:
        Breakdown optimiser settings
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

    def __init__(
        self,
        params: ParameterFrame,
        coilset: CoilSet,
        position_mapper: PositionMapper,
        grid: Grid,
        equilibrium_constraints: list[MagneticConstraint],
        profiles: Profile,
        breakdown_settings: dict | BreakdownCOPConfig | None = None,
        equilibrium_settings: dict | EQConfig | None = None,
        current_opt_constraints: list[UpdateableConstraint] | None = None,
        coil_constraints: list[UpdateableConstraint] | None = None,
        limiter: Limiter | None = None,
        position_settings: dict | PositionConfig | None = None,
    ):
        super().__init__(
            params,
            coilset,
            grid,
            equilibrium_constraints,
            profiles,
            breakdown_settings,
            equilibrium_settings,
            current_opt_constraints,
            coil_constraints,
            limiter,
        )
        self.coilset = self._prepare_coilset(self.coilset)
        self.position_mapper = position_mapper
        self.pos_config = position_settings

    @property
    def pos_config(self) -> PositionConfig:
        """Position COP settings."""
        return self._pos_config

    @pos_config.setter
    def pos_config(self, value: PositionConfig | dict | None = None):
        """Position COP settings."""
        if isinstance(value, PositionConfig):
            self._pos_config = value
        else:
            self._pos_config = PositionConfig(**(value or {}))

    def _prepare_coilset(self, coilset: CoilSet) -> CoilSet:
        coilset = deepcopy(coilset)
        coilset.discretisation = np.where(
            coilset._flag_sizefix,
            self.eq_config.coil_mesh_size,
            coilset.discretisation,
        )
        return coilset

    def optimise(self, *, verbose: bool = False) -> CoilSet:
        """
        Optimise the coil positions for the start and end of the current flat-top.
        """  # noqa: DOC201
        psi_sof, psi_eof = self.calculate_sof_eof_fluxes()
        if self.EQ_REF not in self.snapshots:
            self.run_reference_equilibrium()

        sub_opt_problems = self.get_sof_eof_opt_problems(psi_sof, psi_eof)
        pos_opt_problem = self.pos_config.make_opt_problem(
            self.position_mapper, sub_opt_problems
        )
        result = pos_opt_problem.optimise(verbose=verbose)
        optimised_coilset = self._consolidate_coilset(result.coilset, sub_opt_problems)

        self.converge_and_snapshot(sub_opt_problems)

        # Re-run breakdown
        psi_bd_orig = self._get_psi_premag()
        self.coilset = optimised_coilset
        self.run_premagnetisation()
        if (psi_premag := self._get_psi_premag()) < psi_bd_orig - 2.0:
            bluemira_warn(
                "Breakdown flux significantly lower with optimised coil positions: "
                f"{psi_premag:.2f} < {psi_bd_orig:.2f}"
            )
        return optimised_coilset

    def _consolidate_coilset(
        self, coilset: CoilSet, sub_opt_problems: Iterable[CoilsetOptimisationProblem]
    ) -> CoilSet:
        """
        Set the current bounds on the current optimisation problems, fix coil sizes, and
        mesh.
        """  # noqa: DOC201
        max_cs_currents = coilset.get_coiltype("CS").get_max_current(0.0)
        pf_currents = []
        for problem in sub_opt_problems:
            pf_coils = problem.eq.coilset.get_coiltype("PF").get_control_coils()
            pf_currents.append(np.abs(pf_coils.current))

        max_pf_currents = np.max(pf_currents, axis=0)
        # Relax the max currents a bit to avoid oscillation
        max_pf_current = self.eq_config.peak_PF_current_factor * self.params.I_p.value
        max_pf_currents = np.clip(1.1 * max_pf_currents, 0, max_pf_current)

        for problem in sub_opt_problems:
            pf_coils = problem.eq.coilset.get_coiltype("PF").get_control_coils()
            pf_coils.resize(max_pf_currents)
            pf_coils.fix_sizes()
            pf_coils.discretisation = self.eq_config.coil_mesh_size
            problem.set_current_bounds(
                np.concatenate([max_pf_currents, max_cs_currents])
            )

        consolidated_coilset = deepcopy(problem.eq.coilset)
        consolidated_coilset.fix_sizes()
        consolidated_coilset.get_control_coils().current = 0
        return consolidated_coilset

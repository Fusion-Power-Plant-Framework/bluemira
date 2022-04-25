# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Interface for building and loading equilibria and coilset designs
"""

from copy import deepcopy
from typing import List, Type

import numpy as np

from bluemira.base.look_and_feel import bluemira_print
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Breakdown
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_constraints import (
    coil_field_constraints,
    coil_force_constraints,
)
from bluemira.equilibria.opt_problems import (
    BreakdownZoneStrategy,
    OutboardBreakdownZoneStrategy,
    PremagnetisationCOP,
)
from bluemira.equilibria.physics import calc_psib
from bluemira.equilibria.profiles import Profile
from bluemira.utilities.opt_problems import OptimisationConstraint
from bluemira.utilities.optimiser import Optimiser


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
        self.eq = deepcopy(eq)
        self.coilset = deepcopy(coilset)
        if constraints is not None:
            self.constraints = deepcopy(constraints)
        else:
            self.constraints = None
        if profiles is not None:
            self.profiles = deepcopy(profiles)
        else:
            self.profiles = None
        if limiter is not None:
            self.limiter = deepcopy(limiter)
        else:
            self.limiter = None
        if optimiser is not None:
            self.optimiser = deepcopy(optimiser)
        else:
            self.optimiser = None
        self.tf = tfcoil


class PulsedEquilibriumProblem:
    """
    Procedural design for a pulsed tokamak.
    """

    def __init__(
        self,
        params,
        coilset: CoilSet,
        grid: Grid,
        coil_constraints: List[callable],
        profiles: Profile,
        magnetic_targets,
        breakdown_strategy_cls: Type[BreakdownZoneStrategy],
        breakdown_problem_cls: Type[PremagnetisationCOP],
        breakdown_optimiser: Optimiser = Optimiser(
            "COBYLA", opt_conditions={"max_eval": 5000, "ftol_rel": 1e-10}
        ),
        breakdown_settings: dict = {"B_stray_con_tol": 1e-8, "n_B_stray_points": 20},
        # equilibrium_problem_cls: type[],
        # equilibrium_optimiser: Optimiser,
        # equilibrium_settings: dict,
    ):
        self.params = params
        self.coilset = coilset
        self.grid = grid
        self.profiles = profiles
        self.eq_targets = magnetic_targets
        self.snapshots = {}

        self._bd_strat_cls = breakdown_strategy_cls
        self._bd_prob_cls = breakdown_problem_cls
        self._bd_settings = breakdown_settings
        self._bd_opt = breakdown_optimiser

        # self._eq_prob_cls = equilibrium_problem_cls
        # self._eq_opt = equilibrium_optimiser
        # self._eq_settings = equilibrium_settings
        self._coil_cons = coil_constraints

    def run_premagnetisation(self):
        R_0 = self.params.R_0.value
        strategy = self._bd_strat_cls(
            R_0, self.params.A.value, self.params.tk_sol_ib.value
        )
        coilset = deepcopy(self.coilset)

        def iterate_once(coilset):
            coilset.mesh_coils(0.1)
            breakdown = Breakdown(coilset, self.grid, R_0=R_0)

            constraints = deepcopy(self._coil_cons)
            for constraint in constraints:
                constraint._args["eq"] = breakdown

            # TODO: Ip is in MA already
            max_currents = self.coilset.get_max_currents(
                1.0 * 1e6 * self.params.I_p.value
            )
            problem = self._bd_prob_cls(
                coilset,
                strategy,
                B_stray_max=self.params.B_premag_stray_max.value,
                B_stray_con_tol=self._bd_settings["B_stray_con_tol"],
                n_B_stray_points=self._bd_settings["n_B_stray_points"],
                optimiser=self._bd_opt,
                max_currents=max_currents,
                constraints=constraints,
            )
            coilset = problem.optimise(max_currents / 1e6)
            breakdown = Breakdown(coilset, self.grid, R_0=R_0)
            breakdown.set_breakdown_point(*strategy.breakdown_point)
            psi_premag = breakdown.breakdown_psi
            bluemira_print(f"Premagnetisation flux = {2*np.pi * psi_premag:.2f} V.s")
            return breakdown, psi_premag

        breakdown, psi_1 = iterate_once(coilset)
        breakdown, psi_2 = iterate_once(coilset)
        while not np.isclose(psi_2, psi_1, rtol=1e-2):
            psi_2 = psi_1
            breakdown, psi_1 = iterate_once(coilset)

        return breakdown

    def calculate_sof_eof_fluxes(self, psi_premag: float):
        """
        Calculate the SOF and EOF plasma boundary fluxes.
        """
        psi_sof = calc_psib(
            2 * np.pi * psi_premag,
            self.params.R_0.value,
            1e6 * self.params.I_p.value,
            self.params.l_i.value,
            self.params.C_Ejima.value,
        )
        psi_eof = psi_sof - self.params.tau_flattop.value * self.params.v_burn.value
        return psi_sof, psi_eof


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from bluemira.base.config import Configuration
    from bluemira.builders.EUDEMO.pf_coils import make_coilset
    from bluemira.geometry.parameterisations import PrincetonD

    params = Configuration()

    tf_boundary = PrincetonD({"x1": {"value": 4}, "x2": {"value": 16}}).create_shape()

    tk_cs = 0.5 * params.tk_cs.value
    r_cs = params.r_cs_in.value + tk_cs
    coilset = make_coilset(
        tf_boundary,
        params.R_0.value,
        params.kappa.value,
        params.delta.value,
        r_cs=r_cs,
        tk_cs=tk_cs,
        g_cs=params.g_cs_mod.value,
        n_CS=params.n_CS.value,
        n_PF=params.n_PF.value,
        tk_cs_ins=params.tk_cs_insulation.value,
        tk_cs_cas=params.tk_cs_casing.value,
        PF_jmax=params.PF_jmax.value,
        PF_bmax=params.PF_bmax.value,
        CS_jmax=params.CS_jmax.value,
        CS_bmax=params.CS_bmax.value,
    )

    grid = Grid(0.1, 20, -10, 10, 100, 100)

    constraints = [
        OptimisationConstraint(
            coil_field_constraints,
            f_constraint_args={
                "eq": None,
                "B_max": coilset.get_max_fields(),
                "scale": 1e6,
            },
            tolerance=1e-6 * np.ones(11),
        )
    ]

    problem = PulsedEquilibriumProblem(
        params,
        coilset,
        grid,
        constraints,
        None,
        None,
        OutboardBreakdownZoneStrategy,
        PremagnetisationCOP,
    )

    bd = problem.run_premagnetisation()

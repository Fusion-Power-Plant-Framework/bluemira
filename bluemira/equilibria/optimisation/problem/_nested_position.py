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
OptimisationProblems for coilset design.

New optimisation schemes for the coilset can be provided by subclassing
from CoilsetOP, which is an abstract base class for OptimisationProblems
that use a coilset as their parameterisation object.

Subclasses must provide an optimise() method that returns an optimised
coilset according to a given optimisation objective function.
As the exact form of the state vector that is optimised is often
specific to each objective function, each subclass of CoilsetOP is
generally also specific to a given objective function, since
the method used to map the coilset object to the state vector
(and additional required arguments) will generally differ in each case.

"""

from typing import Dict

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import MagneticConstraintSet
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
)
from bluemira.equilibria.positioner import RegionMapper
from bluemira.geometry.coordinates import Coordinates
from bluemira.optimisation import optimise


class NestedCoilsetPositionCOP(CoilsetOptimisationProblem):
    """
    Coilset OptimisationProblem for coil currents and positions
    subject to maximum current bounds and positions bounded within
    a provided region. Performs a nested optimisation for coil
    currents within each position optimisation function call.

    Parameters
    ----------
    sub_opt:
        Coilset OptimisationProblem to use for the optimisation of
        coil currents at each trial set of coil positions.
        sub_opt.coilset must exist, and will be modified
        during the optimisation.
    eq:
        Equilibrium object used to update magnetic field targets.
    targets:
        Set of magnetic field targets to use in objective function.
    pfregions:
        Dictionary of Coordinates that specify convex hull regions inside which
        each PF control coil position is to be optimised.
        The Coordinates must be 2d in x,z in units of [m].
    opt_algorithm:
        The optimisation algorithm to use (e.g. SLSQP)
    opt_conditions:
        The stopping conditions for the optimiser.

    Notes
    -----
    Setting stopval and maxeval is the most reliable way to stop optimisation
    at the desired figure of merit and number of iterations respectively.
    Some NLOpt optimisers display unexpected behaviour when setting xtol and
    ftol, and may not terminate as expected when those criteria are reached.
    """

    def __init__(
        self,
        sub_opt: CoilsetOptimisationProblem,
        eq: Equilibrium,
        targets: MagneticConstraintSet,
        pfregions: Dict[str, Coordinates],
        opt_algorithm="SBPLX",
        opt_conditions={
            "stop_val": 1.0,
            "max_eval": 100,
        },
    ):
        self.region_mapper = RegionMapper(pfregions)
        self.eq = eq
        self.targets = targets
        _, lower_bounds, upper_bounds = self.region_mapper.get_Lmap(self.coilset)
        self.bounds = (lower_bounds, upper_bounds)
        self.coilset = sub_opt.coilset
        self.sub_opt = sub_opt
        self.opt_algorithm = opt_algorithm
        self.opt_conditions = opt_conditions

        self.initial_state, self.substates = self.read_coilset_state(
            self.coilset, self.scale
        )
        self.I0 = np.array_split(self.initial_state, self.substates)[2]

    def optimise(self):
        """
        Run the optimisation.

        Returns
        -------
        Optimised CoilSet object.
        """
        # Get initial currents, and trim to within current bounds.
        initial_state, substates = self.read_coilset_state(self.coilset, self.scale)
        _, _, initial_currents = np.array_split(initial_state, substates)
        initial_mapped_positions, _, _ = self.region_mapper.get_Lmap(self.coilset)

        # TODO: find more explicit way of passing this to objective?
        self.I0 = initial_currents

        opt_result = optimise(
            f_objective=self.objective,
            x0=initial_mapped_positions,
            algorithm=self.opt_algorithm,
            opt_conditions=self.opt_conditions,
        )
        self.set_coilset_state(self.coilset, opt_result.x, self.scale)
        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)

    def objective(self, vector: npt.NDArray) -> float:
        """Objective function to minimise."""
        self.region_mapper.set_Lmap(vector)
        x_vals, z_vals = self.region_mapper.get_xz_arrays()
        positions = np.concatenate((x_vals, z_vals))
        coilset_state = np.concatenate((positions, self.I0))
        self.set_coilset_state(self.coilset, coilset_state, self.scale)

        # Update targets
        self.eq._remap_greens()
        self.targets(self.eq, I_not_dI=True, fixed_coils=False)

        # Run the sub-optimisation
        sub_opt_result = self.sub_opt.optimise()
        return sub_opt_result.f_x

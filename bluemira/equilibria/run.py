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
        coilset,
        grid,
        profiles,
        magnetic_targets,
        equilibrium_problem,
    ):
        self.params = params
        self.coilset = coilset
        self.grid = grid
        self.profiles = profiles

    def run_premagnetisation(self, breakdown_strategy, breakdown_problem):
        pass

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
import abc
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.opt_constraints import MagneticConstraintSet
from bluemira.optimisation import OptimisationProblem
from bluemira.optimisation.problem import OptimisationProblemBase


@dataclass
class CoilSetOptimisationResult:
    coilset: CoilSet
    """The optimised :class:`.CoilSet`."""


class CoilSetOptimisationProblem(abc.ABC):
    @abc.abstractproperty
    def coilset(self) -> CoilSet:
        """The :class:`.CoilSet` being optimised."""

    def get_current_bounds(
        self, max_currents: npt.ArrayLike, current_scale: float
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Gets the scaled current vector bounds. Must be called prior to optimise.

        Parameters
        ----------
        max_currents:
            Maximum magnitude of currents in each coil [A] permitted during optimisation.
            If max_current is supplied as a float, the float will be set as the
            maximum allowed current magnitude for all coils.
            If the coils have current density limits that are more restrictive than these
            coil currents, the smaller current limit of the two will be used for each
            coil.
        current_scale:
            Factor to scale coilset currents down when returning scaled current limits.

        Returns
        -------
        Tuple of arrays containing lower and upper bounds for currents
        permitted in each control coil.
        """
        coilset = self.coilset
        n_control_currents = len(coilset.current[coilset._control_ind])
        scaled_input_current_limits = np.full(n_control_currents, np.inf)

        if max_currents is not None:
            input_current_limits = np.asarray(max_currents)
            input_size = np.size(np.asarray(input_current_limits))
            if input_size == 1 or input_size == n_control_currents:
                scaled_input_current_limits = input_current_limits / current_scale
            else:
                raise EquilibriaError(
                    "Length of max_currents array provided to optimiser is not"
                    "equal to the number of control currents present."
                )

        # Get the current limits from coil current densities
        coilset_current_limits = np.full(n_control_currents, np.inf)
        coilset_current_limits[coilset._flag_sizefix] = coilset.get_max_current()[
            coilset._flag_sizefix
        ]

        # Limit the control current magnitude by the smaller of the two limits
        control_current_limits = np.minimum(
            scaled_input_current_limits, coilset_current_limits
        )
        return -control_current_limits, control_current_limits


class UnconstrainedTikhonovCurrentGradientCOP(CoilSetOptimisationProblem):
    """Optimise current gradient vector for minimal error to the L2-norm."""

    def __init__(
        self,
        coilset: CoilSet,
        eq: Equilibrium,
        targets: MagneticConstraintSet,
        gamma: float,
    ) -> None:
        self.eq = eq
        self._coilset = coilset
        self.targets = targets
        self.gamma = gamma

    @property
    def coilset(self) -> CoilSet:
        return self._coilset

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return super().bounds()

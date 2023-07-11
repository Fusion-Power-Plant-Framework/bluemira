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
import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.error import EquilibriaError


def bounds_of_currents(coilset: CoilSet, max_currents: npt.ArrayLike) -> npt.NDArray:
    """Calculate the bounds on the currents in the coils."""
    n_control_currents = len(coilset.current[coilset._control_ind])
    scaled_input_current_limits = np.inf * np.ones(n_control_currents)

    if max_currents is not None:
        input_current_limits = np.asarray(max_currents)
        input_size = np.size(np.asarray(input_current_limits))
        if input_size == 1 or input_size == n_control_currents:
            scaled_input_current_limits = input_current_limits
        else:
            raise EquilibriaError(
                "Length of max_currents array provided to optimiser is not"
                "equal to the number of control currents present."
            )

    # Get the current limits from coil current densities
    coilset_current_limits = np.infty * np.ones(n_control_currents)
    coilset_current_limits[coilset._flag_sizefix] = coilset.get_max_current()[
        coilset._flag_sizefix
    ]

    # Limit the control current magnitude by the smaller of the two limits
    control_current_limits = np.minimum(
        scaled_input_current_limits, coilset_current_limits
    )
    return control_current_limits

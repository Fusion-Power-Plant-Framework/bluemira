# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Harmonics constraint classes
"""

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.optimisation.constraints import ConstraintFunction


class SphericalHarmonicConstraintFunction(ConstraintFunction):
    """
    Constraint function to constrain spherical harmonics starting from initial
    coil currents and associated core plasma.

    Parameters
    ----------
    a_mat:
        Response matrix
    b_vec:
        Target value vector
    value:
        Target constraint value
    scale:
        Current scale with which to calculate the constraints
    """

    def __init__(
        self,
        a_mat: np.ndarray,
        b_vec: np.ndarray,
        value: float,
        scale: float,
        name: str | None = None,
    ) -> None:
        self.a_mat = a_mat
        self.b_vec = b_vec
        self.value = value
        self.scale = scale
        self.name = name

    def f_constraint(self, vector: npt.NDArray) -> npt.NDArray:
        """Constraint function"""
        currents = self.scale * vector

        result = self.a_mat[:,] @ currents
        return result - self.b_vec - self.value

    def df_constraint(self, _vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Constraint derivative"""
        return self.scale * self.a_mat

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

from bluemira.base.look_and_feel import bluemira_print
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
        cur_repetition_mat: np.ndarray,
        debug=False,
    ) -> None:
        self.a_mat = a_mat
        self.b_vec = b_vec
        self.value = value
        self.scale = scale
        self.cur_repetition_mat = cur_repetition_mat
        self.debug = debug

    def f_constraint(self, vector: npt.NDArray) -> npt.NDArray:
        """Constraint function"""
        currents = self.cur_repetition_mat @ (self.scale * vector)

        result = self.a_mat[1:,] @ currents
        residual = result - self.b_vec - self.value

        if self.debug:
            bluemira_print(f"""
            refs: {self.b_vec}
            currents: {currents}
            currents_sum: {np.sum(currents)}
            result {result}
            residual: {residual}
            """)
        return residual

    def df_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:  # noqa: ARG002
        """Constraint derivative"""
        return (self.scale * self.a_mat[1:,]) @ self.cur_repetition_mat

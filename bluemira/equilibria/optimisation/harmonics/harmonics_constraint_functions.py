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


class HarmonicConstraintFunction(ConstraintFunction):
    """
    Constraint function to constrain harmonics starting from initial
    coil currents and associated core plasma.
    Used for spherical and toroidal harmonics.
    # FIXME

    Parameters
    ----------
    a_mat_cos:
        Cos response matrix
    a_mat_sin:
        Sin response matrix
    b_vec_cos:
        Target value cos vector
    b_vec_sin:
        Target value sin vector
    value:
        Target constraint value
    scale:
        Current scale with which to calculate the constraints
    """

    def __init__(
        self,
        a_mat_cos: np.ndarray,
        a_mat_sin: np.ndarray,
        b_vec_cos: np.ndarray,
        b_vec_sin: np.ndarray,
        value: float,
        scale: float,
        name: str | None = None,
    ) -> None:
        self.a_mat_cos = a_mat_cos
        self.a_mat_sin = a_mat_sin
        self.b_vec_cos = b_vec_cos
        self.b_vec_sin = b_vec_sin
        self.value = value
        self.scale = scale
        self.name = name

    def f_constraint(self, vector: npt.NDArray) -> npt.NDArray:
        """Constraint function"""  # noqa: DOC201
        currents = self.scale * vector

        result_cos = self.a_mat_cos @ currents
        result_sin = self.a_mat_sin @ currents

        result_cos -= self.b_vec_cos + self.value
        result_sin -= self.b_vec_sin + self.value
        return np.append(result_cos, result_sin, axis=0)

    def df_constraint(self, vector: npt.NDArray) -> npt.NDArray:
        scaled_cos = self.a_mat_cos * self.scale
        scaled_sin = self.a_mat_sin * self.scale
        return np.append(scaled_cos, scaled_sin, axis=0)

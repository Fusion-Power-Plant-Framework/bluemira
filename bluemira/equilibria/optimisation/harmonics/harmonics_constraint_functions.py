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
        """Constraint function"""  # noqa: DOC201
        currents = self.scale * vector

        result = self.a_mat[:,] @ currents
        return result - self.b_vec - self.value

    def df_constraint(self, _vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Constraint derivative"""  # noqa: DOC201
        return self.scale * self.a_mat


class ToroidalHarmonicConstraintFunction(ConstraintFunction):
    """
    Constraint function to constrain harmonics starting from initial
    coil currents and associated core plasma.
    Used for toroidal harmonics.

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

        self.cos_empty = len(b_vec_cos) == 0
        self.sin_empty = len(b_vec_sin) == 0

    def f_constraint(self, vector: npt.NDArray) -> npt.NDArray:
        """Constraint function"""  # noqa: DOC201
        currents = self.scale * vector

        if self.constraint_type == "inequality":
            currents = np.tile(currents, 2)

        if self.cos_empty:
            result_cos = []
        else:
            result_cos = self.a_mat_cos @ currents - self.b_vec_cos

        if self.sin_empty:
            result_sin = []
        else:
            result_sin = self.a_mat_sin @ currents - self.b_vec_sin


        return np.append(result_cos, result_sin, axis=0)
        

    def df_constraint(self, vector: npt.NDArray) -> npt.NDArray:  # noqa: ARG002
        """Constraint derivative"""  # noqa: DOC201
        if self.cos_empty:
            return self.a_mat_sin * self.scale
        if self.sin_empty:
            return self.a_mat_cos * self.scale
        return np.append(
            self.a_mat_cos * self.scale, self.a_mat_sin * self.scale, axis=0
        )

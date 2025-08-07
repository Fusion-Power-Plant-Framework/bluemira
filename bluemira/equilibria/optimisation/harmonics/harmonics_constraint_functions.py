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
        # print("In __init__ of ToroidalHarmonicConstraintFunction")
        # print(f"a_mat_cos = {self.a_mat_cos}")
        # print(f"a_mat_sin = {self.a_mat_sin}")
        # print(f"b_vec_cos = {self.b_vec_cos}")
        # print(f"b_vec_sin = {self.b_vec_sin}")
        # print(f"value = {self.value}")
        # print(f"scale = {self.scale}")
        # print(f"name = {self.name}")

    def f_constraint(self, vector: npt.NDArray) -> npt.NDArray:
        """Constraint function"""  # noqa: DOC201
        currents = self.scale * vector
        # currents = np.array([
        #     6124296.34922996,
        #     -4023488.65084202,
        #     -6999062.3598625,
        #     -10893402.00527329,
        #     -3223584.36167449,
        #     25975901.95919058,
        #     14169552.65907129,
        #     -18605840.14157257,
        #     -33974724.07681966,
        #     -19160081.37992439,
        #     -32764488.39904289,
        # ])
        if self.cos_empty:
            result_cos = []
        else:
            result_cos = self.a_mat_cos @ currents
            result_cos -= self.b_vec_cos + self.value

        if self.sin_empty:
            result_sin = []
        else:
            result_sin = self.a_mat_sin @ currents
            result_sin -= self.b_vec_sin + self.value

        # print("In f_constraint of ToroidalHarmonicConstraintFunction")
        # print(f"a_mat_cos = {self.a_mat_cos}")
        # print(f"a_mat_sin = {self.a_mat_sin}")
        # print(f"b_vec_cos = {self.b_vec_cos}")
        # print(f"b_vec_sin = {self.b_vec_sin}")
        # print(f"value = {self.value}")
        # print(f"scale = {self.scale}")
        # print(f"name = {self.name}")
        # print(f"result_cos = {result_cos}")
        # print(f"result_sin = {result_sin}")

        # print(f"vector = {vector}")
        # print(f"currents = {currents}")

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

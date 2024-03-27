# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Equilibrium optimisation objective functions.

Objective functions must be of the form:

.. code-block:: python

    class Objective(ObjectiveFunction):

        def f_objective(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return objective_calc(vector)

        def df_objective(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return gradient_calc(vector)

The objective function is minimised, so lower values are "better".

Note that the gradient of the objective function is of the form:

:math:`\\nabla f = \\bigg[\\dfrac{\\partial f}{\\partial x_0}, \\dfrac{\\partial f}{\\partial x_1}, ...\\bigg]`
"""  # noqa: W505, E501

import abc

import numpy as np
import numpy.typing as npt

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.error import EquilibriaError


class ObjectiveFunction(abc.ABC):
    """
    Base class for ObjectiveFunctions

    Notes
    -----
    Optionally the function 'df_objective' can be implemented on any child
    classes to calculate the gradient of the objective function.
    The function should take an `npt.NDArray` as its only argument and
    return only an `npt.NDArray`.
    If the `df_objective` function is not provided and the optimisation algorithm
    is gradient based the approximate derivate is calculated.
    """

    @abc.abstractmethod
    def f_objective(self, vector: npt.NDArray[np.float64]) -> float:
        """Objective function for an optimisation."""


class RegularisedLsqObjective(ObjectiveFunction):
    """
    Least-squares objective with Tikhonov regularisation term.

    Parameters
    ----------
    scale:
        Scaling factor for the vector
    a_mat:
        The 2-D a_mat control matrix A (n, m)
    b_vec:
        The 1-D b vector of target values (n)
    gamma:
        The Tikhonov regularisation parameter.
    """

    def __init__(
        self,
        scale: float,
        a_mat: npt.NDArray[np.float64],
        b_vec: npt.NDArray[np.float64],
        gamma: float,
        currents_rep_mat: Optional[npt.NDArray] = None,
    ) -> None:
        self.scale = scale
        self.a_mat = a_mat if currents_rep_mat is None else a_mat @ currents_rep_mat
        self.b_vec = b_vec
        self.gamma = gamma

    def f_objective(self, vector: npt.NDArray[np.float64]) -> float:
        """Objective function for an optimisation."""
        vector = vector * self.scale
        fom, _ = regularised_lsq_fom(vector, self.a_mat, self.b_vec, self.gamma)
        if fom <= 0:
            raise EquilibriaError(
                "Optimiser least-squares objective function less than zero or nan."
            )
        return fom

    def df_objective(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gradient of the objective function for an optimisation."""
        vector = vector * self.scale
        jac = 2 * self.a_mat.T @ self.a_mat @ vector / float(len(self.b_vec))
        jac -= 2 * self.a_mat.T @ self.b_vec / float(len(self.b_vec))
        jac += 2 * self.gamma * self.gamma * vector
        return self.scale * jac


class CoilCurrentsObjective(ObjectiveFunction):
    """Objective function for the minimisation of the sum of coil currents squared."""

    @staticmethod
    def f_objective(vector: npt.NDArray[np.float64]) -> float:
        """Objective function for an optimisation."""
        return np.sum(np.square(vector))

    @staticmethod
    def df_objective(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gradient of the objective function for an optimisation."""
        return 2 * vector


class MaximiseFluxObjective(ObjectiveFunction):
    """
    Objective function to maximise flux

    Parameters
    ----------
    c_psi_mat:
        Response matrix of the coil psi contributions to the point at which the flux
        should be maximised
    scale:
        Scaling factor for the vector
    """

    def __init__(self, c_psi_mat: npt.NDArray[np.float64], scale: float):
        self.c_psi_mat = c_psi_mat
        self.scale = scale

    def f_objective(self, vector: npt.NDArray[np.float64]) -> float:
        """Objective function for an optimisation."""
        return -self.scale * self.c_psi_mat @ vector

    def df_objective(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:  # noqa: ARG002
        """Gradient of the objective function for an optimisation."""
        return -self.scale * self.c_psi_mat


# =============================================================================
# Figures of merit
# =============================================================================


def tikhonov(
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    gamma: float,
    currents_rep_mat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Tikhonov regularisation of Ax-b problem.

    \t:math:`\\textrm{minimise} || Ax - b ||^2 + ||{\\gamma} \\cdot x ||^2`\n
    \t:math:`x = (A^T A + {\\gamma}^2 I)^{-1}A^T b`

    Parameters
    ----------
    a_mat:
        The 2-D A matrix of responses
    b_vec:
        The 1-D b vector of values
    gamma: float
        The Tikhonov regularisation parameter

    Returns
    -------
    x:
        The result vector
    """
    if currents_rep_mat is not None:
        a_mat = a_mat @ currents_rep_mat
    try:
        return np.dot(
            np.linalg.inv(np.dot(a_mat.T, a_mat) + gamma**2 * np.eye(a_mat.shape[1])),
            np.dot(a_mat.T, b_vec),
        )
    except np.linalg.LinAlgError:
        bluemira_warn("Tikhonov singular matrix..!")
        return np.dot(
            np.linalg.pinv(np.dot(a_mat.T, a_mat) + gamma**2 * np.eye(a_mat.shape[1])),
            np.dot(a_mat.T, b_vec),
        )


def regularised_lsq_fom(
    x: np.ndarray, a_mat: np.ndarray, b_vec: np.ndarray, gamma: float
) -> tuple[float, np.ndarray]:
    """
    Figure of merit for the least squares problem Ax = b, with
    Tikhonov regularisation term. Normalised for the number of
    targets.

    ||(Ax - b)||²/ len(b)] + ||Γx||²

    Parameters
    ----------
    x :
        The 1-D x state vector (m)
    a_mat:
        The 2-D a_mat control matrix A (n, m)
    b_vec:
        The 1-D b vector of target values (n)
    gamma:
        The Tikhonov regularisation parameter.

    Returns
    -------
    fom:
        Figure of merit, explicitly given by
        ||(Ax - b)||²/ len(b)] + ||Γx||²
    residual:
        Residual vector (Ax - b)
    """
    residual = np.dot(a_mat, x) - b_vec
    number_of_targets = float(len(residual))
    fom = residual.T @ residual / number_of_targets + gamma * gamma * x.T @ x

    if fom <= 0:
        raise EquilibriaError("Least-squares objective function less than zero or nan.")
    return fom, residual

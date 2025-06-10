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
    ) -> None:
        self.scale = scale
        self.a_mat = a_mat
        self.b_vec = b_vec
        self.gamma = gamma

    def f_objective(self, vector: npt.NDArray[np.float64]) -> float:
        """Objective function for an optimisation.

        Returns
        -------
        :
            The figure of merit

        """
        vector = vector * self.scale  # nlopt read only  # noqa: PLR6104
        return tikhonov(vector, self.a_mat, self.b_vec, self.gamma)

    def df_objective(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gradient of the objective function for an optimisation."""  # noqa: DOC201
        vector = vector * self.scale  # nlopt read only  # noqa: PLR6104
        jac = 2 * self.a_mat.T @ self.a_mat @ vector / float(len(self.b_vec))
        jac -= 2 * self.a_mat.T @ self.b_vec / float(len(self.b_vec))
        jac += 2 * self.gamma * self.gamma * vector
        return self.scale * jac


class CoilCurrentsObjective(ObjectiveFunction):
    """Objective function for the minimisation of the sum of coil currents squared."""

    @staticmethod
    def f_objective(vector: npt.NDArray[np.float64]) -> float:
        """Objective function for an optimisation.

        Returns
        -------
        :
            The figure of merit
        """
        return np.sum(np.square(vector))

    @staticmethod
    def df_objective(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gradient of the objective function for an optimisation."""  # noqa: DOC201
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
        """Objective function for an optimisation."""  # noqa: DOC201
        return -self.scale * self.c_psi_mat @ vector

    def df_objective(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:  # noqa: ARG002
        """Gradient of the objective function for an optimisation."""  # noqa: DOC201
        return -self.scale * self.c_psi_mat


# =============================================================================
# Figures of merit
# =============================================================================


def ols(
    x: np.ndarray,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
):
    """
    Ordinary Least Squares (OLS)

    Can use as objective function to minimise the Residual Sum of Squares (RSS).

    \t:math:`\\textrm{minimise} \\fraction{1}{n} \\sum_{i=1}^{n} ( A_{i}x - b_{i} )^2`

    Also, written as:

    \t:math:`\\textrm{minimise} || Ax - b ||^2`

    where:
    - b is the target vector,
    - n is the number of targets,
    - Ax is the predicted target value, given known (control) matrix A
    and state vector x.

    Raises
    ------
    EquilibriaError
        Least squares result < 0 or NaN

    Returns
    -------
    :
        figure of merit
    """
    residual = np.dot(a_mat, x) - b_vec
    number_of_targets = float(len(residual))
    fom = residual.T @ residual / number_of_targets
    if fom <= 0:
        raise EquilibriaError("Least-squares objective function less than zero or nan.")
    return fom


def lasso(
    x: np.ndarray,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    gamma: np.ndarray,
):
    """
    LASSO (Least Absolute Shrinkage and Selection Operator)  a.k.a. L1 Regression

    Regularisation with the absolute value of magnitude of x as a penalty term
    and the strength of the penalty imposed set by gamma.
    The aim is to reduce some of the values to zero, in order to select only
    the important features and ignore the less important ones.

    Can use as objective function:

    \t:math:`\\textrm{rss} = || Ax - b ||^{2}`\n
    \t:math:`\\textrm{minimise} \\textrm{rss} + \\gamma \\sum_{j=1}^{p} | x_{j} |`

    where:
    - b is the target vector,
    - Ax is the predicted target value, given known (control) matrix A
    and state vector x,
    - gamma is regularization parameter that controls the penalty
    applied to the coefficients,
    - p is the number of predictor variables.

    Raises
    ------
    EquilibriaError
        Least squares result < 0 or NaN

    Returns
    -------
    :
        figure of merit
    """
    fom_ols = ols(x, a_mat, b_vec)
    fom_las = fom_ols + gamma * np.sum(np.abs(x))
    if fom_las <= 0:
        raise EquilibriaError("Least-squares objective function less than zero or nan.")
    return fom_las


def tikhonov(
    x: np.ndarray,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    gamma: np.ndarray,
):
    """
    Tikhonov a.k.a L2 Regression

    Regularisation with the squared magnitude of x as a penalty term
    and the strength of the penalty imposed set by gamma.
    The aim is to reduce the coefficient values (but not to zero as in LASSO).

    Can use as objective function:

    \t:math:`\\textrm{rss} = || Ax - b ||^{2}`\n
    \t:math:`\\textrm{minimise} \\textrm{rss} + || \\gamma x ||^{2}`

    where:
    - b is the target vector,
    - Ax is the predicted target value, given known (control) matrix A
    and state vector x,
    - gamma is regularization parameter that controls the penalty
    applied to the coefficients,

    Note
    ----
    This function replaces the function previously called "regularised_lsq_fom".

    Raises
    ------
    EquilibriaError
        Least squares result < 0 or NaN

    Returns
    -------
    :
        figure of merit
    """
    fom_rss = ols(x, a_mat, b_vec)
    fom_tik = fom_rss + gamma * gamma * x.T @ x
    if fom_tik <= 0:
        raise EquilibriaError("Least-squares objective function less than zero or nan.")
    return fom_tik


def elastic_net(
    x: np.ndarray,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    gamma: np.ndarray,
    alpha: np.ndarray,
):
    """
    Regression with a combination of both L1 and L2 regularization terms.

    Have a hyperparameter, alpha, to control the ratio of regularisation terms.

    L1 - promotes sparsity and variable selection
    L2 - handles multicollinearity

    Can use as objective function:

    \t:math:`\\textrm{rss} = || Ax - b ||^{2}`\n
    \t:math:`\\textrm{minimise} \\textrm{rss} + \\gamma`
    \t:math:`((1-\\alpha) \\sum_{j=1}^{p} | x_{j} | +`
    \t:math:`\\alpha \\gamma || x ||^{2})`

    where:
    - b is the target vector,
    - Ax is the predicted target value, given known (control) matrix A
    and state vector x,
    - gamma is a regularization parameter that controls the penalty
    applied to the coefficients,
    - p is the number of predictor variables,
    - alpha is a parameter to control the ratio of regularisation terms.

    Raises
    ------
    EquilibriaError
        Least squares result < 0 or NaN

    Returns
    -------
    :
        figure of merit
    """
    fom_rss = ols(x, a_mat, b_vec)
    fom_net = fom_rss + gamma * (
        (1 - alpha) * np.sum(np.abs(x)) + alpha * gamma * x.T @ x
    )
    if fom_net <= 0:
        raise EquilibriaError("Least-squares objective function less than zero or nan.")
    return fom_net


# =============================================================================
# Solution Vector
# =============================================================================


def tikhonov_ridge_solution(
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    alpha: np.ndarray,
):
    """
    Ridge a.k.a Tikhonov a.k.a. L2 Regression.

    This is a form of Tikhonov Regularisation which replaces gamma with a multiple
    of the identity matrix (gamma = alpha * I).

    This is used to find a solution vector x:

    \t:math:`x = (A^T A + {\\alpha}^2 I)^{-1}A^T b`

    where:
    - b is the target vector,
    - A is a known (control) matrix A,
    - alpha is a regularization parameter,
    - I is the identity matrix.

    Note
    ----
    This function replaces the function previously called "tikhonov".

    Returns
    -------
    x:
        The result vector
    """
    try:
        return np.dot(
            np.linalg.inv(np.dot(a_mat.T, a_mat) + alpha**2 * np.eye(a_mat.shape[1])),
            np.dot(a_mat.T, b_vec),
        )
    except np.linalg.LinAlgError:
        bluemira_warn("Tikhonov singular matrix..!")
        return np.dot(
            np.linalg.pinv(np.dot(a_mat.T, a_mat) + alpha**2 * np.eye(a_mat.shape[1])),
            np.dot(a_mat.T, b_vec),
        )

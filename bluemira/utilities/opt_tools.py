# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Optimisation utilities
"""

import numpy as np
from bluemira.utilities.error import InternalOptError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.error import EquilibriaError


class ConstraintLibrary:
    """
    Library class containing constraint functions to use in NLOpt constrained
    optimisation problems.

    Constraint functions must be of the form:

    .. code-block:: python

        def f_constraint(constraint, x, grad, args):
            constraint[:] = my_constraint_calc(x)
            if grad.size > 0:
                grad[:] = my_gradient_calc(x)
            return constraint

    The constraint function convention is such that c <= 0 is sought. I.e. all constraint
    values must be negative.

    Note that the gradient (Jacobian) of the constraint function is of the form:

    .. math::

        \\nabla \\mathbf{c} = \\begin{bmatrix}
                \\dfrac{\\partial c_{0}}{\\partial x_0} & \\dfrac{\\partial c_{0}}{\\partial x_1} & ... \n
                \\dfrac{\\partial c_{1}}{\\partial x_0} & \\dfrac{\\partial c_{1}}{\\partial x_1} & ... \n
                ... & ... & ... \n
                \\end{bmatrix}

    The grad and constraint matrices must be assigned in place.
    """  # noqa (W505)

    def objective_constraint(
        self, constraint, vector, grad, objective_function, maximum_fom=1.0
    ):
        """
        Constraint function to constrain the maximum value of an NLOpt objective
        function provided

        Parameters
        ----------
        objective_function: callable
            NLOpt objective function to use in constraint.
        maximum_fom: float (default=1.0)
            Value to constrain the objective function by during optimisation.
        """
        constraint[:] = objective_function(self, vector, grad) - maximum_fom
        return constraint


class ObjectiveLibrary:
    """
    Library class containing objective functions to use in NLOpt constrained
    optimisation problems.

    Objective functions must be of the form:

    .. code-block:: python

        def f_objective(x, grad, args):
            if grad.size > 0:
                grad[:] = my_gradient_calc(x)
            return my_objective_calc(x)

    The objective function is minimised, so lower values are "better".

    Note that the gradient of the objective function is of the form:

    :math:`\\nabla f = \\bigg[\\dfrac{\\partial f}{\\partial x_0}, \\dfrac{\\partial f}{\\partial x_1}, ...\\bigg]`
    """  # noqa (W505)

    def regularised_lsq_objective(self, vector, grad, scale, A, b, gamma):
        """
        Objective function for nlopt optimisation (minimisation),
        consisting of a least-squares objective with Tikhonov
        regularisation term, which updates the gradient in-place.

        Parameters
        ----------
        vector: np.array(n_C)
            State vector of the array of coil currents.
        grad: np.array
            Local gradient of objective function used by LD NLOPT algorithms.
            Updated in-place.

        Returns
        -------
        fom: Value of objective function (figure of merit).
        """
        vector = vector * scale
        fom, err = regularised_lsq_fom(vector, A, b, gamma)
        if grad.size > 0:
            jac = 2 * A.T @ A @ vector / np.float(len(b))
            jac -= 2 * A.T @ b / np.float(len(b))
            jac += 2 * gamma * gamma * vector
            grad[:] = scale * jac
        if not fom > 0:
            raise EquilibriaError(
                "Optimiser least-squares objective function less than zero or nan."
            )
        return fom


# =============================================================================
# Analytical objective functions
# =============================================================================


def tikhonov(A, b, gamma):
    """
    Tikhonov regularisation of Ax-b problem.

    \t:math:`\\textrm{minimise} || Ax - b ||^2 + ||{\\gamma} \\cdot x ||^2`\n
    \t:math:`x = (A^T A + {\\gamma}^2 I)^{-1}A^T b`

    Parameters
    ----------
    A: np.array(n, m)
        The 2-D A matrix of responses
    b: np.array(n)
        The 1-D b vector of values
    gamma: float
        The Tikhonov regularisation parameter

    Returns
    -------
    x: np.array(m)
        The result vector
    """
    try:
        return np.dot(
            np.linalg.inv(np.dot(A.T, A) + gamma ** 2 * np.eye(A.shape[1])),
            np.dot(A.T, b),
        )
    except np.linalg.LinAlgError:
        bluemira_warn("Tikhonov singular matrix..!")
        return np.dot(
            np.linalg.pinv(np.dot(A.T, A) + gamma ** 2 * np.eye(A.shape[1])),
            np.dot(A.T, b),
        )


def regularised_lsq_fom(x, A, b, gamma):
    """
    Figure of merit for the least squares problem Ax = b, with
    Tikhonov regularisation term. Normalised for the number of
    targets.

    ||(Ax - b)||²/ len(b)] + ||Γx||²

    Parameters
    ----------
    x : np.array(m)
        The 1-D x state vector.
    A: np.array(n, m)
        The 2-D A control matrix
    b: np.array(n)
        The 1-D b vector of target values
    gamma: float
        The Tikhonov regularisation parameter.

    Returns
    -------
    fom: float
        Figure of merit, explicitly given by
        ||(Ax - b)||²/ len(b)] + ||Γx||²
    residual: np.array(n)
        Residual vector (Ax - b)
    """
    residual = np.dot(A, x) - b
    number_of_targets = np.float(len(residual))
    fom = residual.T @ residual / number_of_targets + gamma * gamma * x.T @ x

    if not fom > 0:
        raise bluemira_warn("Least-squares objective function less than zero or nan.")
    return fom, residual


def least_squares(A, b):
    """
    Least squares optimisation.

    \t:math:`\\textrm{minimise} || Ax - b ||^{2}_{2}`\n
    \t:math:`x = (A^T A)^{-1}A^T b`

    Parameters
    ----------
    A: np.array(n, m)
        The 2-D A matrix of responses
    b: np.array(n)
        The 1-D b vector of values

    Returns
    -------
    x: np.array(m)
        The result vector
    """
    return np.linalg.solve(A, b)


# =============================================================================
# Scipy interface
# =============================================================================


def process_scipy_result(res):
    """
    Handle a scipy.minimize OptimizeResult object. Process error codes, if any.

    Parameters
    ----------
    res: OptimizeResult

    Returns
    -------
    x: np.array
        The optimal set of parameters (result of the optimisation)

    Raises
    ------
    InternalOptError if an error code returned without a usable result.
    """
    if res.success:
        return res.x

    if not hasattr(res, "status"):
        bluemira_warn("Scipy optimisation was not succesful. Failed without status.")
        raise InternalOptError("\n".join([res.message, res.__str__()]))

    elif res.status == 8:
        # This can happen when scipy is not convinced that it has found a minimum.
        bluemira_warn(
            "\nOptimiser (scipy) found a positive directional derivative,\n"
            "returning suboptimal result. \n"
            "\n".join([res.message, res.__str__()])
        )
        return res.x

    elif res.status == 9:
        bluemira_warn(
            "\nOptimiser (scipy) exceeded number of iterations, returning "
            "suboptimal result. \n"
            "\n".join([res.message, res.__str__()])
        )
        return res.x

    else:
        raise InternalOptError("\n".join([res.message, res.__str__()]))

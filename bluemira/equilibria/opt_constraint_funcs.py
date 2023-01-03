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

"""
Equilibrium optimisation constraint functions.
for use in NLOpt constrained
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

If grad is not updated, the constraint can still be used for derivative-free
optimisaiton algorithms, but will need to be updated or approximated for use
in derivative based algorithms, such as those utilising gradient descent.
"""  # noqa: W505

import numpy as np


def objective_constraint(constraint, vector, grad, objective_function, maximum_fom=1.0):
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
    constraint[:] = objective_function(vector, grad) - maximum_fom
    return constraint


def Ax_b_constraint(constraint, vector, grad, a_mat, b_vec, value, scale):  # noqa: N802
    """
    Constraint function of the form:
        A.x - b < value

    Parameters
    ----------
    constraint: np.ndarray
        Constraint array (modified in place)
    vector: np.ndarray
        Variable vector
    grad: np.ndarray
        Constraint Jacobian (modified in place)
    a_mat: np.ndarray
        Response matrix
    b_vec: np.ndarray
        Target value vector
    value: float
        Target constraint value
    scale: float
        Current scale with which to calculate the constraints
    """
    constraint[:] = np.dot(a_mat, scale * vector) - b_vec - value
    if grad.size > 0:
        grad[:] = scale * a_mat
    return constraint


def L2_norm_constraint(  # noqa: N802
    constraint, vector, grad, a_mat, b_vec, value, scale
):
    """
    Constrain the L2 norm of an Ax-b system of equations.
    ||(Ax - b)||² < value

    Parameters
    ----------
    constraint: np.ndarray
        Constraint array (modified in place)
    vector: np.ndarray
        Variable vector
    grad: np.ndarray
        Constraint Jacobian (modified in place)
    A_mat: np.ndarray
        Response matrix
    b_vec: np.ndarray
        Target value vector
    scale: float
        Current scale with which to calculate the constraints

    Returns
    -------
    constraint: np.ndarray
        Updated constraint vector
    """
    vector = scale * vector
    residual = a_mat @ vector - b_vec
    constraint[:] = residual.T @ residual - value

    if grad.size > 0:
        grad[:] = 2 * scale * (a_mat.T @ a_mat @ vector - a_mat.T @ b_vec)

    return constraint


def current_midplane_constraint(
    constraint, vector, grad, eq, radius, scale, inboard=True
):
    """
    Constraint function to constrain the inboard or outboard midplane
    of the plasma during optimisation.

    Parameters
    ----------
    constraint: np.ndarray
        Constraint array (modified in place)
    vector: np.ndarray
        Current vector
    grad: np.ndarray
        Constraint Jacobian (modified in place)
    eq: Equilibrium
        Equilibrium to use to fetch last closed flux surface from.
    radius: float
        Toroidal radius at which to constrain the plasma midplane.
    scale: float
        Current scale with which to calculate the constraints
    inboard: bool (default=True)
        Boolean controlling whether to constrain the inboard (if True) or
        outboard (if False) side of the plasma midplane.
    """
    eq.coilset.set_control_currents(vector * scale)
    lcfs = eq.get_LCFS()
    if inboard:
        constraint[:] = radius - min(lcfs.x)
    else:
        constraint[:] = max(lcfs.x) - radius
    return constraint


def coil_force_constraints(
    constraint,
    vector,
    grad,
    a_mat,
    b_vec,
    n_PF,
    n_CS,
    PF_Fz_max,
    CS_Fz_sum_max,
    CS_Fz_sep_max,
    scale,
):
    """
    Current optimisation force constraints on coils

    Parameters
    ----------
    constraint: np.ndarray
        Constraint array (modified in place)
    vector: np.ndarray
        Current vector
    grad: np.ndarray
        Constraint Jacobian (modified in place)
    a_mat: np.ndarray
        Response matrix block for Fx and Fz
    b_vec: np.ndarray
        Background value vector block for Fx and Fz
    n_PF: int
        Number of PF coils
    n_CS: int
        Number of CS coils
    PF_Fz_max: float
        Maximum vertical force on each PF coil [N]
    CS_Fz_sum_max: float
        Maximum total vertical force on the CS stack [N]
    CS_Fz_sep_max: float
        Maximum vertical separation force in the CS stack [N]
    scale: float
        Current scale with which to calculate the constraints

    Returns
    -------
    constraint: np.ndarray
        Updated constraint vector
    """
    n_coils = len(vector)
    currents = scale * vector

    # get coil force and jacobian
    F = np.zeros((n_coils, 2))
    PF_Fz_max /= scale
    CS_Fz_sep_max /= scale
    CS_Fz_sum_max /= scale

    for i in range(2):  # coil force
        # NOTE: * Hadamard matrix product
        F[:, i] = currents * (a_mat[:, :, i] @ currents + b_vec[:, i])

    F /= scale  # Scale down to MN

    # Absolute vertical force constraint on PF coils
    constraint[:n_PF] = F[:n_PF, 1] ** 2 - PF_Fz_max**2

    if n_CS != 0:
        # vertical forces on CS coils
        cs_fz = F[n_PF:, 1]
        # vertical force on CS stack
        cs_z_sum = np.sum(cs_fz)
        # Absolute sum of vertical force constraint on entire CS stack
        constraint[n_PF] = cs_z_sum**2 - CS_Fz_sum_max**2
        for i in range(n_CS - 1):  # evaluate each gap in CS stack
            # CS seperation constraints
            f_sep = np.sum(cs_fz[: i + 1]) - np.sum(cs_fz[i + 1 :])
            constraint[n_PF + 1 + i] = f_sep - CS_Fz_sep_max

    # calculate constraint jacobian
    if grad.size > 0:
        dF = np.zeros((n_coils, n_coils, 2))  # noqa: N806
        im = currents.reshape(-1, 1) @ np.ones((1, n_coils))  # current matrix
        for i in range(2):
            dF[:, :, i] = im * a_mat[:, :, i]
            diag = (
                a_mat[:, :, i] @ currents
                + currents * np.diag(a_mat[:, :, i])
                + b_vec[:, i]
            )
            np.fill_diagonal(dF[:, :, i], diag)

        # Absolute vertical force constraint on PF coils
        grad[:n_PF] = 2 * dF[:n_PF, :, 1]

        if n_CS != 0:
            # Absolute sum of vertical force constraint on entire CS stack
            grad[n_PF] = 2 * np.sum(dF[n_PF:, :, 1], axis=0)

            for i in range(n_CS - 1):  # evaluate each gap in CS stack
                # CS separation constraint Jacobians
                f_up = np.sum(dF[n_PF : n_PF + i + 1, :, 1], axis=0)
                f_down = np.sum(dF[n_PF + i + 1 :, :, 1], axis=0)
                grad[n_PF + 1 + i] = f_up - f_down
    return constraint


def field_constraints(
    constraint, vector, grad, ax_mat, az_mat, bxp_vec, bzp_vec, B_max, scale
):
    """
    Current optimisation poloidal field constraints at prescribed locations

    Parameters
    ----------
    constraint: np.ndarray
        Constraint array (modified in place)
    vector: np.ndarray
        Current vector
    grad: np.ndarray
        Constraint Jacobian (modified in place)
    ax_mat: np.ndarray
        Response matrix for Bx (active coil contributions)
    az_mat: np.ndarray
        Response matrix for Bz (active coil contributions)
    bxp_vec: np.ndarray
        Background vector for Bx (passive coil contributions)
    bzp_vec: np.ndarray
        Background vector for Bz (passive coil contributions)
    B_max: np.ndarray
        Maximum fields inside the coils
    scale: float
        Current scale with which to calculate the constraints

    Returns
    -------
    constraint: np.ndarray
        Updated constraint vector
    """
    currents = scale * vector
    Bx_a = ax_mat @ currents
    Bz_a = az_mat @ currents

    B = np.hypot(Bx_a + bxp_vec, Bz_a + bzp_vec)
    if grad.size > 0:
        grad[:] = (
            Bx_a * (Bx_a @ currents + bxp_vec) + Bz_a * (Bz_a @ currents + bzp_vec)
        ) / (B * scale**2)

    constraint[:] = B - B_max
    return constraint

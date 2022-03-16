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
"""  # noqa (W505)

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


def current_midplane_constraint(constraint, vector, grad, eq, radius, inboard=True):
    """
    Constraint function to constrain the inboard or outboard midplane
    of the plasma during optimisation.

    Parameters
    ----------
    eq: Equilibrium
        Equilibrium to use to fetch last closed flux surface from.
    radius: float
        Toroidal radius at which to constrain the plasma midplane.
    inboard: bool (default=True)
        Boolean controlling whether to constrain the inboard (if True) or
        outboard (if False) side of the plasma midplane.
    """
    eq.coilset.set_control_currents(vector * 1e6)
    lcfs = eq.get_LCFS()
    if inboard:
        constraint[:] = radius - min(lcfs.x)
    else:
        constraint[:] = max(lcfs.x) - radius
    return constraint


def coil_force_constraints(
    constraint, vector, grad, eq, n_PF, n_CS, PF_Fz_max, CS_Fz_sum, CS_Fz_sep, scale
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
    eq: Equilibrium
        Equilibrium object with which to calculate constraints
    n_PF: int
        Number of PF coils
    n_CS: int
        Number of CS coils
    PF_Fz_max: float
        Maximum vertical force on each PF coil [MN]
    CS_Fz_sum: float
        Maximum total vertical force on the CS stack [MN]
    CS_Fz_sep: float
        Maximum vertical separation force in the CS stack [MN]
    scale: float
        Current scale with which to calculate the constraints

    Returns
    -------
    constraint: np.ndarray
        Updated constraint vector
    """
    # get coil force and jacobian
    F, dF = eq.force_field.calc_force(vector * scale)  # noqa :N803
    F /= scale  # Scale down to MN
    # dF /= self.scale

    # calculate constraint jacobian
    if grad.size > 0:
        # PFz lower bound
        grad[:n_PF] = -dF[:n_PF, :, 1]
        # PFz upper bound
        grad[n_PF : 2 * n_PF] = dF[:n_PF, :, 1]

        if n_CS != 0:
            # CS sum lower
            grad[2 * n_PF] = -np.sum(dF[n_PF:, :, 1], axis=0)
            # CS sum upper
            grad[2 * n_PF + 1] = np.sum(dF[n_PF:, :, 1], axis=0)
            for j in range(n_CS - 1):  # evaluate each gap in CS stack
                # CS separation constraint Jacobians
                f_up = np.sum(dF[n_PF : n_PF + j + 1, :, 1], axis=0)
                f_down = np.sum(dF[n_PF + j + 1 :, :, 1], axis=0)
                grad[2 * n_PF + 2 + j] = f_up - f_down

    # vertical force on PF coils
    pf_fz = F[:n_PF, 1]
    # PFz lower bound
    constraint[:n_PF] = -PF_Fz_max - pf_fz
    # PFz upper bound
    constraint[n_PF : 2 * n_PF] = pf_fz - PF_Fz_max

    if n_CS != 0:
        # vertical force on CS coils
        cs_fz = F[n_PF:, 1]
        # vertical force on CS stack
        cs_z_sum = np.sum(cs_fz)
        # CSsum lower bound
        constraint[2 * n_PF] = -CS_Fz_sum - cs_z_sum
        # CSsum upper bound
        constraint[2 * n_PF + 1] = cs_z_sum - CS_Fz_sum
        for j in range(n_CS - 1):  # evaluate each gap in CS stack
            # CS seperation constraints
            f_sep = np.sum(cs_fz[: j + 1]) - np.sum(cs_fz[j + 1 :])
            constraint[2 * n_PF + 2 + j] = f_sep - CS_Fz_sep
    return constraint


def coil_field_constraints(constraint, vector, grad, eq, B_max, scale):
    """
    Current optimisation field constraints on coils

    Parameters
    ----------
    constraint: np.ndarray
        Constraint array (modified in place)
    vector: np.ndarray
        Current vector
    grad: np.ndarray
        Constraint Jacobian (modified in place)
    eq: Equilibrium
        Equilibrium object with which to calculate constraints
    B_max: np.ndarray
        Maximum fields inside the coils
    scale: float
        Current scale with which to calculate the constraints

    Returns
    -------
    constraint: np.ndarray
        Updated constraint vector
    """
    B, dB = eq.force_field.calc_field(vector * scale)  # noqa :N803
    dB /= scale**2
    if grad.size > 0:
        grad[:] = dB
    constraint[:] = B - B_max
    return constraint

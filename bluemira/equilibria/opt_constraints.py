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
import numpy as np
from matplotlib import scale
from matplotlib.pyplot import axis

from bluemira.equilibria.winding_pack import (
    generate_cable_current,
    plot_2D_field_map,
    select_temperature,
)

# from bluemira.magnetostatics.circuits import HelmholtzCage

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
    constraint,
    vector,
    grad,
    eq,
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
    eq: Equilibrium
        Equilibrium object with which to calculate constraints
    n_PF: int
        Number of PF coils
    n_CS: int
        Number of CS coils
    PF_Fz_max: float
        Maximum vertical force on each PF coil [MN]
    CS_Fz_sum_max: float
        Maximum total vertical force on the CS stack [MN]
    CS_Fz_sep_max: float
        Maximum vertical separation force in the CS stack [MN]
    scale: float
        Current scale with which to calculate the constraints

    Returns
    -------
    constraint: np.ndarray
        Updated constraint vector

    Notes
    -----
    TODO: Presently only handles CoilSets with Coils (SymmetricCircuits not yet
    supported)
    """
    # get coil force and jacobian
    F, dF = eq.force_field.calc_force(vector * scale)  # noqa :N803
    F /= scale  # Scale down to MN
    # dF /= self.scale

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


def coil_field_constraints(constraint, vector, grad, eq, B_max, scale):
    """
    Current optimisation poloidal field constraints on coils

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

    Notes
    -----
    TODO: Presently only handles CoilSets with Coils (SymmetricCircuits not yet
    supported)
    TODO: Presently only accounts for poloidal field contributions from PF coils and
    plasma (TF from TF coils not accounted for if PF coils are inside the TF coils.)
    """
    B, dB = eq.force_field.calc_field(vector * scale)  # noqa :N803
    dB /= scale**2
    if grad.size > 0:
        grad[:] = dB
    constraint[:] = B - B_max
    return constraint
    
def critical_current_constraint(
    constraint,
    vector,
    grad,
    eq,
    tf_source,
    tf,
    tf_centerline,
    hmc,
    conductor_id,
    temperature_id,
    conductors,
    scale,
):  # tf_source after eq
    
    eq.coilset.set_control_currents(vector * scale)
    x, z = eq.coilset.get_positions()  # arrays
    y = np.zeros_like(x)
    coords = np.stack([x, z], axis=0)
    # import ipdb
    # ipdb.set_trace()
    Bx = eq.Bx(x, z)  # array
    Bz = eq.Bz(x, z)  # array

    # hmc = HelmholtzCage(tf_source,tf.params.n_TF)

    if hmc == None:
        B_tf = eq.Bt(x)
        B = np.sqrt(Bx**2 + Bz**2 + B_tf**2)
    else:
        B_vec = hmc.field(x, y, z)  # HM cage
        B_pf_plasma = np.stack([eq.Bx(x, z), eq.Bz(x, z)], axis=0)

        B_vec[
            (0, 2),
        ] += B_pf_plasma  # stacked array
        # import ipdb
        # ipdb.set_trace()

        B = np.sqrt(B_vec[0] ** 2 + B_vec[1] ** 2 + B_vec[2] ** 2)  # stacked array
        import ipdb

        ipdb.set_trace()
        plot_2D_field_map(coords, B, eq.coilset, tf_centerline)
        import ipdb

        ipdb.set_trace()

    # user imputed temperatures for HTS and LTS"
    T = select_temperature(conductor_id[0], temperature_id)
    I_crit = generate_cable_current(conductor_id[0], conductors, B, T)  # np.array

    print("I_crit", I_crit)  # diagnostic
    constraint[:] = abs(vector) - abs(I_crit)
    return constraint

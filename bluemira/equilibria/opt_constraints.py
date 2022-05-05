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

from bluemira.utilities.opt_problems import OptimisationConstraint


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


def Ax_b_constraint(constraint, vector, grad, a_mat, b_vec):  # noqa: N802
    """
    Constraint function of the form:
        A.x - b < 0.0

    Parameters
    ----------
    constraint: np.ndarray
        Constraint array (modified in place)
    vector: np.ndarray
        Current vector
    grad: np.ndarray
        Constraint Jacobian (modified in place)
    A_mat: np.ndarray
        Response matrix
    b_vec: np.ndarray
        Target value vector
    """
    constraint[:] = np.dot(a_mat, vector) - b_vec
    if grad.size > 0:
        grad[:] = a_mat
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


from abc import ABC, abstractmethod
from typing import Union

from bluemira.utilities.tools import is_num


class MagneticConstraint(ABC, OptimisationConstraint):
    """
    Abstract base class for a magnetic optimisation constraint.
    """

    def __init__(
        self,
        target_value=None,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance=1e-6,
        constraint_type="inequality",
    ):
        self.target_value = target_value * np.ones(len(self))
        if is_num(tolerance):
            tolerance = tolerance * np.ones(len(self))
        self.weights = weights
        args = {"a_mat": None, "b_vec": None}
        super().__init__(
            f_constraint=Ax_b_constraint,
            f_constraint_args=args,
            tolerance=tolerance,
            constraint_type=constraint_type,
        )

    def prepare(self, equilibrium, I_not_dI=False, fixed_coils=False):  # noqa :N803
        """
        Prepare the constraint for use in an equilibrium optimisation problem.
        """
        if I_not_dI:
            # hack to change from dI to I optimiser (and keep both)
            # When we do dI optimisation, the background vector includes the
            # contributions from the whole coilset (including active coils)
            # When we do I optimisation, the background vector only includes
            # contributions from the passive coils (plasma)
            # TODO: Add passive coil contributions here
            dummy = equilibrium.plasma_coil()
            dummy.coilset = equilibrium.coilset
            equilibrium = dummy

        # Re-build control response matrix
        if not fixed_coils or (fixed_coils and self._args["a_mat"] is None):
            self._args["a_mat"] = self.control_response(equilibrium.coilset)
            self.update_target(equilibrium)

        self._args["b_vec"] = self.target_value - self.evaluate(equilibrium)

    def __call__(self, constraint, vector, grad):
        return super().__call__(constraint, vector, grad)

    def update_target(self, equilibrium):
        pass

    @abstractmethod
    def control_response(self, coilset):
        """
        Calculate control response of a CoilSet to the constraint.
        """
        pass

    @abstractmethod
    def evaluate(self, eq):
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        pass

    @abstractmethod
    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """
        pass

    def __len__(self):
        """
        The mathematical size of the constraint.

        Notes
        -----
        Length of the array if an array is specified, otherwise 1 for a float.
        """
        return len(self.x) if hasattr(self.x, "__len__") else 1


class AbsoluteMagneticConstraint(MagneticConstraint):
    def __init__(
        self,
        x,
        z,
        target_value,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance=1e-6,
        constraint_type="inequality",
    ):
        self.x = x
        self.z = z
        super().__init__(
            target_value, weights, tolerance=tolerance, constraint_type=constraint_type
        )


class RelativeMagneticConstraint(MagneticConstraint):
    def __init__(
        self,
        x,
        z,
        ref_x,
        ref_z,
        target_value: float = 0.0,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance=1e-6,
        constraint_type="inequality",
    ):
        self.x = x
        self.z = z
        self.ref_x = ref_x
        self.ref_z = ref_z
        super().__init__(
            target_value, weights, tolerance=tolerance, constraint_type=constraint_type
        )

    @abstractmethod
    def update_target(self, equilibrium):
        pass


class BxConstraint(AbsoluteMagneticConstraint):
    """
    Absolute Bx value constraint.
    """

    def control_response(self, coilset):
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return coilset.control_Bx(self.x, self.z)

    def evaluate(self, eq):
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return eq.Bx(self.x, self.z)

    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """
        kwargs = {"marker": 9, "markersize": 15, "color": "b", "zorder": 45}
        ax.plot(self.x, self.z, "s", **kwargs)


class FieldNullConstraint(AbsoluteMagneticConstraint):
    """
    Magnetic field null constraint. In practice sets the Bx and Bz field components
    to be 0 at the specified location.
    """

    def __init__(
        self,
        x,
        z,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance=1e-6,
        constraint_type="equality",
    ):
        super().__init__(x, z, 0.0, weights, tolerance, constraint_type)

    def control_response(self, coilset):
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return np.array(
            [coilset.control_Bx(self.x, self.z), coilset.control_Bz(self.x, self.z)]
        )

    def evaluate(self, eq):
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return np.array([eq.Bx(self.x, self.z), eq.Bz(self.x, self.z)])

    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """
        kwargs = {"marker": "X", "color": "b", "markersize": 10, "zorder": 45}
        ax.plot(self.x, self.z, "s", **kwargs)

    def __len__(self):
        """
        The mathematical size of the constraint.
        """
        return 2


class IsofluxConstraint(RelativeMagneticConstraint):
    """
    Isoflux constraint for a set of points relative to a reference point.
    """

    def control_response(self, coilset):
        """
        Calculate control response of a CoilSet to the constraint.
        """
        c_ref = coilset.control_psi(self.ref_x, self.ref_z)
        return (
            np.array(coilset.control_psi(self.x, self.z))
            - np.array(c_ref)[:, np.newaxis]
        ).T

    def evaluate(self, eq):
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return eq.psi(self.x, self.z)

    def update_target(self, eq):
        """
        We need to update the target value, as it is a relative constraint.
        """
        self.target_value = eq.psi(self.ref_x, self.ref_z)

    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """
        kwargs = {
            "marker": "o",
            "fillstyle": "none",
            "markeredgewidth": 3,
            "markeredgecolor": "b",
            "markersize": 10,
            "zorder": 45,
        }
        ax.plot(self.x, self.z, "s", **kwargs)
        kwargs["markeredgewidth"] = 5
        ax.plot(self.ref_x, self.ref_z, "s", **kwargs)

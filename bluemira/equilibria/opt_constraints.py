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

from bluemira.equilibria.plotting import ConstraintPlotter
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


def Ax_b_constraint(constraint, vector, grad, a_mat, b_vec, value, scale):  # noqa: N802
    """
    Constraint function of the form:
        A.x - b < value

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
        Current vector
    grad: np.ndarray
        Constraint Jacobian (modified in place)
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
from typing import List, Union

from bluemira.utilities.tools import is_num


def _get_dummy_equilibrium(equilibrium, I_not_dI):
    """
    Get a dummy equilibrium for current optimisation where the background response is
    solely due to the plasma and passive coils.
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
    return equilibrium


class MagneticConstraint(ABC, OptimisationConstraint):
    """
    Abstract base class for a magnetic optimisation constraint.

    Can be used as a standalone constraint for use in an optimisation problem. In which
    case the constraint is of the form:
        ||(Ax - b)||² < target_value

    Can be used in a MagneticConstraintSet
    """

    def __init__(
        self,
        target_value: float = 0.0,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance=1e-6,
        f_constraint=L2_norm_constraint,
        constraint_type="inequality",
    ):
        self.target_value = target_value * np.ones(len(self))
        if is_num(tolerance):
            if f_constraint == L2_norm_constraint:
                tolerance = tolerance * np.ones(1)
            else:
                tolerance = tolerance * np.ones(len(self))
        self.weights = weights
        args = {"a_mat": None, "b_vec": None, "value": 0.0, "scale": 1.0}
        super().__init__(
            f_constraint=f_constraint,
            f_constraint_args=args,
            tolerance=tolerance,
            constraint_type=constraint_type,
        )

    def prepare(self, equilibrium, I_not_dI=False, fixed_coils=False):  # noqa :N803
        """
        Prepare the constraint for use in an equilibrium optimisation problem.
        """
        equilibrium = _get_dummy_equilibrium(equilibrium, I_not_dI)

        # Re-build control response matrix
        if not fixed_coils or (fixed_coils and self._args["a_mat"] is None):
            self._args["a_mat"] = self.control_response(equilibrium.coilset)

        self.update_target(equilibrium)
        self._args["b_vec"] = self.target_value - self.evaluate(equilibrium)

    def __call__(self, constraint, vector, grad):
        return super().__call__(constraint, vector, grad)

    def update_target(self, equilibrium):
        """
        Update the target value of the magnetic constraint.
        """
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
    """
    Abstract base class for absolute magnetic constraints, where the target
    value is prescribed in absolute terms.
    """

    def __init__(
        self,
        x,
        z,
        target_value,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance=1e-6,
        f_constraint=Ax_b_constraint,
        constraint_type="equality",
    ):
        self.x = x
        self.z = z
        super().__init__(
            target_value,
            weights,
            tolerance=tolerance,
            f_constraint=f_constraint,
            constraint_type=constraint_type,
        )


class RelativeMagneticConstraint(MagneticConstraint):
    """
    Abstract base class for relative magnetic constraints, where the target
    value is prescribed with respect to a reference point.
    """

    def __init__(
        self,
        x,
        z,
        ref_x,
        ref_z,
        constraint_value: float = 0.0,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance=1e-6,
        f_constraint=L2_norm_constraint,
        constraint_type="inequality",
    ):
        self.x = x
        self.z = z
        self.ref_x = ref_x
        self.ref_z = ref_z
        super().__init__(
            0.0,
            weights,
            tolerance=tolerance,
            f_constraint=f_constraint,
            constraint_type=constraint_type,
        )
        self._args["value"] = constraint_value

    @abstractmethod
    def update_target(self, equilibrium):
        pass


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
        super().__init__(
            x,
            z,
            0.0,
            weights,
            tolerance=tolerance,
            constraint_type=constraint_type,
            f_constraint=L2_norm_constraint,
        )

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


class PsiConstraint(AbsoluteMagneticConstraint):
    """
    Absolute psi value constraint.
    """

    def __init__(
        self, x, z, target_value, weights: Union[float, np.ndarray] = 1.0, tolerance=1e-6
    ):
        super().__init__(
            x,
            z,
            target_value,
            weights,
            tolerance,
            f_constraint=Ax_b_constraint,
            constraint_type="equality",
        )

    def control_response(self, coilset):
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return np.array(coilset.control_psi(self.x, self.z))

    def evaluate(self, eq):
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return eq.psi(self.x, self.z)

    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """
        kwargs = {"marker": "s", "markersize": 8, "color": "b"}
        ax.plot(self.x, self.z, "s", **kwargs)


class IsofluxConstraint(RelativeMagneticConstraint):
    """
    Isoflux constraint for a set of points relative to a reference point.
    """

    def __init__(
        self,
        x,
        z,
        ref_x,
        ref_z,
        constraint_value,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance: float = 1e-6,
    ):

        super().__init__(
            x,
            z,
            ref_x,
            ref_z,
            constraint_value,
            weights=weights,
            f_constraint=L2_norm_constraint,
            tolerance=tolerance,
            constraint_type="inequality",
        )

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
        self.target_value = float(eq.psi(self.ref_x, self.ref_z))

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


class PsiBoundaryConstraint(AbsoluteMagneticConstraint):
    """
    Absolute psi value constraint on the plasma boundary. Gets updated when
    the plasma boundary flux value is changed.
    """

    def control_response(self, coilset):
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return np.array(coilset.control_psi(self.x, self.z)).T

    def evaluate(self, eq):
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return eq.psi(self.x, self.z)

    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """
        kwargs = {"marker": "o", "markersize": 8, "color": "b"}
        ax.plot(self.x, self.z, "s", **kwargs)


class MagneticConstraintSet(ABC):
    """
    A set of magnetic constraints to be applied to an equilibrium. The optimisation
    problem is of the form:

        [A][x] = [b]

    where:

        [b] = [target] - [background]

    The target vector is the vector of desired values. The background vector
    is the vector of values due to uncontrolled current terms (plasma and passive
    coils).


    Use of class:

        - Inherit from this class
        - Add a __init__(args) method
        - Populate constraints with super().__init__(List[MagneticConstraint])
    """

    constraints: List[MagneticConstraint]
    eq: object
    A: np.array
    target: np.array
    background: np.array

    __slots__ = ["constraints", "eq", "coilset", "A", "w", "target", "background"]

    def __init__(self, constraints):
        self.constraints = constraints

    def __call__(self, equilibrium, I_not_dI=False, fixed_coils=False):  # noqa :N803
        equilibrium = _get_dummy_equilibrium(equilibrium, I_not_dI)

        self.eq = equilibrium
        self.coilset = equilibrium.coilset

        # Update relative magnetic constraints without updating A matrix
        for constraint in self.constraints:
            if isinstance(constraint, RelativeMagneticConstraint):
                constraint.update_target(equilibrium)

        # Re-build control response matrix
        if not fixed_coils or (fixed_coils and self.A is None):
            self.build_control_matrix()
            self.build_target()

        self.build_background()
        self.build_weight_matrix()

    def __len__(self):
        """
        The mathematical size of the constraint set.
        """
        return sum([len(c) for c in self.constraints])

    def get_weighted_arrays(self):
        """
        Get [A] and [b] scaled by weight matrix.
        Weight matrix assumed to be diagonal.
        """
        weights = self.w
        weighted_a = weights[:, np.newaxis] * self.A
        weighted_b = weights * self.b
        return weights, weighted_a, weighted_b

    def build_weight_matrix(self):
        """
        Build the weight matrix used in optimisation.
        Assumed to be diagonal.
        """
        self.w = np.zeros(len(self))

        i = 0
        for constraint in self.constraints:
            n = len(constraint)
            self.w[i : i + n] = constraint.weights
            i += n

    def build_control_matrix(self):
        """
        Build the control response matrix used in optimisation.
        """
        self.A = np.zeros((len(self), len(self.coilset._ccoils)))

        i = 0
        for constraint in self.constraints:
            n = len(constraint)
            self.A[i : i + n, :] = constraint.control_response(self.coilset)
            i += n

    def build_target(self):
        """
        Build the target value vector.
        """
        self.target = np.zeros(len(self))

        i = 0
        for constraint in self.constraints:
            n = len(constraint)
            self.target[i : i + n] = constraint.target_value * np.ones(n)
            i += n

    def build_background(self):
        """
        Build the background value vector.
        """
        self.background = np.zeros(len(self))

        i = 0
        for constraint in self.constraints:
            n = len(constraint)
            self.background[i : i + n] = constraint.evaluate(self.eq)
            i += n

    @property
    def b(self):
        """
        The b vector of target - background values.
        """
        return self.target - self.background

    # def update_psi_boundary(self, psi_bndry):
    #     """
    #     Update the target value for all PsiBoundaryConstraints.

    #     Parameters
    #     ----------
    #     psi_bndry: float
    #         The target psi boundary value [V.s/rad]
    #     """
    #     for constraint in self.constraints:
    #         if isinstance(constraint, PsiBoundaryConstraint):
    #             constraint.target_value = psi_bndry
    #     self.build_target()

    def plot(self, ax=None):
        """
        Plots constraints
        """
        return ConstraintPlotter(self, ax=ax)

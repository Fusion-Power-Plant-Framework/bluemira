# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Equilibrium optimisation constraint functions.

Constraint functions must be of the form:

.. code-block:: python


    class Constraint(ConstraintFunction):

        def f_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return constraint_calc(vector)

        def df_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return gradient_calc(vector)


The constraint function convention is such that c <= 0 is sought. I.e. all constraint
values must be negative.

Note that the gradient (Jacobian) of the constraint function is of the form:

.. math::

    \\nabla \\mathbf{c} = \\begin{bmatrix}
            \\dfrac{\\partial c_{0}}{\\partial x_0} & \\dfrac{\\partial c_{0}}{\\partial x_1} & ... \n
            \\dfrac{\\partial c_{1}}{\\partial x_0} & \\dfrac{\\partial c_{1}}{\\partial x_1} & ... \n
            ... & ... & ... \n
            \\end{bmatrix}


If the `df_constraint` function is not provided, the constraint can still be used for
derivative-free optimisation algorithms, but will need to be updated or approximated for
use in derivative based algorithms, such as those utilising gradient descent.
"""  # noqa: W505, E501

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy._typing import NDArray
import numpy.typing as npt

from bluemira.base.look_and_feel import bluemira_warn

if TYPE_CHECKING:
    from bluemira.equilibria.equilibrium import Equilibrium


class ConstraintFunction(abc.ABC):
    """Override to define a numerical constraint for a coilset optimisation."""

    @property
    def name(self) -> str | None:
        """The name of the constraint"""
        try:
            return self._name
        except AttributeError:
            return None

    @name.setter
    def name(self, value: str | None):
        self._name = value

    @abc.abstractmethod
    def f_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The constraint function."""

    @property
    def constraint_type(self) -> Literal["inequality", "equality"]:
        """The type of constraint"""
        try:
            return self._constraint_type
        except AttributeError:
            return "inequality"

    @constraint_type.setter
    def constraint_type(self, constraint_t: Literal["inequality", "equality"]):
        if constraint_t not in {"inequality", "equality"}:
            bluemira_warn(
                f"Unknown nonlinear constraint type '{constraint_t}', "
                "defaulting to 'inequality'."
            )
        self._constraint_type = constraint_t


class AxBConstraint(ConstraintFunction):
    """
    Constraint function of the form:
        A.x - b < value

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
        a_mat: npt.NDArray[np.float64],
        b_vec: npt.NDArray[np.float64],
        value: float,
        scale: float,
        name: str | None = None,
    ):
        self.a_mat = a_mat
        self.b_vec = b_vec
        self.value = value
        self.scale = scale
        self.name = name

    def f_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Constraint function"""
        currents = self.scale * vector
        return self.a_mat @ currents - self.b_vec - self.value

    def df_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:  # noqa: ARG002
        """Constraint derivative"""
        return self.scale * self.a_mat


class L2NormConstraint(ConstraintFunction):
    """
    Constrain the L2 norm of an Ax = b system of equations.

    ||(Ax - b)||² < value

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
        a_mat: npt.NDArray[np.float64],
        b_vec: npt.NDArray[np.float64],
        value: float,
        scale: float,
        name: str | None = None,
    ):
        self.a_mat = a_mat
        self.b_vec = b_vec
        self.value = value
        self.scale = scale
        self.name = name

    def f_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Constraint function"""
        currents = self.scale * vector
        residual = self.a_mat @ currents - self.b_vec
        return residual.T @ residual - self.value

    def df_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Constraint derivative"""
        currents = self.scale * vector
        df = 2 * (self.a_mat.T @ self.a_mat @ currents - self.a_mat.T @ self.b_vec)
        return self.scale * df


class FieldConstraintFunction(ConstraintFunction):
    """
    Current optimisation poloidal field constraints at prescribed locations

    Parameters
    ----------
    ax_mat:
        Response matrix for Bx (active coil contributions)
    az_mat:
        Response matrix for Bz (active coil contributions)
    bxp_vec:
        Background vector for Bx (passive coil contributions)
    bzp_vec:
        Background vector for Bz (passive coil contributions)
    B_max:
        Maximum fields inside the coils
    scale:
        Current scale with which to calculate the constraints
    """

    def __init__(
        self,
        ax_mat: npt.NDArray[np.float64],
        az_mat: npt.NDArray[np.float64],
        bxp_vec: npt.NDArray[np.float64],
        bzp_vec: npt.NDArray[np.float64],
        B_max: npt.NDArray[np.float64],
        scale: float,
        name: str | None = None,
    ):
        self.ax_mat = ax_mat
        self.az_mat = az_mat
        self.bxp_vec = bxp_vec
        self.bzp_vec = bzp_vec
        self.B_max = B_max
        self.scale = scale
        self.name = name

    def f_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Constraint function"""
        currents = self.scale * vector

        Bx_a = self.ax_mat @ currents
        Bz_a = self.az_mat @ currents

        B = np.hypot(Bx_a + self.bxp_vec, Bz_a + self.bzp_vec)
        return B - self.B_max

    def df_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Constraint derivative"""
        currents = self.scale * vector

        Bx_a = self.ax_mat @ currents
        Bz_a = self.az_mat @ currents
        B = np.hypot(Bx_a + self.bxp_vec, Bz_a + self.bzp_vec)

        Bx = Bx_a * (Bx_a * currents + self.bxp_vec)
        Bz = Bz_a * (Bz_a * currents + self.bzp_vec)
        return (Bx + Bz) / (B * self.scale**2)


class CurrentMidplanceConstraint(ConstraintFunction):
    """
    Constraint function to constrain the inboard or outboard midplane
    of the plasma during optimisation.

    Parameters
    ----------
    eq:
        Equilibrium to use to fetch last closed flux surface from.
    radius:
        Toroidal radius at which to constrain the plasma midplane.
    scale:
        Current scale with which to calculate the constraints
    inboard:
        Boolean controlling whether to constrain the inboard (if True) or
        outboard (if False) side of the plasma midplane.
    """

    def __init__(
        self,
        eq: Equilibrium,
        radius: float,
        scale: float,
        *,
        inboard: bool,
        name: str | None = None,
    ):
        self.eq = eq
        self.radius = radius
        self.scale = scale
        self.inboard = inboard
        self.name = name

    def f_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Constraint function"""
        self.eq.coilset.get_control_coils().current = self.scale * vector
        lcfs = self.eq.get_LCFS()
        if self.inboard:
            return self.radius - min(lcfs.x)
        return max(lcfs.x) - self.radius


class CoilForceConstraintFunctions:
    """
    Constraint functions to constrain the force applied to the coils

    Parameters
    ----------
    a_mat:
        Response matrix
    b_vec:
        Target value vector
    n_PF:
        Number of PF coils
    n_CS:
        Number of CS coils
    scale:
        Current scale with which to calculate the constraints
    """

    def __init__(
            self,
            a_mat: npt.NDArray[np.float64],
            b_vec: npt.NDArray[np.float64],
            n_PF: int,
            n_CS: int,
            scale: float,
            ):
        self.a_mat = a_mat
        self.b_vec = b_vec
        self.n_PF = n_PF
        self.n_CS = n_CS
        self.scale = scale
        if self.n_CS == 0 and self.n_PF == 0:
            raise ValueError(
                "n_PF and n_CS are both 0. Make sure the coils in the coilset "
                "have the correct ctype set."
            )
        self.n_coils = self.n_CS + self.n_PF
        self._constraint = np.zeros(self.n_coils)
        self._grad = np.zeros((self.n_coils, self.n_coils))

    @property
    def constraint(self):
        """Constraint"""
        return self._constraint

    @constraint.setter
    def constraint(self, value):
        self._constraint = value

    @property
    def grad(self):
        """Constraint Gradient"""
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    def calc_f_matx(self, currents):
        """Force"""
        F = np.zeros((self.n_coils, 2))
        for i in range(2):  # coil force
            # NOTE: * Hadamard matrix product
            F[:, i] = currents * (self.a_mat[:, :, i] @ currents + self.b_vec[:, i])
        return F / self.scale  # Scale down to MN

    def calc_df_matx(self, currents):
        """Jacobian"""
        dF = np.zeros((self.n_coils, self.n_coils, 2))
        im = currents.reshape(-1, 1) @ np.ones((1, self.n_coils))  # current matrix
        for i in range(2):
            dF[:, :, i] = im * self.a_mat[:, :, i]
            diag = (
                self.a_mat[:, :, i] @ currents
                + currents * np.diag(self.a_mat[:, :, i])
                + self.b_vec[:, i]
            )
            np.fill_diagonal(dF[:, :, i], diag)
        return dF

    def cs_fz(self, f_matx):
        """Vertical forces on CS coils."""
        return f_matx[self.n_PF :, 1]

    def pf_z_constraint(self, f_matx, max_value):
        """Constraint Function: Absolute vertical force constraint on PF coils."""
        scaled_max_value = max_value / self.scale
        self.constraint[: self.n_PF] = f_matx[: self.n_PF, 1] ** 2 - scaled_max_value**2

    def pf_z_constraint_grad(self, df_matx):
        """Constraint Derivative: Absolute vertical force constraint on PF coils."""
        self.grad[: self.n_PF] = 2 * df_matx[: self.n_PF, :, 1]

    def cs_z_constraint(self, f_matx, max_value):
        """
        Constraint Function:
        Absolute sum of vertical force constraint on entire CS stack.
        """
        scaled_max_value = max_value / self.scale
        # vertical force on CS stack
        cs_z_sum = np.sum(self.cs_fz(f_matx))
        self.constraint[self.n_PF] = cs_z_sum**2 - scaled_max_value**2

    def cs_z_grad(self, df_matx):
        """
        Constraint Derivative:
        Absolute sum of vertical force constraint on entire CS stack
        """
        self.grad[self.n_PF] = 2 * np.sum(df_matx[self.n_PF :, :, 1], axis=0)

    def cs_z_sep_constraint(self, f_matx, max_value):
        """Constraint Function: CS separation constraints."""
        scaled_max_value = max_value / self.scale
        cs_fz = self.cs_fz(f_matx)
        for i in range(self.n_CS - 1):  # evaluate each gap in CS stack
            f_sep = np.sum(cs_fz[: i + 1]) - np.sum(cs_fz[i + 1 :])
            self.constraint[self.n_PF + 1 + i] = f_sep - scaled_max_value

    def cs_z_sep_grad(self, df_matx):
        """Constraint Derivative: CS separation constraints."""
        for i in range(self.n_CS - 1):  # evaluate each gap in CS stack
            # CS separation constraint Jacobians
            f_up = np.sum(df_matx[self.n_PF : self.n_PF + i + 1, :, 1], axis=0)
            f_down = np.sum(df_matx[self.n_PF + i + 1 :, :, 1], axis=0)
            self.grad[self.n_PF + 1 + i] = f_up - f_down


class CoilForceConstraint(ConstraintFunction, CoilForceConstraintFunctions):
    """
    Combined onstraint function for:
        - PF vertical force,
        - CS vertical force,
        - and CS separation.

    Parameters
    ----------
    a_mat:
        Response matrix
    b_vec:
        Target value vector
    n_PF:
        Number of PF coils
    n_CS:
        Number of CS coils
    PF_Fz_max:
        The maximum force in the z direction for a PF coil
    CS_Fz_sum_max:
        The total maximum force in the z direction for the CS coils
    CS_Fz_sep_max:
        The individual maximum force in the z direction for the CS coils
    scale:
        Current scale with which to calculate the constraints
    """

    def __init__(
        self,
        a_mat: npt.NDArray[np.float64],
        b_vec: npt.NDArray[np.float64],
        n_PF: int,
        n_CS: int,
        PF_Fz_max: float,
        CS_Fz_sum_max: float,
        CS_Fz_sep_max: float,
        scale: float,
        name: str | None = None
        ):
        super().__init__(a_mat, b_vec, n_PF, n_CS, scale)
        self.PF_Fz_max = PF_Fz_max
        self.CS_Fz_sum_max = CS_Fz_sum_max
        self.CS_Fz_sep_max = CS_Fz_sep_max
        self.name = name

    def f_constraint(self, vector):
        """Constraint function"""
        currents = self.scale * vector
        f_matx = self.calc_f_matx(currents)
        self.pf_z_constraint(f_matx, self.PF_Fz_max)
        if self.n_CS != 0:
            self.cs_z_constraint(f_matx, self.CS_Fz_sum_max)
            self.cs_z_sep_constraint(f_matx, self.CS_Fz_sep_max)
        return self.constraint

    def df_constraint(self, vector):
        """Constraint derivative"""
        currents = self.scale * vector
        df_matx = self.calc_df_matx(currents)
        self.pf_z_constraint_grad(df_matx)
        if self.n_CS != 0:
            self.cs_z_grad(df_matx)
            self.cs_z_sep_grad(df_matx)
        return self.grad

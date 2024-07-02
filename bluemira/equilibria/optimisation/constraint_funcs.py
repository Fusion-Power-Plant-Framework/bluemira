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

if TYPE_CHECKING:
    from bluemira.equilibria.grid import Grid
    from bluemira.geometry.coordinates import Coordinates
    from bluemira.geometry.wire import BluemiraWire

import numpy as np
import numpy.typing as npt

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.constants import PSI_NORM_TOL
from bluemira.equilibria.find_legs import LegFlux, NumNull, get_legs_length_and_angle

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


class GrazingAngleConstraintFunction(ConstraintFunction):
    """
    Constraint Function for divertor leg grazing angles.

    Parameters
    ----------
    eq:
        Equilibrium to use to fetch last closed flux surface from.
    scale:
        Current scale with which to calculate the constraints
    min_angles:
        The required minimum angles

    """

    def __init__(
        self,
        eq: Equilibrium,
        scale: float,
        min_angles: npt.NDArray,
        double_null: bool,
        psi_n_tol: float = PSI_NORM_TOL,
        plasma_facing_boundary: Grid | Coordinates | BluemiraWire | None = None,
    ) -> None:
        self.eq = eq
        self.scale = scale
        self.min_angles = min_angles
        self.double_null = double_null
        self.psi_n_tol = psi_n_tol
        self.plasma_facing_boundary = plasma_facing_boundary

    def f_constraint(self, vector: npt.NDArray) -> npt.NDArray:
        """Constraint function"""
        self.eq.coilset.get_control_coils().current = vector * self.scale

        legs = LegFlux(self.eq).get_legs(delta=self.delta_start)
        _lengths, angles = get_legs_length_and_angle(
            self.eq, legs, self.plasma_facing_boundary
        )

        if LegFlux(self.eq).n_null == NumNull.SN:
            location = "lower" if any("lower" in name for name in angles) else "upper"
            return (
                np.array(
                    angles[f"{location}_inner"][0],
                    angles[f"{location}_outer"][0],
                )
                - self.min_angles
            )

        return (
            np.array(
                angles["lower_inner"][0],
                angles["lower_outer"][0],
                angles["upper_inner"][0],
                angles["upper_outer"][0],
            )
            - self.min_angles
        )


class CoilForceConstraint(ConstraintFunction):
    """
    Constraint function to constrain the force applied to the coils

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
        name: str | None = None,
    ):
        self.a_mat = a_mat
        self.b_vec = b_vec
        self.n_PF = n_PF
        self.n_CS = n_CS
        self.PF_Fz_max = PF_Fz_max
        self.CS_Fz_sum_max = CS_Fz_sum_max
        self.CS_Fz_sep_max = CS_Fz_sep_max
        self.scale = scale
        self.name = name

        if self.n_CS == 0 and self.n_PF == 0:
            raise ValueError(
                "n_PF and n_CS are both 0. Make sure the coils in the coilset "
                "have the correct ctype set."
            )

    def f_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Constraint function"""
        currents = self.scale * vector

        n_coils = self.n_CS + self.n_PF
        constraint = np.zeros(n_coils)

        # get coil force and jacobian
        F = np.zeros((n_coils, 2))
        PF_Fz_max = self.PF_Fz_max / self.scale
        CS_Fz_sep_max = self.CS_Fz_sep_max / self.scale
        CS_Fz_sum_max = self.CS_Fz_sum_max / self.scale

        for i in range(2):  # coil force
            # NOTE: * Hadamard matrix product
            F[:, i] = currents * (self.a_mat[:, :, i] @ currents + self.b_vec[:, i])

        F /= self.scale  # Scale down to MN

        # Absolute vertical force constraint on PF coils
        constraint[: self.n_PF] = F[: self.n_PF, 1] ** 2 - PF_Fz_max**2

        if self.n_CS != 0:
            # vertical forces on CS coils
            cs_fz = F[self.n_PF :, 1]
            # vertical force on CS stack
            cs_z_sum = np.sum(cs_fz)
            # Absolute sum of vertical force constraint on entire CS stack
            constraint[self.n_PF] = cs_z_sum**2 - CS_Fz_sum_max**2
            for i in range(self.n_CS - 1):  # evaluate each gap in CS stack
                # CS separation constraints
                f_sep = np.sum(cs_fz[: i + 1]) - np.sum(cs_fz[i + 1 :])
                constraint[self.n_PF + 1 + i] = f_sep - CS_Fz_sep_max
        return constraint

    def df_constraint(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Constraint derivative"""
        currents = self.scale * vector

        n_coils = self.n_CS + self.n_PF
        grad = np.zeros((n_coils, n_coils))
        dF = np.zeros((n_coils, n_coils, 2))  # noqa: N806

        im = currents.reshape(-1, 1) @ np.ones((1, n_coils))  # current matrix
        for i in range(2):
            dF[:, :, i] = im * self.a_mat[:, :, i]
            diag = (
                self.a_mat[:, :, i] @ currents
                + currents * np.diag(self.a_mat[:, :, i])
                + self.b_vec[:, i]
            )
            np.fill_diagonal(dF[:, :, i], diag)

        # Absolute vertical force constraint on PF coils
        grad[: self.n_PF] = 2 * dF[: self.n_PF, :, 1]

        if self.n_CS != 0:
            # Absolute sum of vertical force constraint on entire CS stack
            grad[self.n_PF] = 2 * np.sum(dF[self.n_PF :, :, 1], axis=0)

            for i in range(self.n_CS - 1):  # evaluate each gap in CS stack
                # CS separation constraint Jacobians
                f_up = np.sum(dF[self.n_PF : self.n_PF + i + 1, :, 1], axis=0)
                f_down = np.sum(dF[self.n_PF + i + 1 :, :, 1], axis=0)
                grad[self.n_PF + 1 + i] = f_up - f_down

        return grad

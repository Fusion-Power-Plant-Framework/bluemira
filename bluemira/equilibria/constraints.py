# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
Plasma magnetic constraint objects and auto-generation tools
"""
from abc import ABC, abstractmethod
from typing import List, Union
from dataclasses import dataclass
import numpy as np
from copy import deepcopy
from bluemira.equilibria.plotting import ConstraintPlotter

__all__ = ["MagneticConstraintSet"]


class MagneticConstraint(ABC):
    """
    Abstract base class for magnetic constraints.
    """

    target_value: float

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


@dataclass
class AbsoluteMagneticConstraint(MagneticConstraint):
    """
    Abstract base class for absolute magnetic constraints, where the target
    value is prescribed in absolute terms.
    """

    x: Union[float, np.array]
    z: Union[float, np.array]
    target_value: float
    weight = NotImplemented  # TODO: address weights in future MR


@dataclass
class RelativeMagneticConstraint(MagneticConstraint):
    """
    Abstract base class for relative magnetic constraints, where the target
    value is prescribed with respect to a reference point.
    """

    x: Union[float, np.array]
    z: Union[float, np.array]
    ref_x: float
    ref_z: float
    target_value: float = 0.0
    weight = NotImplemented  # TODO: address weights in future MR

    @abstractmethod
    def update(self, eq):
        """
        We need to update the target value, as it is a relative constraint.
        """
        pass


@dataclass
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


@dataclass
class BzConstraint(AbsoluteMagneticConstraint):
    """
    Absolute Bz value constraint.
    """

    def control_response(self, coilset):
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return coilset.control_Bz(self.x, self.z)

    def evaluate(self, eq):
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return eq.Bz(self.x, self.z)

    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """
        kwargs = {"marker": 10, "markersize": 15, "color": "b", "zorder": 45}
        ax.plot(self.x, self.z, "s", **kwargs)


@dataclass
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


@dataclass
class PsiConstraint(AbsoluteMagneticConstraint):
    """
    Absolute psi value constraint.
    """

    def control_response(self, coilset):
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return coilset.control_psi(self.x, self.z)

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


@dataclass
class FieldNullConstraint(AbsoluteMagneticConstraint):
    """
    Magnetic field null constraint. In practice sets the Bx and Bz field components
    to be 0 at the specified location.
    """

    target_value: float = 0.0

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


@dataclass
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

    def update(self, eq):
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
        - Populate constraints: List[MagneticConstraint]
    """

    constraints: List[MagneticConstraint]
    eq: object
    A: np.array
    target: np.array
    background: np.array

    __slots__ = ["constraints", "eq", "coilset", "A", "target", "background"]

    def __call__(self, equilibrium, I_not_dI=False, fixed_coils=False):  # noqa (N803)

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

        self.eq = equilibrium
        self.coilset = equilibrium.coilset

        # Update relative magnetic constraints without updating A matrix
        for constraint in self.constraints:
            if isinstance(constraint, RelativeMagneticConstraint):
                constraint.update(equilibrium)

        # Re-build control response matrix
        if not fixed_coils or (fixed_coils and self.A is None):
            self.build_control_matrix()
            self.build_target()

        self.build_background()

    def __len__(self):
        """
        The mathematical size of the constraint set.
        """
        return sum([len(c) for c in self.constraints])

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

    def update_psi_boundary(self, psi_bndry):
        """
        Update the target value for all PsiBoundaryConstraints.

        Parameters
        ----------
        psi_bndry: float
            The target psi boundary value [V.s/rad]
        """
        for constraint in self.constraints:
            if isinstance(constraint, PsiBoundaryConstraint):
                constraint.target_value = psi_bndry
        self.build_target()

    def copy(self):
        """
        Get a deep copy of the MagneticConstraintSet instance.
        """
        return deepcopy(self)

    def plot(self, ax=None):
        """
        Plots constraints
        """
        return ConstraintPlotter(self, ax=ax)

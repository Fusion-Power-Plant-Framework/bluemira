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
Plasma position and field constraint objects and auto-generation tools
"""
from abc import ABC, abstractmethod
from typing import List, Union
from dataclasses import dataclass
import numpy as np
from copy import deepcopy
from BLUEPRINT.equilibria.plotting import ConstraintPlotter
from BLUEPRINT.equilibria.shapes import flux_surface_johner
from BLUEPRINT.geometry.loop import Loop


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

    Notes
    -----
    The optional weights parameter is used to scale the control matrix and
    constraint vector passed to the constraint optimiser for a given
    constraint. The weights are assumed to be uncorrelated, such that the
    weight matrix W_ij used to define the least-squares objective function
    to be minimised, (Ax - b)ᵀ W (Ax - b), is diagonal, such that
    weights[i] = w[i] = sqrt(W[i,i]).
    """

    x: Union[float, np.array]
    z: Union[float, np.array]
    target_value: float
    weights: Union[float, np.array] = 1.0


@dataclass
class RelativeMagneticConstraint(MagneticConstraint):
    """
    Abstract base class for relative magnetic constraints, where the target
    value is prescribed with respect to a reference point.

    Notes
    -----
    The optional weights parameter is used to scale the control matrix and
    constraint vector passed to the constraint optimiser for a given
    constraint. The weights are assumed to be uncorrelated, such that the
    weight matrix W_ij used to define the least-squares objective function
    to be minimised, (Ax - b)ᵀ W (Ax - b), is diagonal, such that
    weights[i] = w[i] = sqrt(W[i,i]).
    """

    x: Union[float, np.array]
    z: Union[float, np.array]
    ref_x: float
    ref_z: float
    target_value: float = 0.0
    weights: Union[float, np.array] = 1.0

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
    w: np.array
    target: np.array
    background: np.array

    __slots__ = ["constraints", "eq", "coilset", "A", "w", "target", "background"]

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
        self.build_weight_matrix()

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

    def build_target(self):
        """
        Build the target value vector.
        """
        self.target = np.zeros(len(self))

        i = 0
        for constraint in self.constraints:
            n = len(constraint)
            self.target[i : i + n] = constraint.target_value
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

    def plot(self, ax=None, **kwargs):
        """
        Plots constraints
        """
        return ConstraintPlotter(self, ax=ax, **kwargs)


class SNReference(MagneticConstraintSet):
    """
    Parameterised family of magnetic constraints for a typical EU-DEMO-like single
    null equilibrium.
    """

    def __init__(self, R_0, Z_0, A, kappa, delta, psibval, n=100, upper=True):
        self.constraints = []
        f_s = flux_surface_johner(
            R_0,
            Z_0,
            R_0 / A,
            kappa,
            1.2 * kappa,  # parámetro mágico!
            delta,
            1.2 * delta,  # parámetro mágico!
            -20,
            5,
            60,
            30,  # ángulos mágicos!
            n=200,
            upper=upper,
        )

        if upper:
            arg_x = np.argmin(f_s.z)
        else:
            arg_x = np.argmax(f_s.z)

        x_point = [f_s.x[arg_x], f_s.z[arg_x]]

        self.constraints = [FieldNullConstraint(*x_point)]

        f_s.interpolate(n)
        x_s, z_s = f_s.x, f_s.z

        self.constraints.append(PsiBoundaryConstraint(x_s, z_s, psibval))

        x_leg1, z_leg1 = self.calc_divertor_leg(
            x_point, 50, 1.45, int(n / 10), loc="lower", pos="outer"
        )

        x_leg2, z_leg2 = self.calc_divertor_leg(
            x_point, 40, 1, int(n / 10), loc="lower", pos="inner"
        )

        x_legs = np.append(x_leg1, x_leg2)
        z_legs = np.append(z_leg1, z_leg2)
        self.constraints.append(PsiBoundaryConstraint(x_legs, z_legs, psibval))

    @staticmethod
    def calc_line(p1, p2, n):
        """
        Calculate a linearly spaced series of points on a line between p1 and p2.
        """
        xn = np.linspace(p1[0], p2[0], int(n))
        zn = np.linspace(p1[1], p2[1], int(n))
        return xn, zn

    def calc_divertor_leg(
        self, x_point, angle, length, n, loc="lower", pos="outer"
    ):  # noqa (N802)
        """
        Calculate the position of a straight line divertor leg.
        """
        if loc == "upper":
            z = x_point[1] + length * np.sin(np.deg2rad(angle))
        elif loc == "lower":
            z = x_point[1] - length * np.sin(np.deg2rad(angle))
        else:
            raise ValueError('Please specify loc: "upper" or "lower" X-point.')
        if pos == "inner":
            x = x_point[0] - length * np.cos(np.deg2rad(angle))
        elif pos == "outer":
            x = x_point[0] + length * np.cos(np.deg2rad(angle))
        else:
            raise ValueError('Please specify pos: "inner" or "outer" X leg.')
        return self.calc_line(x_point, (x, z), n)


class DNReference(MagneticConstraintSet):
    """
    Parameterised family of magnetic constraints for a typical EU-DEMO-like double
    null equilibrium.
    """

    def __init__(self, R_0, Z_0, A, kappa, delta, psibval, n=400):
        super().__init__()
        f_s = flux_surface_johner(
            R_0,
            Z_0,
            R_0 / A,
            kappa,
            kappa,  # parámetro mágico!
            delta,
            delta,  # parámetro mágico!
            50,
            30,
            50,
            30,  # ángulos mágicos!
            n=200,
        )

        arg_xl = np.argmin(f_s.z)
        arg_xu = np.argmax(f_s.z)
        self.constraints = [
            FieldNullConstraint(f_s.x[arg_xl], f_s.z[arg_xl]),
            FieldNullConstraint(f_s.x[arg_xu], f_s.z[arg_xu]),
        ]
        f_s.interpolate(n)
        x_s, z_s = f_s.x, f_s.z

        self.constraints.append(PsiBoundaryConstraint(x_s, z_s, psibval))


class AutoConstraints(MagneticConstraintSet):
    """
    An object to auto-generate a set of constraints from an equilibrium.
    """

    def __init__(self, eq, npoints=100):
        super().__init__()

        self.constraints = []
        if eq._eqdsk:
            loop = Loop(x=eq._eqdsk["xbdry"], z=eq._eqdsk["zbdry"])
            psi = eq._eqdsk["psibdry"]
            argx = np.argmin(loop.z)
            self.constraints.append(FieldNullConstraint(loop.x[argx], loop.z[argx]))
            loop.interpolate(npoints)
            x, z = loop.x, loop.z
        else:
            lcfs = eq.get_LCFS()
            lcfs.interpolate(npoints)
            x, z = lcfs.x, lcfs.z
            psi = float(eq.psi(x[0], z[0]))

        self.constraints.append(PsiBoundaryConstraint(x, z, psi))


class XzTesting(MagneticConstraintSet):
    """
    Pure x-z psi constraints only. Useful in testing.
    """

    def __init__(self, x, z, psival, n=40):
        super().__init__()
        arg_x = np.argmin(z)
        self.constraints = [FieldNullConstraint(x[arg_x], z[arg_x])]
        s = Loop(x, 0, z)
        s.interpolate(n)
        x, z = s.x, s.z

        self.constraints.append(PsiBoundaryConstraint(x, z, psival))


if __name__ == "__main__":
    from BLUEPRINT import test

    test()

# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

"""
Plasma position and field constraint objects and auto-generation tools
"""
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d, RectBivariateSpline
from BLUEPRINT.base.error import EquilibriaError
from BLUEPRINT.equilibria.plotting import ConstraintPlotter
from BLUEPRINT.equilibria.shapes import flux_surface_manickam, flux_surface_johner
from BLUEPRINT.equilibria.find import get_psi
from BLUEPRINT.base.lookandfeel import bpwarn
from BLUEPRINT.geometry.geomtools import (
    anticlock,
    tangent,
    vector_intersect,
)
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.utilities.tools import lengthnorm


class ConstraintBuilder:
    """
    Constraints object which is called on a equilibrium
    Generate the [G] matrix and [T] vector

    Collection of methods to build constraints

    Use of class:
        - Inherit from this class
        - Add a __init__(args) method
        - Add a __call__(EqObject) method and use the super()
    """

    def __call__(self, equilibrium):
        """
        Applies constraints to EqObject. Must be overriden in inherited classes
        and called with super()
        """
        self.fix = {
            "x": np.array([]),
            "z": np.array([]),
            "BC": np.array([]),
            "value": np.array([]),  # [b] vector
            "Bdir": np.array([[], []]).T,
            "factor": np.array([]),
            "n": 0,
        }
        self.eq = equilibrium
        self.coilset = equilibrium.coilset

    def build_A(self):  # noqa (N802)
        """
        Builds up the [G] matrix from the coil control response
        [G][I] = [T-BG]
        """
        self.G = np.zeros((len(self.fix["BC"]), len(self.coilset._ccoils)))
        x_fix, z_fix, b_conditions, bdir, val = self.unpack_fix()
        for i, (xf, zf, bc, bdir, v) in enumerate(
            zip(x_fix, z_fix, b_conditions, bdir, val)
        ):
            if "psi" in bc:
                c_i2 = np.array(self.coilset.control_psi(xf, zf))
                if bc == "psi":  # Absolute value psi constraint
                    self.G[i, :] = c_i2
                elif bc == "psi_bndry":  # Relative psi constraint
                    self.G[i, :] = c_i2 - self.Iref
            else:
                for j, coil in enumerate(self.coilset._ccoils):
                    gx = coil.control_Bx(xf, zf)
                    gz = coil.control_Bz(xf, zf)
                    self.G[i, j] = self.field(bc, gx, gz, bdir)

    def build_b(self):
        """
        Build the [BG] vector for the passive plasma response.

        Notes
        -----
        If current gradient optimisation is used, the background values are
        from the entire equilibrium. If current optimisation is used, the
        self.eq object then only considers the plasma contribution. This is
        handled in __call__.
        """
        self.BG = np.zeros(len(self.fix["BC"]))  # background

        x_fix, z_fix, b_conditions, bdir, _ = self.unpack_fix()
        for i, (xf, zf, bc, bdir) in enumerate(zip(x_fix, z_fix, b_conditions, bdir)):
            if bc in ["psi", "psi_bndry"]:
                self.BG[i] = self.eq.psi(xf, zf)
            else:
                Bx = self.eq.Bx(xf, zf)
                Bz = self.eq.Bz(xf, zf)
                self.BG[i] = self.field(bc, Bx, Bz, bdir)

    def get_weights_new(self):
        """
        Get the weights of the constraints.
        """
        self.generate_interpolators()  # To enable gradient calculations
        x_fix, z_fix, b_conditions, bdir, val = self.unpack_fix()
        w = np.zeros(self.fix["n"])
        if self.fix["n"] > 0:
            for i, (x, z, bc) in enumerate(zip(x_fix, z_fix, b_conditions)):
                if "psi" in bc:
                    w[i] = 1 / self.Bp_spline(x, z) ** 2
                else:
                    w[i] = 1
        return w

    def get_bdir(self, x, z):
        """
        Generates the direction vectors of the desired flux surface shape
        """
        nx = -self.psi_spline.ev(x, z, dx=1, dy=0)
        nz = -self.psi_spline.ev(x, z, dx=0, dy=1)
        length_norm = lengthnorm(x, z)
        l_c = np.linspace(0, 1, len(x) + 1)[:-1]
        b_dir = np.array(
            [interp1d(length_norm, nx)(l_c), interp1d(length_norm, nz)(l_c)]
        ).T
        return b_dir

    def get_weights(self):
        """
        Returns the weight vector
        """
        self.generate_interpolators()  # To enable gradient calculations
        x_fix, z_fix, b_conditions, b_dirs, val = self.unpack_fix()
        x, z = [], []
        idx = []
        for i, bc in enumerate(b_conditions):
            if "psi" in bc:
                x.append(x_fix[i])
                z.append(z_fix[i])
                idx.append(i)
        bdir = self.get_bdir(x, z)
        b_dirs[idx] = bdir

        n = self.fix["n"]
        weight = np.zeros(n)
        if n > 0:
            for i, (xf, zf, bc, bdir, factor) in enumerate(
                zip(x_fix, z_fix, b_conditions, b_dirs, self.fix["factor"])
            ):
                d_dx, d_dz = self.get_gradients(bc, xf, zf)
                if "psi" not in bc:  # (Bx,Bz)
                    weight[i] = 1  # /abs(np.sqrt(d_dx**2+d_dz**2))
                elif "psi" in bc:
                    weight[i] = 1 / abs(
                        np.dot([d_dx, d_dz], bdir)
                    )  # /abs(np.dot([d_dx, d_dz], bdir))
            # ==========================================================================
            #             if 'psi_bndry' in self.fix['BC']:
            #                 wbar = np.mean([weight[i]
            #                                 for i, bc in enumerate(self.fix['BC'])
            #                                 if bc == 'psi_bndry'])
            #             else:
            #                 wbar = np.mean(weight)
            #             for i, bc in enumerate(BC):
            #                 if bc == 'psi_x' or bc == 'psi':  # psi point weights
            #                     weight[i] = 1/abs(np.sqrt(d_dx**2+d_dz**2))#wbar
            # ==========================================================================
            if (weight == 0).any():
                bpwarn("Unweighted constraint target value! (== 0)")
        return weight

    def generate_interpolators(self):
        """
        Sets up spline interpolations for psi and Bp (including coils). Used
        to calculate local gradients
        """
        # NOTE: Separation of terms to enable I and dI optimisations
        # (For the former a PlasmaCoil is passed as the EqObject)
        x, z = self.eq.grid.x, self.eq.grid.z
        psi = self.eq.plasma_psi + self.coilset.psi(x, z)
        Bp = self.eq.plasma_Bp + self.coilset.Bp(x, z)
        self.psi_spline = RectBivariateSpline(x[:, 0], z[0, :], psi)
        self.Bp_spline = RectBivariateSpline(x[:, 0], z[0, :], Bp)

    def get_gradients(self, bc, xf, zf):
        """
        Get the gradients of the boundary conditions
        """
        try:
            if "psi" in bc:
                d_dx = self.psi_spline.ev(xf, zf, dx=1, dy=0)
                d_dz = self.psi_spline.ev(xf, zf, dx=0, dy=1)
            else:
                d_dx = self.Bp_spline.ev(xf, zf, dx=1, dy=0)
                d_dz = self.Bp_spline.ev(xf, zf, dx=0, dy=1)
        except ValueError:
            bpwarn("Field gradient evaluation failed.")
            raise EquilibriaError("Field gradient evaluation failed.")
            # d_dx, d_dz = np.ones(len(bc)), np.ones(len(bc))

        return d_dx, d_dz

    def swing_flux(self, flux):
        """
        Adjusts [b] vector for flux
        """
        # NOTE: Relies on __call__(*args, **kwargs) beauty
        self.psival = flux

    def field(self, bc, Bx, Bz, b_dir):
        """
        Calculate the field at the boundary condition.
        """
        if bc == "Bx":
            value = Bx
        elif bc == "Bz":
            value = Bz
        elif bc == "null":
            value = np.sqrt(Bx ** 2 + Bz ** 2)
        elif bc == "Bdir":
            nhat = b_dir / np.sqrt(b_dir[0] ** 2 + b_dir[1] ** 2)
            value = np.dot([Bx, Bz], nhat)
        elif bc == "Bp":
            value = Bx - Bz / 2
        return value

    def _add_c(self, x, z, bc, value, b_dir, factor):
        """
        Parent method for adding a constraint to the `fix` construct
        """
        var = {"x": x, "z": z, "BC": bc, "value": value, "Bdir": b_dir, "factor": factor}
        nvar = len(x)
        self.fix["n"] += nvar
        for name in ["value", "Bdir", "BC", "factor"]:
            if np.shape(var[name])[0] != nvar:  # Reshape input into nvar-array
                var[name] = np.array([var[name]] * nvar)
        for name in var.keys():
            if name == "Bdir":
                for i in range(nvar):
                    norm = np.sqrt(var[name][i][0] ** 2 + var[name][i][1] ** 2)
                    if norm != 0:
                        var[name][i] /= norm  # normalise tangent vectors
                self.fix[name] = np.append(self.fix[name], var[name], axis=0)
            else:
                self.fix[name] = np.append(self.fix[name], var[name])

    def add_X_point(self, x, z, factor=1):  # noqa (N802)
        """
        Adds an X-point constraint
        """
        self.add_Bxo(x, z, factor=factor)
        self.add_Bzo(x, z, factor=factor)

    def add_Bxval(self, x, z, val, factor=1):  # noqa (N802)
        """
        Sets a constraint on the radial field at (x, z)
        Bx(x, z) = val
        """
        self._add_c([x], [z], ["Bx"], [val], np.array([[1.0], [0.0]]).T, [factor])

    def add_Bzval(self, x, z, val, factor=1):  # noqa (N802)
        """
        Sets a constraint on the radial field at (x, z)
        Bz(x, z) = val
        """
        self._add_c([x], [z], ["Bz"], [val], np.array([[0.0], [1.0]]).T, [factor])

    def add_Bpval(self, x, z, val, factor=1):  # noqa (N802)
        """
        Sets a constraint on the poloidal field at (x, z)
        Bp(x, z) = val
        """
        self._add_c([x], [z], ["Bp"], [val], np.array([[0.0], [1.0]]).T, [factor])

    def add_Bxo(self, x, z, factor=1):  # noqa (N802)
        """
        Sets a constraint on the radial field at (x, z)
        Bx(x, z) = 0
        """
        self.add_Bxval(x, z, 0, factor=factor)

    def add_Bzo(self, x, z, factor=1):  # noqa (N802)
        """
        Sets a constraint on the vertical field at (x, z)
        Bz(x, z) = 0
        """
        self.add_Bzval(x, z, 0, factor=factor)

    def add_isoflux(self, x, z, psiref, b_dir=None, factor=1):
        """
        Sets an isoflux constraint at (x, z) with respect to a reference point
        with self.psiref
        psi(x, z) = psi_ref
        """
        if b_dir is None:
            b_dir = np.array([0, 0])
        self._add_c([x], [z], ["psi_bndry"], [psiref], b_dir, [factor])

    def add_psival(self, x, z, psival, b_dir=None, factor=1, **kwargs):
        """
        Sets a psi constraint at (x, z)
        psi(x, z) = psi_val
        """
        if b_dir is None:
            b_dir = np.array([0, 0])
        self._add_c([x], [z], ["psi"], [psival], b_dir, [factor])

    def add_psinval(self, x, z, psin, b_dir=None, factor=1, **kwargs):
        """
        Sets a normalised psi constraint at (x, z)
        psi(x, z) = psi_n*self.psiref
        """
        psival = psin * self.psiref
        self.add_psival(x, z, psival, b_dir=b_dir, factor=factor, **kwargs)

    def add_Bdir(self, x, z, b_dir, factor=1, **kwargs):  # noqa (N802)
        """
        Sets a constraint on the direction of the field
        """
        if len(b_dir) == 1:  # normal angle from horizontal in degrees
            arg = b_dir[0]
            b_dir = [-np.sin(arg * np.pi / 180), np.cos(arg * np.pi / 180)]
        b_dir /= np.sqrt(b_dir[0] ** 2 + b_dir[1] ** 2)
        self._add_c([x], [z], ["Bdir"], [0], b_dir, [factor])

    def unpack_fix(self):
        """
        Unpack the fix tuple.
        """
        xf, zf = self.fix["x"], self.fix["z"]
        bc, bdir = self.fix["BC"], self.fix["Bdir"]
        return xf, zf, bc, bdir, self.fix["value"]

    def plot(self, ax=None, **kwargs):
        """
        Plots constraints
        """
        return ConstraintPlotter(self, ax=ax, **kwargs)


class XpointConstraint:
    """
    Abstract object for an X-point constraint.
    Se o punto X e ativo, aplique uma regla de isoflux tambem
    """

    def __init__(self, x, z, active=True, loc="lower"):
        self.x = x
        self.z = z
        self.active = active
        self.loc = loc

    def __iter__(self):
        """
        Imbue XpointConstraint with generator-like behaviour.
        """
        yield self.x
        yield self.z

    def __getitem__(self, i):
        """
        Imbue XpointConstraint with list-like behaviour.
        """
        if i == 0:
            return self.x
        elif i == 1:
            return self.z


class XlegConstraint:
    """
    Abstract object for an X-leg constraint.
    """

    def __init__(self, x, z, loc="inner", pos="outer"):
        self.x = x
        self.z = z
        self.n = len(x)
        self.loc = loc
        self.pos = pos

    def __iter__(self):
        """
        Do nothing if treated like a generator.
        """
        pass


class BdirConstraint:
    """
    Field direction constraint.
    """

    def __init__(self, x, z, b_dir):
        self.x, self.z = x, z
        self.Bdir = b_dir


class BxConstraint:
    """
    X-field constraint.
    """

    def __init__(self, x, z, Bx):
        self.x, self.z = x, z
        self.Bx = Bx


class BzConstraint:
    """
    Z-field constraint.
    """

    def __init__(self, x, z, Bz):
        self.x, self.z = x, z
        self.Bz = Bz


class BpConstraint:
    """
    Poloidal field constraint.
    """

    def __init__(self, x, z, Bp):
        self.x, self.z = x, z
        self.Bp = Bp


class IsofluxConstraint:
    """
    Isoflux constraint.
    """

    def __init__(self, x, z, refpoint):
        self.x, self.z = x, z
        self.refpoint = refpoint


class PsinormConstraint:
    """
    Normalised magnetic flux per radian constraint.
    """

    def __init__(self, x, z, psi_norm):
        self.x, self.z = x, z
        self.psi_norm = psi_norm


class PsiConstraint:
    """
    Magnetic flux per radian constraint.
    """

    __slots__ = ["x", "z", "psi", "Bdir"]

    def __init__(self, x, z, psi, b_dir=None):
        self.x, self.z = x, z
        self.psi = psi
        if b_dir is None:
            self.Bdir = np.array([0, 0])
        else:
            self.Bdir = b_dir


class ConstraintCalculator:
    """
    Method mixin class
    """

    def calc_X_point(self, loc="lower", active=True):  # noqa (N802)
        """
        Calcule a posicao do X-point
        """
        ni = int(self.n / 12)
        if loc == "upper":
            ix = np.argmax(self.Z)
            if not active:
                j = 1
            else:
                j = -1
        elif loc == "lower":
            ix = np.argmin(self.Z)
            j = -1
        else:
            raise ValueError('Please specify "upper" or "lower" X-point.')

        if active:
            self.X = np.delete(self.X, list(range(ix - ni, ix + ni)))
            self.Z = np.delete(self.Z, list(range(ix - ni, ix + ni)))
            self._X = np.delete(self._X, list(range(ix - ni, ix + ni)))
            self._Z = np.delete(self._Z, list(range(ix - ni, ix + ni)))
            o = ix - ni
            i = o - 2
            abcd = np.array(
                [self.X[i : o + 2][::-j], self.Z[i : o + 2]][::-j]
            ).T  # points of perp
            x_point = vector_intersect(*abcd)
            self.xpoints.append(XpointConstraint(*x_point, active, loc))
            self._X = np.append(self._X, x_point[0])
            self._Z = np.append(self._Z, x_point[1])
        else:
            x = np.delete(self.X, list(range(ix - ni, ix + ni)))
            z = np.delete(self.Z, list(range(ix - ni, ix + ni)))
            o = ix - ni
            i = o - 2
            abcd = np.array([x[i : o + 2][::-j], z[i : o + 2]][::-j]).T  # points of perp
            xp = vector_intersect(*abcd)[::-1]
            x_point = [xp[0] * 1, xp[1] * 1.1]  # TODO: GUt number
            self.xpoints.append(XpointConstraint(*x_point, active, loc))

    def add_line_isoflux(self, p1, p2, n):
        """
        Ajoute une serie de points sur une ligne entre p1 et p2 ou des
        contraintes de isoflux seront ajoutees
        """
        xn = np.linspace(p1[0], p2[0], int(n))
        zn = np.linspace(p1[1], p2[1], int(n))
        self.X = np.append(self.X, xn)
        self.Z = np.append(self.Z, zn)

    def calc_X_leg(self, angle, length, loc="lower", pos="outer"):  # noqa (N802)
        """
        Calcule a posicao da perna do X-point.
        """
        x_point = [x for x in self.xpoints if x.loc == loc][0]
        if loc == "upper":
            z = x_point.z + length * np.sin(np.deg2rad(angle))
        elif loc == "lower":
            z = x_point.z - length * np.sin(np.deg2rad(angle))
        else:
            raise ValueError('Please specify loc: "upper" or "lower" X-point.')
        if pos == "inner":
            x = x_point.x - length * np.cos(np.deg2rad(angle))
        elif pos == "outer":
            x = x_point.x + length * np.cos(np.deg2rad(angle))
        else:
            raise ValueError('Please specify pos: "inner" or "outer" X leg.')
        self.add_line_isoflux(x_point, (x, z), self.n / 10)

    def generate_normals(self):
        """
        Generate the normal field constraints.
        """
        x, z = self.X, self.Z
        for xpt in self.xpoints:
            if xpt.active:
                x = np.append(x, xpt.x)
                z = np.append(z, xpt.z)
        x, z = anticlock(x, z)
        t_x, t_z = tangent(x, z)
        args = []
        for xpt in self.xpoints:
            if xpt.active:
                args.append(np.argmin(x - xpt.x + abs(z - xpt.z)))
        x = np.delete(x, args)
        z = np.delete(z, args)
        t_x = np.delete(t_x, args)
        t_z = np.delete(t_z, args)
        for xi, zi, tx, tz in zip(x, z, t_x, t_z):
            con = BdirConstraint(xi, zi, [tx, tz])  # NOTE: Bdir and psi cross-product
            self.bdir.append(con)


class SilverSurfer(ConstraintBuilder, ConstraintCalculator):
    """
    Abstract base class for the automated calculation of constraints:
        Matrix A
        Vector b

    Provides __call__ method for subclasses

    Attributes
    ----------
    .xpoints
    .xlegs
    .bdir
    .bxval
    .bzval
    .bpval
    .psi
    .psinorm
    .isoflux
    """

    def __init__(self):
        self.xpoints = []
        self.xlegs = []
        self.psinvals = []
        self.psivals = []
        self.isoflux = {}
        self.bdir = []
        self.bxval = []
        self.bzval = []
        self.bpval = []
        self.flag_A_built = False

    def __call__(self, equilibrium, I_not_dI=False, fixed_coils=False):  # noqa (N803)
        """
        The base function for application of constraints to an Equilibrium.
        """
        if I_not_dI:
            # hack to change from dI to I optimiser (and keep both)
            dummy = equilibrium.plasma_coil()
            dummy.coilset = equilibrium.coilset
            equilibrium = dummy

        super().__call__(equilibrium)
        for xpt in self.xpoints:
            f = 1 if xpt.active else 1.5  # TODO: Gut number
            self.add_X_point(*xpt, factor=f)  # Bp = 0 only

        for bpt in self.bdir:
            self.add_Bdir(bpt.x, bpt.z, bpt.Bdir)

        for bxpt in self.bxval:
            self.add_Bxval(bxpt.x, bxpt.z, bxpt.Bx)

        for bzpt in self.bzval:
            self.add_Bzval(bzpt.x, bzpt.z, bzpt.Bz)

        for bppt in self.bpval:
            self.add_Bpval(bppt.x, bppt.z, bppt.Bp)

        for isoflux in self.isoflux.values():
            self.psiref = equilibrium.psi(*isoflux.refpoint)
            self.Iref = self.coilset.control_psi(*isoflux.refpoint)
            for x, z in zip(isoflux.x, isoflux.z):
                self.add_isoflux(x, z, self.psiref)

        if self.psivals:
            for p in self.psivals:
                self.add_psival(p.x, p.z, p.psi, b_dir=p.Bdir, factor=1)

        if self.psinvals:
            o_points, x_points = self.eq.get_OX_points()
            for pn in self.psinvals:
                psi = get_psi(pn.psi_norm, o_points[0].psi, x_points[0].psi)
                self.add_psival(pn.x, pn.z, psi, factor=1)
        # TODO: Make A once for fixed coil positions
        if fixed_coils:
            if not self.flag_A_built:
                self.build_A()
                self.flag_A_built = True
        else:
            self.build_A()
        self.build_b()

    def set_psib(self, psib):
        """
        Sets the boundary flux value used for the PsiPointConstraints.
        """
        for con in self.psivals:
            con.psi = psib

    def plot_shape(self, ax=None, color="b"):
        """
        Plot the shape of the constraints in space.
        """
        if ax is None:
            ax = plt.gca()
        for i, (x, z) in enumerate(zip(self.X, self.Z)):
            ax.plot(x, z, "s", marker="o", color=color)
            # ax.annotate(i, xy=[x, z])
        for x, z in self.xpoints:
            ax.plot(x, z, marker="X", ms=15, color=color)
        ax.set_aspect("equal")

    def copy(self):
        """
        Get a deep copy of the SilverSurfer instance.
        """
        return deepcopy(self)


class SNReference(SilverSurfer):
    """
    The default single null reference surface set of constraints using the
    Johner separatrix shape parameterisation.
    """

    def __init__(self, R_0, Z_0, A, kappa, delta, psibval, n=100, upper=True):
        super().__init__()
        self.n = n
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
        self.xpoints = [XpointConstraint(f_s.x[arg_x], f_s.z[arg_x])]
        f_s.interpolate(self.n)
        x_s, z_s = f_s.x, f_s.z

        self.psivals = [PsiConstraint(x, z, psibval) for x, z in zip(x_s, z_s)]
        self.X, self.Z = x_s, z_s
        self.calc_X_leg(50, 1.45)
        self.calc_X_leg(40, 1, pos="inner")
        self.psivals = [PsiConstraint(x, z, psibval) for x, z in zip(self.X, self.Z)]


class DNReference(SilverSurfer):
    """
    The default double null reference surface set of constraints using the
    Johner separatrix shape parameterisation.
    """

    def __init__(self, R_0, Z_0, A, kappa, delta, psibval, n=400):
        super().__init__()
        self.n = n
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
        self.xpoints = [
            XpointConstraint(f_s.x[arg_xl], f_s.z[arg_xl]),
            XpointConstraint(f_s.x[arg_xu], f_s.z[arg_xu]),
        ]
        f_s.interpolate(self.n)
        x_s, z_s = f_s.x, f_s.z

        self.psivals = [PsiConstraint(x, z, psibval) for x, z in zip(x_s, z_s)]
        self.X, self.Z = x_s, z_s


class XzTesting(SilverSurfer):
    """
    Pure x-z psi constraints only. Useful in testing.
    """

    def __init__(self, x, z, psival, n=40):
        super().__init__()
        arg_x = np.argmin(z)
        self.xpoints = [XpointConstraint(x[arg_x], z[arg_x])]
        s = Loop(x, 0, z)
        s.interpolate(n)
        x, z = s.x, s.z

        self.psivals = [PsiConstraint(x, z, psival) for x, z in zip(x, z)]
        self.X, self.Z = x, z


class STReference(SilverSurfer):
    """
    The default spherical reference surface set of constraints using the
    Manickam separatrix shape parameterisation. Handles double and single
    nulls.
    """

    def __init__(
        self,
        R_0,
        Z_0,
        A,
        kappa,
        delta,
        indent=0.05,
        psival=None,
        psinval=None,
        double_null=False,
    ):
        super().__init__()
        self.n = 100
        self.R_0 = R_0
        self.Z_0 = Z_0
        self.A = A
        self.a = R_0 / A
        self.kappa = kappa  # PROCESS classic
        self.delta = delta
        self.indent = indent
        self.psivals = psival
        xx, dz = R_0 - self.delta * R_0 / A, R_0 / A * self.kappa

        pn = 1
        loop = flux_surface_manickam(R_0, Z_0, R_0 / A, kappa, delta, indent, n=self.n)

        bottom_clip = np.where(loop.z > -0.8 * (Z_0 + dz))
        loop = Loop(x=loop.x[bottom_clip], z=loop.z[bottom_clip])

        self.xpoints = [
            XpointConstraint(xx, Z_0 - dz, active=True),
        ]

        if double_null:
            up_clip = np.where(loop.z < 0.8 * (Z_0 + dz))
            loop = Loop(x=loop.x[up_clip], z=loop.z[up_clip])
            self.xpoints.append(XpointConstraint(xx, Z_0 + dz, active=True, loc="upper"))

        x = [
            R_0 - pn * R_0 / A,
            R_0 + pn * R_0 / A,
            xx,
        ]
        z = [0, 0, Z_0 - dz]

        if double_null:
            x.append(xx)
            z.append(Z_0 + dz)

        x = np.append(loop.x, np.array(x))
        z = np.append(loop.z, np.array(z))
        self.isoflux[pn] = IsofluxConstraint(x, z, [R_0 - pn * R_0 / A, Z_0])
        self.X, self.Z = x, z
        # self.calc_X_leg(60, 1)
        # self.calc_X_leg(60, 1, loc='upper')
        self.bxval = []
        self.bzval = []


class AutoConstraints(SilverSurfer):
    """
    An object to auto-generate a set of constraints from an equilibrium.
    """

    def __init__(self, eq, npoints=100):
        super().__init__()

        if eq._eqdsk:
            loop = Loop(x=eq._eqdsk["xbdry"], z=eq._eqdsk["zbdry"])
            psi = eq._eqdsk["psibdry"]
            argx = np.argmin(loop.z)
            self.xpoints = [XpointConstraint(loop.x[argx], loop.z[argx])]
            loop.interpolate(npoints)
            x, z = loop.x, loop.z
        else:
            lcfs = eq.get_LCFS()
            lcfs.interpolate(npoints)
            x, z = lcfs.x, lcfs.z
            psi = float(eq.psi(x[0], z[0]))

        self.psivals = [PsiConstraint(xi, zi, psi) for xi, zi in zip(x, z)]
        self.X = x
        self.Z = z


class ColocationConstraints(SilverSurfer):
    """
    WIP
    """

    def __init__(self, x, z, b_dir, psibval, x_xpoint, z_xpoint):
        super().__init__()
        self.bdir = [BdirConstraint(xi, zi, bi) for xi, zi, bi in zip(x, z, b_dir)]
        self.psivals = [
            PsiConstraint(xi, zi, psibval, bi) for xi, zi, bi in zip(x, z, b_dir)
        ]
        self.xpoints = [XpointConstraint(x_xpoint, z_xpoint)]


if __name__ == "__main__":
    from BLUEPRINT import test

    test()

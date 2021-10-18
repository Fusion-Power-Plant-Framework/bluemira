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
Coil and coil grouping objects
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from typing import Any, Optional

from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.tools import is_num
from bluemira.magnetostatics.greens import (
    greens_psi,
    greens_Bx,
    greens_Bz,
)
from bluemira.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.constants import (
    I_MIN,
    J_TOR_MIN,
    NBTI_B_MAX,
    NBTI_J_MAX,
    X_TOLERANCE,
)
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.plotting import CoilPlotter, CoilSetPlotter, PlasmaCoilPlotter


PF_COIL_NAME = "PF_{}"
CS_COIL_NAME = "CS_{}"
NO_COIL_NAME = "Unclassified_{}"


class CoilNamer:
    """
    Coil naming-numbering utility class. Coil naming convention is not enforced here.
    """

    __PF_counter: int = 1
    __CS_counter: int = 1
    __no_counter: int = 1

    @staticmethod
    def _get_prefix(coil: Any):
        if not hasattr(coil, "ctype") or coil.ctype not in ["PF", "CS"]:
            return NO_COIL_NAME
        elif coil.ctype == "CS":
            return CS_COIL_NAME
        elif coil.ctype == "PF":
            return PF_COIL_NAME

    @staticmethod
    def generate_name(coil: Any, idx: Optional[int] = None):
        """
        Generate a coil name based on its type and indexing if specified. If no index is
        specified, an encapsulated global counter assigns an index.

        Parameters
        ----------
        coil: Any
            Object to name
        idx: Optional[int]
            Name index. If None, assigned automatically

        Returns
        -------
        name: str
            Coil name
        """
        prefix = CoilNamer._get_prefix(coil)
        if idx is None:
            if prefix == NO_COIL_NAME:
                idx = CoilNamer.__no_counter
                CoilNamer.__no_counter += 1
            elif prefix == CS_COIL_NAME:
                idx = CoilNamer.__CS_counter
                CoilNamer.__CS_counter += 1
            elif prefix == PF_COIL_NAME:
                idx = CoilNamer.__PF_counter
                CoilNamer.__PF_counter += 1

        return prefix.format(idx)


def get_max_current(dx, dz, j_max):
    """
    Get the maximum current in a coil cross-sectional area

    Parameters
    ----------
    dx: float
        Coil half-width [m]
    dz: float
        Coil half-height [m]
    j_max: float
        Coil current density [MA/m^2]

    Returns
    -------
    max_current: float
        Maximum current [A]
    """
    return abs(j_max * 1e6 * (4 * dx * dz))


class CoilSizer:
    """
    Coil sizing utility class (observer pattern).

    Parameters
    ----------
    coil: Coil
        Coil to size

    Notes
    -----
    Maximum currents are not enforced anywhere in Coils. If you want constrain currents,
    you should use constrained optimisation techniques (with current bounds).
    """

    def __init__(self, coil):
        self.update(coil)

        dxdz_specified = is_num(self.dx) and is_num(self.dz)

        if not dxdz_specified and not (self.dx is None and self.dz is None):
            # Check that we don't have dx = None and dz = float or vice versa
            raise EquilibriaError("Must specify either dx and dz or neither.")

        if dxdz_specified and not self.flag_sizefix:
            # If dx and dz are specified, we presume the coil size should remain fixed
            self.flag_sizefix = True

        if dxdz_specified:
            self._set_coil_attributes(coil)

        if not dxdz_specified and not self.j_max:
            # Check there is a viable way to size the coil
            raise EquilibriaError("Must specify either dx and dz or j_max.")

        if not dxdz_specified and self.flag_sizefix:
            # If dx and dz are not specified, we cannot fix the size of the coil
            self.flag_sizefix = False

        coil.flag_sizefix = self.flag_sizefix

    def __call__(self, coil, current=None):
        """
        Apply the CoilSizer to a Coil.

        Parameters
        ----------
        coil: Coil
            Coil to size
        current: Optional[float]
            The current to use when sizing the coil. Defaults to the present coil
            current.
        """
        self.update(coil)

        if not self.flag_sizefix:
            # Adjust the size of the coil
            coil.dx, coil.dz = self._make_size(current)
            self._set_coil_attributes(coil)

    def _set_coil_attributes(self, coil):
        coil.rc = 0.5 * np.hypot(coil.dx, coil.dz)
        coil.x_corner, coil.z_corner = self._make_corners(
            coil.x, coil.z, coil.dx, coil.dz
        )

    def update(self, coil):
        """
        Update the CoilSizer

        Parameters
        ----------
        coil: Coil
            Coil to size
        """
        self.dx = coil.dx
        self.dz = coil.dz
        self.current = coil.current
        self.j_max = coil.j_max
        self.flag_sizefix = coil.flag_sizefix

    def get_max_current(self, coil):
        """
        Get the maximum current of a coil size.

        Parameters
        ----------
        coil: Coil
            Coil to get the maximum current for

        Returns
        -------
        max_current: float
            Maximum current for the coil

        Raises
        ------
        EquilibriaError:
            If the coil size is not fixed or no current density is specified
        """
        self.update(coil)
        if not self.flag_sizefix:
            raise EquilibriaError(
                "Cannot get the maximum current of a coil of an unspecified size."
            )

        if self.j_max is None:
            raise EquilibriaError(
                "Cannot get the maximum current of a coil of unspecified current density."
            )

        return get_max_current(self.dx, self.dz, self.j_max)

    def _make_size(self, current=None):
        """
        Size the coil based on a current and a current density.
        """
        if current is None:
            current = self.current

        half_width = 0.5 * np.sqrt((abs(current) / (1e6 * self.j_max)))
        return half_width, half_width

    @staticmethod
    def _make_corners(x_c, z_c, dx, dz):
        """
        Makes the coil corner vectors
        """
        xx, zz = np.ones(4) * x_c, np.ones(4) * z_c
        x_corner = xx + dx * np.array([-1, 1, 1, -1])
        z_corner = zz + dz * np.array([-1, -1, 1, 1])
        return x_corner, z_corner


class Coil:
    """
    Poloidal field coil with a rectangular cross-section.

    Parameters
    ----------
    x: float
        Coil geometric centre x coordinate [m]
    z: float
        Coil geometric centre z coordinate [m]
    current: float (default = 0)
        Coil current [A]
    dx: float
        Coil radial half-width [m] from coil centre to edge (either side)
    dz: float
        Coil vertical half-width [m] from coil centre to edge (either side)
    n_turns: int (default = 1)
        Number of turns
    n_filaments: int (default = 1)
        Number of filaments (for multi-filament coils)
    control: bool
        Enable or disable control system
    ctype: str
        Type of coil ['PF', 'CS', 'Plasma']
    j_max: float (default = 12.5)
        Maximum current density in the coil [MA/m^2]
    b_max: float (default = 12)
        Maximum magnetic field at the coil [T]
    name: str or None
        The name of the coil
    """

    # TODO - make Coil inherit from CoilGroup as a group of size 1

    __slots___ = [
        "x",
        "z",
        "current",
        "dx",
        "dz",
        "j_max",
        "b_max",
        "n_filaments",
        "n_turns",
        "flag_sizefix",
        "name",
        "sub_coils",
        "control",
        "ctype",
    ]

    def __init__(
        self,
        x,
        z,
        current=0,
        dx=None,
        dz=None,
        n_turns=1,
        control=True,
        ctype="PF",
        j_max=None,
        b_max=None,
        name=None,
        n_filaments=1,
        flag_sizefix=False,
    ):

        self.x = x
        self.z = z
        self.dx = dx
        self.dz = dz
        self.current = current
        self.j_max = j_max
        self.b_max = b_max
        self.n_turns = n_turns
        self.n_filaments = n_filaments
        self.control = control
        self.ctype = ctype
        self.flag_sizefix = flag_sizefix

        if name is None:
            # We need to have a reasonable coil name
            name = CoilNamer.generate_name(self, None)
        self.name = name

        self.__sizer = CoilSizer(self)
        self.__sizer(self)

        self.sub_coils = None

    @property
    def n_coils(self):
        """
        Number of coils.
        """
        return 1

    def set_current(self, current):
        """
        Sets the current in a Coil object

        Parameters
        ----------
        current: float
            The current to set in the coil
        """
        self.current = current
        if self.sub_coils is not None:
            for coil in self.sub_coils.values():
                coil.set_current(current / len(self.sub_coils))

    def adjust_current(self, d_current):
        """
        Adjusts the current in a Coil object

        Parameters
        ----------
        d_current: float
            The change in current to apply to the coil
        """
        self.current += d_current
        if self.sub_coils is not None:
            for coil in self.sub_coils.values():
                coil.adjust_current(d_current / len(self.sub_coils))

    def set_position(self, x, z, dz=None):
        """
        Sets the position of the Coil object

        Parameters
        ----------
        x: float
            The radial position of the coil
        z: float
            The vertical position of the coil
        dz: float or None
            The vertical extent of the coil (applied to CS coils only)
        """
        self.x = x
        self.z = z
        if dz is None:
            self.__sizer(self)
        else:
            self.set_dz(dz)
        self.sub_coils = None  # Need to re-mesh if this is what you want

    def adjust_position(self, dx, dz):
        """
        Adjusts the position of the Coil object

        Parameters
        ----------
        dx: float
            The change in radial position to apply to the coil
        dz: float
            The change in vertical position to apply to the coil
        """
        self.set_position(self.x + dx, self.z + dz)

    def set_dx(self, dx):
        """
        Adjusts the radial thickness of the Coil object (meshing not handled)
        """
        self.dx = dx
        self.__sizer(self)

    def set_dz(self, dz):
        """
        Adjusts the vertical thickness of the Coil object (meshing not handled)
        """
        self.dz = dz
        self.__sizer(self)

    def make_size(self, current=None):
        """
        Size the coil based on a current and a current density.
        """
        self.__sizer(self, current)
        if self.flag_sizefix is False:
            self.sub_coils = None  # Need to re-mesh if this is what you want

    def fix_size(self):
        """
        Fixes the size of the coil
        """
        self.flag_sizefix = True
        self.__sizer.update(self)

    def _points_inside_coil(self, x, z):
        """
        Determine which points lie inside or on the coil boundary.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinates to check
        z: Union[float, np.array]
            The z coordinates to check

        Returns
        -------
        inside: np.array(dtype=np.bool)
            The Boolean array of point indices inside/outside the coil boundary
        """
        x, z = np.ascontiguousarray(x), np.ascontiguousarray(z)
        # Add an offset, to ensure points very near the edge are counted as
        # being on the edge of a coil
        atol = X_TOLERANCE
        x_min, x_max = self.x - self.dx - atol, self.x + self.dx + atol
        z_min, z_max = self.z - self.dz - atol, self.z + self.dz + atol
        return (x >= x_min) & (x <= x_max) & (z >= z_min) & (z <= z_max)

    def assign_material(self, j_max=NBTI_J_MAX, b_max=NBTI_B_MAX):
        """
        Assigns EM material properties to coil

        Parameters
        ----------
        j_max: float (default None)
            Overwrite default constant material max current density [MA/m^2]
        b_max: float (default None)
            Overwrite default constant material max field [T]
        """
        if not is_num(j_max):
            raise EquilibriaError(f"j_max must be specified as a number, not: {j_max}")
        if not is_num(b_max):
            raise EquilibriaError(f"b_max must be specified as a number, not: {b_max}")

        self.j_max = j_max
        self.b_max = b_max
        self.__sizer.update(self)

    def get_max_current(self):
        """
        Gets the maximum current for a coil with a specified size

        Returns
        -------
        Imax: float
            The maximum current that can be produced by the coil [A]
        """
        return self.__sizer.get_max_current(self)

    def mesh_coil(self, d_coil):
        """
        Mesh an individual coil into smaller subcoils.
        This handles both variable area PFs with fixed Jmax [MA/m^2]
        and fixed area CS modules with fixed A [m^2]

        Parameters
        ----------
        d_coil: float > 0
            The coil sub-division size

        Note
        ----
        Breaks down the coil into SubCoils and adds to the self.subcoils dict.
            .subcoils[sub_name] = SubCoil
        """
        dx_full, dz_full = abs(self.dx), abs(self.dz)

        nx = max(1, int(np.ceil(dx_full * 2 / d_coil)))
        nz = max(1, int(np.ceil(dz_full * 2 / d_coil)))
        self.n_filaments = nx * nz

        if self.n_filaments == 1:
            # Catches the cases where:
            #    dx = dz = 0
            #    the mesh scale is bigger than the coil
            return  # Do nothing, coil is not meshed, .sub_coils is not created
        else:
            dx, dz = dx_full / nx, dz_full / nz

            # Calculate sub-coil centroids
            x_sc = (self.x - dx_full) + dx * np.arange(1, 2 * nx, 2)
            z_sc = (self.z - dz_full) + dz * np.arange(1, 2 * nz, 2)
            x_sc, z_sc = np.meshgrid(x_sc, z_sc)

            self.sub_coils = {}
            current = self.current / self.n_filaments  # Current per coil filament

            for i, (xc, zc) in enumerate(zip(x_sc.flat, z_sc.flat)):
                sub_name = self.name + f"_{i + 1:1.0f}"
                c = Coil(
                    xc,
                    zc,
                    current,
                    n_filaments=1,
                    dx=dx,
                    dz=dz,
                    ctype=self.ctype,
                    name=sub_name,
                )
                self.sub_coils[sub_name] = c

    def psi(self, x, z):
        """
        Calculate poloidal flux at (x, z)
        """
        return self.control_psi(x, z) * self.current

    def psi_greens(self, pgreen):
        """
        Calculate plasma psi from Greens functions and current
        """
        return self.current * pgreen

    def Bx(self, x, z):
        """
        Calculate radial magnetic field Bx at (x, z)
        """
        return self.control_Bx(x, z) * self.current

    def Bx_greens(self, bgreen):
        """
        Uses the Greens mapped dict to quickly compute the Bx
        """
        return self.current * bgreen

    def Bz(self, x, z):
        """
        Calculate vertical magnetic field Bz at (x, z)
        """
        return self.control_Bz(x, z) * self.current

    def Bz_greens(self, bgreen):
        """
        Uses the Greens mapped dict to quickly compute the Bx
        """
        return self.current * bgreen

    def Bp(self, x, z):
        """
        Calculate poloidal magnetic field Bp at (x, z)
        """
        return np.hypot(
            self.control_Bx(x, z) * self.current, self.control_Bz(x, z) * self.current
        )

    def control_psi(self, x, z):
        """
        Calculate poloidal flux at (x, z) due to a unit current
        """
        if self.sub_coils is None:
            return greens_psi(self.x, self.z, x, z, self.dx, self.dz) * self.n_turns

        gpsi = [greens_psi(c.x, c.z, x, z, c.dx, c.dz) for c in self.sub_coils.values()]
        return sum(gpsi) / self.n_filaments

    def control_Bx(self, x, z):
        """
        Calculate the radial magnetic field response at (x, z) due to a unit
        current. Green's functions are used outside the coil, and a semianalytic
        method is used for the field inside the coil.

        Parameters
        ----------
        x: Union[float, int, np.array]
            The x values at which to calculate the Bx response
        z: Union[float, int, np.array]
            The z values at which to calculate the Bx response

        Returns
        -------
        Bx: Union[float, np.array]
            The radial magnetic field response at the x, z coordinates.
        """
        return self._mix_control_method(
            x, z, self._control_Bx_greens, self._control_Bx_analytical
        )

    def control_Bz(self, x, z):
        """
        Calculate the vertical magnetic field response at (x, z) due to a unit
        current. Green's functions are used outside the coil, and a semianalytic
        method is used for the field inside the coil.

        Parameters
        ----------
        x: Union[float, int, np.array]
            The x values at which to calculate the Bz response
        z: Union[float, int, np.array]
            The z values at which to calculate the Bz response

        Returns
        -------
        Bz: Union[float, np.array]
            The vertical magnetic field response at the x, z coordinates.
        """
        return self._mix_control_method(
            x, z, self._control_Bz_greens, self._control_Bz_analytical
        )

    def _mix_control_method(self, x, z, greens_func, semianalytic_func):
        """
        Boiler-plate helper function to mixed the Green's function responses
        with the semi-analytic function responses, as a function of position
        outside/inside the coil boundary.
        """
        x, z = np.ascontiguousarray(x), np.ascontiguousarray(z)
        if self.dx == 0 or self.dz == 0:
            response = greens_func(x, z)

        else:
            inside = self._points_inside_coil(x, z)
            response = np.zeros(x.shape)
            if np.any(~inside):
                response[~inside] = greens_func(x[~inside], z[~inside])
            if np.any(inside):
                response[inside] = semianalytic_func(x[inside], z[inside])
        if x.size == 1:
            return response[0]
        return response

    def _control_Bx_greens(self, x, z):
        """
        Calculate radial magnetic field Bx respose at (x, z) due to a unit
        current using Green's functions.
        """
        if self.sub_coils is None:
            return greens_Bx(self.x, self.z, x, z) * self.n_turns

        gx = [greens_Bx(c.x, c.z, x, z) for c in self.sub_coils.values()]
        return sum(gx) / self.n_filaments

    def _control_Bz_greens(self, x, z):
        """
        Calculate vertical magnetic field Bz at (x, z) due to a unit current
        """
        if self.sub_coils is None:
            return greens_Bz(self.x, self.z, x, z) * self.n_turns

        gz = [greens_Bz(c.x, c.z, x, z) for c in self.sub_coils.values()]
        return sum(gz) / self.n_filaments

    def _control_Bx_analytical(self, x, z):
        """
        Calculate radial magnetic field Bx response at (x, z) due to a unit
        current using semi-analytic method.
        """
        return semianalytic_Bx(self.x, self.z, x, z, d_xc=self.dx, d_zc=self.dz)

    def _control_Bz_analytical(self, x, z):
        """
        Calculate vertical magnetic field Bz response at (x, z) due to a unit
        current using semi-analytic method.
        """
        return semianalytic_Bz(self.x, self.z, x, z, d_xc=self.dx, d_zc=self.dz)

    def F(self, eqcoil):  # noqa (N802)
        """
        Calculate the force response at the coil centre including the coil
        self-force.

        \t:math:`\\mathbf{F} = \\mathbf{j}\\times \\mathbf{B}`\n
        \t:math:`F_x = IB_z+\\dfrac{\\mu_0I^2}{4\\pi X}\\textrm{ln}\\bigg(\\dfrac{8X}{r_c}-1+\\xi/2\\bigg)`\n
        \t:math:`F_z = -IBx`
        """  # noqa (W505)
        Bx, Bz = eqcoil.Bx(self.x, self.z), eqcoil.Bz(self.x, self.z)
        if self.rc != 0:  # true divide errors for zero current coils
            a = MU_0 * self.current ** 2 / (4 * np.pi * self.x)
            fx = a * (np.log(8 * self.x / self.rc) - 1 + 0.25)

        else:
            fx = 0
        return np.array(
            [
                (self.current * Bz + fx) * 2 * np.pi * self.x,
                -self.current * Bx * 2 * np.pi * self.x,
            ]
        )

    def control_F(self, coil):  # noqa (N802)
        """
        Returns the Green's matrix element for the coil mutual force.

        \t:math:`Fz_{i,j}=-2\\pi X_i\\mathcal{G}(X_j,Z_j,X_i,Z_i)`
        """
        if coil.x == self.x and coil.z == self.z:
            # self inductance
            if self.rc != 0:
                a = MU_0 / (4 * np.pi * self.x)
                Bz = a * (np.log(8 * self.x / self.rc) - 1 + 0.25)
            else:
                Bz = 0
            Bx = 0  # Should be 0 anyway

        else:
            Bz = coil.control_Bz(self.x, self.z)
            Bx = coil.control_Bx(self.x, self.z)
        return 2 * np.pi * self.x * np.array([Bz, -Bx])  # 1 cross B

    def toggle_control(self):
        """
        Toggles coil control functionality
        """
        self.control = not self.control
        if self.sub_coils is not None:
            for coil in self.sub_coils.values():
                coil.toggle_control()

    def copy(self):
        """
        Get a deepcopy of the Coil.

        Returns
        -------
        copy: Coil
            The deepcopy of the Coil
        """
        return deepcopy(self)

    @property
    def n_control(self):
        """
        The length of the controls.
        """
        return 1

    def n_constraints(self):
        """
        The length of the constraints.
        """
        return 1

    @property
    def area(self):
        """
        The cross-sectional area of the coil

        Returns
        -------
        area: float
            The cross-sectional area of the coil [m^2]
        """
        return 4 * self.dx * self.dz

    @property
    def volume(self):
        """
        The volume of the coil

        Returns
        -------
        volume: float
            The volume of the coil [m^3]
        """
        return self.area * 2 * np.pi * self.x

    def plot(self, ax=None, subcoil=True, **kwargs):
        """
        Plots the Coil object onto `ax`. Should only be used for individual
        coils. Use CoilSet.plot() for CoilSet objects
        """
        return CoilPlotter(self, ax=ax, subcoil=subcoil, **kwargs)

    def __repr__(self):
        """
        Pretty console coil rendering.
        """
        return f"{self.__class__.__name__}({self.__str__()})"

    def __str__(self):
        """
        Pretty coil printing.
        """
        return (
            f"X={self.x:.2f} m, Z={self.z:.2f} m, I={self.current/1e6:.2f} MA "
            f"control={self.control}"
        )

    def to_dict(self):
        """
        Returns a parameter dictionary for the Coil object
        """
        return {
            "x": self.x,
            "z": self.z,
            "current": self.current,
            "x_corner": self.x_corner,
            "z_corner": self.z_corner,
            "dx": self.dx,
            "dz": self.dz,
            "rc": self.rc,
            "n_filaments": self.n_filaments,
            "n_turns": self.n_turns,
            "ctype": self.ctype,
            "control": self.control,
            "name": self.name,
        }

    def to_group_vecs(self):
        """
        Convert Coil properties to numpy arrays

        Returns
        -------
        x: np.ndarray(n_coils)
            The x-positions of coils
        z: np.ndarray(n_coils)
            The z-positions of coils.
        dx: np.ndarray(n_coils)
            The coil size in the x-direction.
        dz: np.ndarray(n_coils)
            The coil size in the z-direction.
        currents: np.ndarray(n_coils)
            The coil currents.
        """
        return (
            np.array([self.x]),
            np.array([self.z]),
            np.array([self.dx]),
            np.array([self.dz]),
            np.array([self.current]),
        )


class CoilGroup:
    """
    Abstract grouping of Coil objects. Handles ordering of coils by type and name,
    and grouping of methods.

    Parameters
    ----------
    coils: List[Coil]
        The list of Coils to group together
    """

    def __init__(self, coils):
        self.coils = self.sort_coils(coils)

    @staticmethod
    def sort_coils(coils):
        """
        Sort coils in an ordered dictionary, by type and by name.
        """
        pf_coils = [coil for coil in coils if coil.ctype == "PF"]
        cs_coils = [coil for coil in coils if coil.ctype == "CS"]
        other = [coil for coil in coils if coil.ctype not in ["PF", "CS"]]

        pf_coils.sort(key=lambda x: x.name)
        cs_coils.sort(key=lambda x: x.name)
        other.sort(key=lambda x: x.name)

        all_coils = pf_coils + cs_coils + other

        return {coil.name: coil for coil in all_coils}

    def __getitem__(self, name):
        """
        Dict-like behaviour for CoilGroup object
        """
        try:
            return self.coils[name]
        except IndexError:
            raise KeyError(f'CoilGroup does not contain coil index "{name}"')

    def __repr__(self):
        """
        Pretty CoilGroup console rendering.
        """
        try:
            return "\n".join([n + ": " + c.__str__() for n, c in self.coils.items()])

        except AttributeError:  # Coils still unnamed (list)
            return "\n".join(c.__str__() for c in self.coils)

    def to_dict(self):
        """
        Convert Group to dictionary.

        Returns
        -------
        dict
            a dict of coil dicts

        Notes
        -----
        Example output:

        {'PF_1': {'X': 10.4, 'Z': 4.4, 'I': 36, ...},
        'PF_2': {'X': 4.4, ....}}
        """
        cdict = {}

        coils = self.coils

        for name, coil in coils.items():
            cdict[name] = coil.to_dict()
        return cdict

    def copy(self):
        """
        Get a deep copy of the CoilGroup.
        """
        return deepcopy(self)

    def to_group_vecs(self):
        """
        Convert CoilGroup properties to numpy arrays

        Returns
        -------
        x: np.ndarray(n_coils)
            The x-positions of coils
        z: np.ndarray(n_coils)
            The z-positions of coils.
        dx: np.ndarray(n_coils)
            The coil size in the x-direction.
        dz: np.ndarray(n_coils)
            The coil size in the z-direction.
        currents: np.ndarray(n_coils)
            The coil currents.
        """
        x, z, dx, dz, currents = [], [], [], [], []

        for coil in self.coils.values():
            xi, zi, dxi, dzi, ci = coil.to_group_vecs()
            x.extend(xi)
            z.extend(zi)
            dx.extend(dxi)
            dz.extend(dzi)
            currents.extend(ci)
        return np.array(x), np.array(z), np.array(dx), np.array(dz), np.array(currents)

    def add_coil(self, coil):
        """
        Add a coil to the CoilGroup.

        Parameters
        ----------
        coil: Coil object
            The coil to be added to the CoilGroup
        """
        self.coils[coil.name] = coil
        self.coils = self.sort_coils(list(self.coils.values()))

    def remove_coil(self, coilname):
        """
        Remove a coil from the CoilGroup.

        Parameters
        ----------
        coilname: str
            The coil name (key in .coils) of the Coil to be removed
        """
        try:
            del self.coils[coilname]
        except KeyError:
            raise KeyError("No coil with such a name in the CoilGroup.")

    def mesh_coils(self, d_coil=-1):
        """
        Sub-divide coils into subcoils based on size. Coils are meshed within
        the Coil object, such that Coil().sub_coils = [Coil(), Coil(), ..]
        """
        if d_coil < 0:  # dCoil not set, use stored value
            if not hasattr(self, "d_coil"):
                self.d_coil = 0
        else:
            self.d_coil = d_coil
        #  Do not sub-divide coils
        if self.d_coil == 0:
            return
        else:
            for coil in self.coils.values():
                if coil.ctype == "PF":
                    coil.make_size()
                coil.mesh_coil(self.d_coil)

    def fix_sizes(self):
        """
        Holds the sizes of all coils constant
        """
        for coil in self.coils.values():
            coil.fix_size()

    # Grouping methods
    @staticmethod
    def _sum_all(seq, method, *args):
        """
        Sums all the responses in a sequence of objects for obj.method(*args)
        """
        a = np.array([getattr(obj, method)(*args) for obj in seq])
        return a.sum(axis=0)

    @staticmethod
    def _all_if(seq, method, *args):
        """
        Sums all the responses in a sequence of objects for obj.method(*args)
        if obj.control is True
        """
        return [getattr(obj, method)(*args) for obj in seq if obj.control]

    def psi(self, x, z):
        """
        Poloidal flux due to coils
        """
        return self._sum_all(self.coils.values(), "psi", x, z)

    def map_psi_greens(self, x, z):
        """
        Mapping of the psi Greens functions into a dict for each coil
        """
        pgreen = {}
        for name, coil in self.coils.items():
            pgreen[name] = coil.control_psi(x, z)
        return pgreen

    def psi_greens(self, pgreen):
        """
        Uses the Greens mapped dict to quickly compute the psi
        """
        psi_coils = 0
        for name, coil in self.coils.items():
            psi_coils += coil.psi_greens(pgreen[name])
        return psi_coils

    def Bx(self, x, z):
        """
        Radial magnetic field at x, z
        """
        return self._sum_all(self.coils.values(), "Bx", x, z)

    def map_Bx_greens(self, x, z):
        """
        Mapping of the Bx Greens function into a dict for each coil
        """
        bgreen = {}
        for name, coil in self.coils.items():
            bgreen[name] = coil.control_Bx(x, z)
        return bgreen

    def Bx_greens(self, bgreen):
        """
        Uses the Greens mapped dict to quickly compute the Bx
        """
        bx_coils = 0
        for name, coil in self.coils.items():
            bx_coils += coil.Bx_greens(bgreen[name])
        return bx_coils

    def Bz(self, x, z):
        """
        Vertical magnetic field at x, z
        """
        return self._sum_all(self.coils.values(), "Bz", x, z)

    def map_Bz_greens(self, x, z):
        """
        Mapping of the Bz Greens function into a dict for each coil
        """
        bgreen = {}
        for name, coil in self.coils.items():
            bgreen[name] = coil.control_Bz(x, z)
        return bgreen

    def Bz_greens(self, bgreen):
        """
        Uses the Greens mapped dict to quickly compute the Bx
        """
        bz_coils = 0
        for name, coil in self.coils.items():
            bz_coils += coil.Bz_greens(bgreen[name])
        return bz_coils

    def Bp(self, x, z):
        """
        Poloidal magnetic field at x, z
        """
        bx = self._sum_all(self.coils.values(), "Bx", x, z)
        bz = self._sum_all(self.coils.values(), "Bz", x, z)
        return np.hypot(bx, bz)

    def control_Bx(self, x, z):
        """
        Returns a list of control responses for Bx at the given (x, z)
        location(s)
        """
        return self._all_if(self.coils.values(), "control_Bx", x, z)

    def control_Bz(self, x, z):
        """
        Returns a list of control responses for Bz at the given (x, z)
        location(s)
        """
        return self._all_if(self.coils.values(), "control_Bz", x, z)

    def control_psi(self, x, z):
        """
        Returns a list of control responses for psi at the given (x, z)
        location(s)
        """
        return self._all_if(self.coils.values(), "control_psi", x, z)

    def F(self, eqcoil):  # noqa (N802)
        """
        Returns the forces in the CoilGroup as a response to an equilibrium or
        other CoilGroup

        Parameters
        ----------
        eqcoil: Equilibrium or Coil/CoilSet object
            Any grouping with .Bx() and .Bz() functionality
        """
        forces = np.zeros((self.n_coils, 2))
        for i, coil in enumerate(self.coils.values()):
            forces[i, :] = coil.F(eqcoil)
        return forces

    def control_F(self, coil):
        """
        Returns a list of control responses for F at the given coil location(s)
        """
        c_forces = np.zeros((self.n_coils, 2))
        for i, coil in enumerate(self.coils.values()):
            c_forces[i, :] = coil.control_F(coil)
        return c_forces

    def toggle_control(self, *name):
        """
        Toggles the control of a coil in a CoilGroup. Tracks control of a coil
        in the CoilGroup
        """
        for n in name:
            self.coils[n].toggle_control()
        self._classify_control()

    @property
    def n_coils(self):
        """
        The number of coils.
        """
        return sum([c.n_coils for c in self.coils.values()])

    @property
    def area(self):
        """
        The total cross-sectional area of the coil

        Returns
        -------
        area: float
            The total cross-sectional area of the coil [m^2]
        """
        return sum([c.area for c in self.coils.values()])

    @property
    def volume(self):
        """
        The total volume of the CoilGroup

        Returns
        -------
        volume: float
            The total volume of the CoilGroup [m^3]
        """
        return sum([c.volume for c in self.coils.values()])

    def _classify_control(self):
        """
        Tracks controlled and uncontrolled coils. _ccoils used in constraints
        for optimisation of controlled coils only
        """
        self._ccoils = [c for c in self.coils.values() if c.control]
        self._ucoils = [c for c in self.coils.values() if not c.control]


class Solenoid(CoilGroup):
    """
    Solenoid object for a vertically arranged stack of CS coils. Will default
    to an ITER-like equispaced arrangement.

    Parameters
    ----------
    x: float
        Solenoid geometric centre axis location (radius from machine axis)
    dx: float
        Solenoid thickness either side of centre
    z_min, z_max: float, float
        Minimum and maximum Z coordinates of the Solenoid
    n_CS: int
        Number of central solenoid modules
    """

    control = True

    def __init__(self, x, dx, z_min, z_max, n_CS, gap=0.1, j_max=12.5, coils=None):
        self.radius = x
        self.dx = dx
        self.z_min = z_min
        self.z_max = z_max
        self.n_CS = n_CS
        self.gap = gap
        self.j_max = j_max
        if coils is None:
            self.coils = []
            self.grid()
        else:
            self.coils = coils

    @classmethod
    def from_coils(cls, coils):
        """
        Initialises a Solenoid object from a list of input coils

        Parameters
        ----------
        coils: Iterable[Coil]
            The list of coils from which to make the Solenoid
        """
        if not coils:
            return None

        x = coils[0].x
        d_x = coils[0].dx
        n_cs = len(coils)
        z_min = min([c.z - c.dz for c in coils])
        z_max = max([c.z + c.dz for c in coils])
        coils = sorted(coils, key=lambda c_: -c_.z)  # Order from top to bottom
        for i, coil in enumerate(coils):
            coil.name = CoilNamer.generate_name(coil, i + 1)

        return cls(x, d_x, z_min, z_max, n_cs, coils=coils)

    def grid(self):
        """
        Initial CS sub-divisions (equi-spaced)
        """
        dz = ((self.z_max - self.z_min) - self.gap * (self.n_CS - 1)) / self.n_CS / 2
        v1 = np.arange(0, self.n_CS)
        v2 = np.arange(1, self.n_CS * 2, 2)
        zc = self.z_max - self.gap * v1 - dz * v2
        for zc in zc:
            self.add_coil(self.radius, zc, dx=self.dx, dz=dz)

    def add_coil(self, x, z, dx, dz):
        """
        Adds a coil to the Solenoid object
        """
        coil = Coil(
            x,
            z,
            current=0,
            n_turns=1,
            control=True,
            ctype="CS",
            j_max=self.j_max,
            dx=dx,
            dz=dz,
        )
        self.coils.append(coil)

    def calc_psi_max(self, x, z):
        """
        Calculates the maximum flux which can be generated by the Solenoid at a
        particular location

        Parameters
        ----------
        x, z: float, float or array(N, M), array(N, M)
            Coordinates of the point at which to evaluate the maximum flux

        Returns
        -------
        psimax: float or array(N, M)
            The maximum psi at x, z [V.s] NOTE: V.s!!!
        """
        old_currents = [coil.current for coil in self.coils]
        for coil in self.coils:
            max_current = coil.get_max_current()
            coil.current = max_current
        self.mesh_coils(0.2)
        psimax = self.psi(x, z)

        for current, coil in zip(old_currents, self.coils):
            coil.current = current
        return 2 * np.pi * psimax


class PlasmaCoil:
    """
    PlasmaCoil object for finite difference representation of toroidal current
    carrying plasma

    Parameters
    ----------
    j_tor: np.array(n, m)
        The toroidal current density array from which to make a PlasmaCoil
    grid: Grid object
        The grid on which the values will be calculated

    Notes
    -----
    Uses direct summing of Green's functions to avoid SIGKILL and MemoryErrors
    when using very dense grids (e.g. CREATE).
    """

    def __init__(self, j_tor, grid):
        self.j_tor = j_tor
        self.grid = grid
        self.plasma_psi = None

        if j_tor is not None:
            self._ii, self._jj = np.where(j_tor > J_TOR_MIN)
            self.plasma_psi = self._convolve(greens_psi, self.grid.x, self.grid.z)

            # Everything past here used to pretend PlasmaCoil is an EqObject
            self.psi_func = RectBivariateSpline(
                grid.x[:, 0], grid.z[0, :], self.plasma_psi
            )
            Bx = -self.psi_func(grid.x, grid.z, dy=1, grid=False) / grid.x
            Bz = self.psi_func(grid.x, grid.z, dx=1, grid=False) / grid.x
            self.plasma_Bx = Bx
            self.plasma_Bz = Bz
            self.plasma_Bp = np.hypot(Bx, Bz)

        else:  # Breakdown state (no plasma at all)
            self._ii, self._jj = [], []
            self.plasma_psi = np.zeros([self.grid.nx, self.grid.nz])
            self.psi_func = RectBivariateSpline(
                grid.x[:, 0], grid.z[0, :], self.plasma_psi
            )
            self.plasma_Bx = np.zeros([self.grid.nx, self.grid.nz])
            self.plasma_Bz = np.zeros([self.grid.nx, self.grid.nz])
            self.plasma_Bp = np.zeros([self.grid.nx, self.grid.nz])

    def _convolve(self, func, x, z):
        """
        Map a Green's function across the grid at a point, without crashing or
        running out of memory.
        """
        array = np.zeros_like(x, dtype=np.float)
        for i, j in zip(self._ii, self._jj):
            current = self.j_tor[i, j] * self.grid.dx * self.grid.dz
            array += current * func(self.grid.x[i, j], self.grid.z[i, j], x, z)
        return array

    def psi(self, x=None, z=None):
        """
        Poloidal magnetic flux at x, z
        """
        if x is None and z is None:
            return self.plasma_psi
        else:
            return self._convolve(greens_psi, x, z)

    def Bx(self, x=None, z=None):
        """
        Horizontal magnetic field at x, z
        """
        if x is None and z is None:
            return self.plasma_Bx
        else:
            return self._convolve(greens_Bx, x, z)

    def Bz(self, x=None, z=None):
        """
        Vertical magnetic field at x, z
        """
        if x is None and z is None:
            return self.plasma_Bx
        else:
            return self._convolve(greens_Bz, x, z)

    def Bp(self, x=None, z=None):
        """
        Poloidal magnetic field at x, z
        """
        if x is None and z is None:
            return self.plasma_Bx
        else:
            return np.hypot(self.Bx(x, z), self.Bz(x, z))

    def plot(self, ax=None):
        """
        Plot the PlasmaCoil.

        Parameters
        ----------
        ax: Axes object
            The matplotlib axes on which to plot the Loop
        """
        return PlasmaCoilPlotter(self, ax=ax)

    def __repr__(self):
        """
        Get a simple string representation of the PlasmaCoil.
        """
        n_filaments = len(np.where(self.j_tor > 0)[0])
        return f"{self.__class__.__name__}: {n_filaments} filaments"


class Circuit(CoilGroup):
    """
    A grouping of Coils that are force to have the same current. The first coil in
    the Circuit is the controlled Coil.

    Parameters
    ----------
    coils: List[Coil]
        The list of Coils to group into a Circuit
    """

    def __init__(self, coils):
        if len(coils) < 2:
            raise EquilibriaError("A Circuit must be initialised with more than 1 Coil.")

        super().__init__(coils)
        self.control = True
        self.current = coils[0].current

    def adjust_current(self, d_current):
        """
        Adjust the current in the Circuit.
        """
        for i, coil in enumerate(self.coils.values()):
            coil.adjust_current(d_current)
        self.current += d_current

    def set_current(self, current):
        """
        Set the current in the Circuit.
        """
        for i, coil in enumerate(self.coils.values()):
            coil.set_current(current)
        self.current = current

    def get_control_current(self):
        """
        Get the control current from the Circuit.
        """
        return self.current * np.ones(len(self.coils))

    def map_psi_greens(self, x, z):
        """
        Mapping of the psi Greens functions into a dict for each coil
        """
        return {self.name: self.control_psi(x, z)}

    def psi_greens(self, pgreen):
        """
        Calculate psi from Greens functions and current
        """
        return self.current * pgreen

    def map_Bx_greens(self, x, z):
        """
        Mapping of the Bx Greens functions into a dict for each coil
        """
        return {self.name: self.control_Bx(x, z)}

    def Bx_greens(self, bx_green):
        """
        Calculate Bx from Greens functions and current
        """
        return self.current * bx_green

    def map_Bz_greens(self, x, z):
        """
        Mapping of the Bz Greens functions into a dict for each coil
        """
        return {self.name: self.control_Bz(x, z)}

    def Bz_greens(self, bz_green):
        """
        Calculate Bz from Greens functions and current
        """
        return self.current * bz_green

    def control_Bx(self, x, z):
        """
        Returns a list of control responses for Bx at the given (x, z)
        location(s)
        """
        return sum(super().control_Bx(x, z))

    def control_Bz(self, x, z):
        """
        Returns a list of control responses for Bz at the given (x, z)
        location(s)
        """
        return sum(super().control_Bz(x, z))

    def control_psi(self, x, z):
        """
        Returns a list of control responses for psi at the given (x, z)
        location(s)
        """
        return sum(super().control_psi(x, z))

    def control_F(self, coil):
        """
        Returns a list of control responses for F at the given (x, z)
        location(s)
        """
        return np.sum(super().control_F(coil), axis=0)

    def F(self, eqcoil):
        """
        Get the sum of the forces on a coil from the Circuit.
        """
        return np.sum(super().F(eqcoil), axis=0)

    @property
    def n_control(self):
        """
        The length of the controls.
        """
        return 1

    @property
    def n_constraints(self):
        """
        The length of the constraints.
        """
        return len(self.coils)

    def make_size(self, current=None):
        """
        Set the size of the coils in the Circuit.
        """
        for coil in self.coils.values():
            coil.make_size(current=current)

    def plot(self, ax=None, subcoil=True, **kwargs):
        """
        Plot the Circuit.
        """
        if ax is None:
            ax = plt.gca()
        for coil in self.coils.values():
            coil.plot(ax=ax, subcoil=subcoil, **kwargs)


class SymmetricCircuit(Circuit):
    """
    A grouping of Coils that are force to have the same  with symmetry about z=0.
    The first coil in the Circuit is the controlled Coil.

    Parameters
    ----------
    coils: Coil
        The coil from which to make a SymmetricCircuit
    """

    def __init__(self, coil):

        if coil.z == 0:
            raise EquilibriaError(
                "SymmetricCircuit must be initialised with a Coil with z != 0."
            )

        self.name = coil.name
        coil.name += ".1"

        mirror = Coil(
            x=coil.x,
            z=-coil.z,
            dx=coil.dx,
            dz=coil.dz,
            current=coil.current,
            n_turns=coil.n_turns,
            control=coil.control,
            ctype=coil.ctype,
            j_max=coil.j_max,
            b_max=coil.b_max,
            name=self.name + ".2",
            flag_sizefix=coil.flag_sizefix,
        )

        super().__init__([coil, mirror])

    def __setattr__(self, attr, value):
        """
        Custom __setattr__ to default to setting the attributes of
        the member coils if not found in SymmetricCircuit.

        Parameters
        ----------
        attr: str
              Name of attribute to fetch.
        value:
            Object to assign attribute to.
        """
        super().__setattr__(attr, value)
        if hasattr(self, "name") and hasattr(self, "coils"):
            name = self.name
            coil = self.coils[name + ".1"]
            if hasattr(coil, attr):
                if attr == "z":
                    coil1 = self.coils[name + ".1"]
                    coil1.__setattr__(attr, value)
                    coil2 = self.coils[name + ".2"]
                    coil2.__setattr__(attr, -value)
                else:
                    for cl_n in [".1", ".2"]:
                        coil = self.coils[name + cl_n]
                        coil.__setattr__(attr, value)

    def __getattr__(self, attr):
        """
        Custom __getattr__ to default to returning the attribute of
        the primary coil if not found in SymmetricCircuit.

        Parameters
        ----------
        attr: str
              Name of attribute to fetch.

        """
        name = self.__getattribute__("name")
        coil = self.__getattribute__("coils")[name + ".1"]
        if hasattr(coil, attr):
            return coil.__getattribute__(attr)
        else:
            return self.__getattribute__(attr)

    def apply_coil_method(self, method_name, *args, **kwargs):
        """
        Calls the coil method method_name on both coils in the
        SymmetricCircuit, and returns any results in a list.

        Parameters
        ----------
        method_name: str
            Name of method in Coil to call for each member of
            the SymmetricCircuit.

        args: tuple
            Arguments to pass to into Coil method function call.

        kwargs: dict
            Keyword arguments to pass to Coil method function call.

        Returns
        -------
        results: list
            List containing outputs of coil method applied
            to each member of the SymmetricCircuit
        """
        results = []
        for cl_n in [".1", ".2"]:
            coil = self.coils[self.name + cl_n]
            result = coil.__getattribute__(method_name)(*args, **kwargs)
            results.append(result)
        return results

    def set_dx(self, _dx):
        """
        Adjusts the radial thickness of all Coils in the SymmetricCircuit.

        Parameters
        ----------
        _dx: float
            The change in radial position to apply to the coils
        """
        self.apply_coil_method("set_dx", _dx)

    def set_dz(self, _dz):
        """
        Adjusts the vertical thickness of all Coils in the SymmetricCircuit.

        Parameters
        ----------
        _dz: float
            The change in vertical position to apply to the coils
        """
        self.apply_coil_method("set_dz", _dz)

    def mesh_coil(self, d_coil):
        """
        Mesh an coils in the SymmetricCircuit into smaller subcoils.

        Parameters
        ----------
        d_coil: float > 0
            The coil sub-division size
        """
        self.apply_coil_method("mesh_coil", d_coil)


class CoilSet(CoilGroup):
    """
    Poloidal field coil set

    Parameters
    ----------
    coils: Iterable[Coil]
        The list of poloidal field coils
    R_0: float
        Major radius [m] of machine (used to order coil numbers)
    d_coil: float
        Coil mesh length [m]
    """

    def __init__(self, coils, d_coil=0.5):
        super().__init__(coils)
        self._classify_control()

    @classmethod
    def from_eqdsk(cls, filename, force_symmetry=False):
        """
        Initialises a CoilSet object from an eqdsk file.

        Parameters
        ----------
        filename: str
            Filename
        force_symmetry: bool (default = False)
            Whether or not to force symmetrisation in the CoilSet
        """
        eqdsk = EQDSKInterface()
        e = eqdsk.read(filename)
        if "equilibria" not in e["name"]:
            # SCENE or CREATE
            e["dxc"] = e["dxc"] / 2
            e["dzc"] = e["dzc"] / 2

        if force_symmetry:
            return symmetrise_coilset(cls.from_group_vecs(e))
        else:
            return cls.from_group_vecs(e)

    @classmethod
    def from_group_vecs(cls, groupvecs):
        """
        Initialises an instance of CoilSet from group vectors. This has been
        implemented as a dict operation, because it will occur for eqdsks only.
        Future dict instantiation methods will likely differ, hence the
        confusing name of this method.
        """
        pfcoils = []
        cscoils = []
        passivecoils = []
        i_pf = 1
        i_cs = 1
        for i in range(groupvecs["ncoil"]):
            dx = groupvecs["dxc"][i]
            dz = groupvecs["dzc"][i]
            if abs(groupvecs["Ic"][i]) < I_MIN:
                # Catch CREATE's crap 0's
                passivecoils.append(
                    Coil(
                        groupvecs["xc"][i],
                        groupvecs["zc"][i],
                        current=0,
                        dx=dx,
                        dz=dz,
                        ctype="Passive",
                        control=False,
                    )
                )
            else:
                if dx != dz:  # Rough and ready
                    cscoils.append(
                        Coil(
                            groupvecs["xc"][i],
                            groupvecs["zc"][i],
                            groupvecs["Ic"][i],
                            dx=dx,
                            dz=dz,
                            ctype="CS",
                        )
                    )
                    i_cs += 1
                else:
                    coil = Coil(
                        groupvecs["xc"][i],
                        groupvecs["zc"][i],
                        groupvecs["Ic"][i],
                        dx=dx,
                        dz=dz,
                        ctype="PF",
                    )
                    i_pf += 1
                    coil.fix_size()  # Oh ja
                    pfcoils.append(coil)

        coils = pfcoils
        if len(cscoils) != 0:
            coils.extend(cscoils)
        coils.extend(passivecoils)
        return cls(coils)

    @property
    def n_PF(self):
        """
        The number of PF coils.
        """
        return len([c for c in self.coils.values() if c.ctype == "PF"])

    @property
    def n_CS(self):
        """
        The number of CS coils.
        """
        return len([c for c in self.coils.values() if c.ctype == "CS"])

    @property
    def n_control(self):
        """
        The length of the controls.
        """
        return sum([coil.n_control for coil in self.coils.values()])

    @property
    def n_constraints(self):
        """
        The length of the constraints.
        """
        return sum([coil.n_constraints for coil in self.coils.values()])

    def reassign_coils(self, coils):
        """
        Re-set the coils in the CoilSet.
        """
        self.coils = self.sort_coils(coils)
        self._classify_control()

    def add_coil(self, coil):
        """
        Add a coil to the CoilSet and re-order coil numbering.
        """
        super().add_coil(coil)
        self._classify_control()

    def remove_coil(self, coilname):
        """
        Remove a coil from the Coilset and re-order coil numbering.
        """
        super().remove_coil(coilname)
        self._classify_control()

    def reset(self):
        """
        Returns coilset to virgin current and size state. Positions are fixed.
        Coils are controlled by default
        """
        for coil in self.coils.values():
            coil.current = 0
            coil.control = True
            if coil.ctype == "PF":
                coil.make_size()

    def adjust_sizes(self, max_currents=None):
        """
        Resize coils based on coil currents (PF coils only)

        Parameters
        ----------
        max_currents: np.array(n_C)
            The array of peak coil currents (including CS for simplicity) [A]
        """
        for i, coil in enumerate(self.coils.values()):
            if coil.ctype == "PF":
                if max_currents is not None:
                    current = max_currents[i]
                else:
                    current = None  # defaults to self.current
                coil.make_size(current=current)
            else:
                pass

    def get_control_currents(self):
        """
        Returns a list of controlled coil currents (if coil.control == True)
        """
        return np.array([c.current for c in self.coils.values() if c.control])

    def get_control_names(self):
        """
        Returns a list of controlled coil names (if coil.control == True)
        """
        return [coil.name for coil in self.coils.values() if coil.control]

    def get_uncontrol_names(self):
        """
        Returns a list of uncontrolled coil names (if coil.control == False)
        """
        return [coil.name for coil in self.coils.values() if not coil.control]

    def get_PF_names(self):
        """
        Return a list of PF coil names
        """
        return [coil.name for coil in self.coils.values() if "PF" in coil.name]

    def get_CS_names(self):
        """
        Return a list of CS coil names
        """
        return [coil.name for coil in self.coils.values() if "CS" in coil.name]

    def get_solenoid(self):
        """
        Returns the central Solenoid object for a CoilSet
        """
        names = self.get_CS_names()
        coils = [self.coils[name].copy() for name in names]
        return Solenoid.from_coils(coils)

    def get_positions(self):
        """
        Returns the arrays of the coil centre coordinates
        """
        x, z = np.zeros(self.n_coils), np.zeros(self.n_coils)
        for i, coil in enumerate(self.coils.values()):
            x[i] = float(coil.x)
            z[i] = float(coil.z)
        return x, z

    def assign_coil_materials(self, name, j_max=None, b_max=None):
        """
        Assigns material limits to coils

        Parameters
        ----------
        name: str
            Name of the coil to assign the material to
            from ['PF', 'CS', 'PF_x', 'CS_x'] with x a valid str(int)
        j_max: float (default None)
            Overwrite default constant material max current density [MA/m^2]
        b_max: float (default None)
            Overwrite default constant material max field [T]
        """
        if name == "PF":
            names = self.get_PF_names()
            for name in names:
                self.assign_coil_materials(name, j_max=j_max, b_max=b_max)
        elif name == "CS":
            names = self.get_CS_names()
            for name in names:
                self.assign_coil_materials(name, j_max=j_max, b_max=b_max)
        else:
            self.coils[name].assign_material(j_max=j_max, b_max=b_max)

    def get_max_fields(self):
        """
        Returns a vector of the maximum magnetic fields

        Returns
        -------
        b_max: np.array(self.n_C)
            An array of maximum field values [T]
        """
        b_max = np.zeros(len(self.coils.values()))
        for i, coil in enumerate(self.coils.values()):
            b_max[i] = coil.b_max
        return b_max

    def get_max_currents(self, max_pf_current):
        """
        Returns a vector of the maximum coil currents. If a PF coil size is
        fixed, will return the correct maximum, overwriting the input maximum.

        Parameters
        ----------
        max_pf_current: float or None
            Maximum PF coil current [A]

        Returns
        -------
        max_currents: np.array(n_C)
        """
        max_currents = max_pf_current * np.ones(self.n_coils)
        pf_names = self.get_PF_names()
        for i, name in enumerate(pf_names):
            if self.coils[name].flag_sizefix:
                max_currents[i] = self.coils[name].get_max_current()

        cs_names = self.get_CS_names()
        for i, name in enumerate(cs_names):
            max_currents[self.n_PF + i] = self.coils[name].get_max_current()
        return max_currents

    def adjust_currents(self, current_change):
        """
        Modify currents in coils: [I]<--[I]+[dI]
        """
        for coil, d_i in zip(self._ccoils, current_change):
            coil.adjust_current(d_i)

    def set_control_currents(self, currents, update_size=True):
        """
        Sets the currents in the coils being controlled
        """
        for coil, current in zip(self._ccoils, currents):
            coil.set_current(current)
        if update_size:
            self.adjust_sizes()
            self.mesh_coils()

    def adjust_positions(self, position_changes):
        """
        Modify coil positions: [X]<--[X]+[dX]
        """
        for coil, position in zip(self._ccoils, position_changes):
            coil.adjust_position(*position)

    def set_positions(self, positions):
        """
        Sets the positions of the coils being controlled
        """
        if isinstance(positions, list):
            self._set_positions_list(positions)
        elif isinstance(positions, dict):
            self._set_positions_dict(positions)
        else:
            raise TypeError("Coil positions type should be either list or dict")

    def _set_positions_list(self, positions):
        for coil, position in zip(self._ccoils, positions):
            coil.set_position(*position)

    def _set_positions_dict(self, positions):
        for coil in self._ccoils:
            if coil.name in positions:
                coil.set_position(*positions[coil.name])

    def plot(self, ax=None, snap=None, subcoil=True, **kwargs):
        """
        Plots the CoilSet object onto `ax`
        """
        if snap is not None:
            for name, coil in self.coils.items():
                coil.current = self.Iswing[name][snap]
        return CoilSetPlotter(self, ax=ax, subcoil=subcoil, **kwargs)


def _get_symmetric_coils(coilset):
    """
    Coilset symmetry utility
    """
    x, z, dx, dz, currents = coilset.to_group_vecs()
    coil_matrix = np.array([x, np.abs(z), dx, dz, currents]).T

    sym_stack = [[coil_matrix[0], 1]]
    for i in range(1, len(x)):
        coil = coil_matrix[i]

        for j, sym_coil in enumerate(sym_stack):
            if np.allclose(coil, sym_coil[0]):
                sym_stack[j][1] += 1
                break

        else:
            sym_stack.append([coil, 1])

    return sym_stack


def check_coilset_symmetric(coilset):
    """
    Check whether or not a CoilSet is purely symmetric about z=0.

    Parameters
    ----------
    coilset: CoilSet
        CoilSet to check for symmetry

    Returns
    -------
    symmetric: bool
        Whether or not the CoilSet is symmetric about z=0
    """
    sym_stack = _get_symmetric_coils(coilset)
    for coil, count in sym_stack:
        if count != 2:
            if not np.isclose(coil[1], 0.0):
                # z = 0
                return False
    return True


def symmetrise_coilset(coilset):
    """
    Symmetrise a CoilSet by converting any coils that are up-down symmetric about
    z=0 to SymmetricCircuits.

    Parameters
    ----------
    coilset: CoilSet
        CoilSet to symmetrise

    Returns
    -------
    symmetric_coilset: CoilSet
        New CoilSet with SymmetricCircuits where appropriate
    """
    if not check_coilset_symmetric(coilset):
        bluemira_warn(
            "Symmetrising a CoilSet which is not purely symmetric about z=0. This can result in undesirable behaviour."
        )
    coilset = coilset.copy()

    sym_stack = _get_symmetric_coils(coilset)
    counts = np.array(sym_stack, dtype=object).T[1]

    new_coils = []
    for coil, count in zip(coilset.coils.values(), counts):
        if count == 1:
            new_coils.append(coil)
        elif count == 2:
            if isinstance(coil, SymmetricCircuit):
                new_coils.append(coil)
            elif isinstance(coil, Coil):
                new_coils.append(SymmetricCircuit(coil))
            else:
                raise EquilibriaError(f"Unrecognised class {coil.__class__.__name__}")
        else:
            raise EquilibriaError("There are super-posed Coils in this CoilSet.")

    return CoilSet(new_coils)

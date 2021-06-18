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
Coil and coil grouping objects
"""

from copy import deepcopy
import numpy as np
from scipy.interpolate import RectBivariateSpline
from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_warn
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
    NB3SN_B_MAX,
    NB3SN_J_MAX,
    X_TOLERANCE,
)
from bluemira.equilibria.file import EQDSKInterface


PF_COIL_NAME = "PF_{}"
CS_COIL_NAME = "CS_{}"


def make_coil_corners(x_c, z_c, dx, dz):
    """
    Make coil x, z corner vectors (ANTI-CLOCKWISE).
    """
    xx, zz = np.ones(4) * x_c, np.ones(4) * z_c
    x = xx + dx * np.array([-1, 1, 1, -1])
    z = zz + dz * np.array([-1, -1, 1, 1])
    return x, z


def name_coil(coil, i):
    """
    Name a coil based on its type and type-number. The coil naming convention
    is not directly enforced here.
    """
    if coil.ctype == "CS":
        return CS_COIL_NAME.format(i)
    else:
        return PF_COIL_NAME.format(i)


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
        n_turns=1,
        control=True,
        ctype="PF",
        j_max=NBTI_J_MAX,
        b_max=NBTI_B_MAX,
        name=None,
        **kwargs,
    ):

        self.x = x
        self.z = z
        self.current = current
        self.j_max = j_max
        self.b_max = b_max

        # Default: free sizes for PF coils
        self.flag_sizefix = kwargs.get("flag_sizefix", False)

        if "dx" and "dz" not in kwargs:
            self.make_size()
        else:
            self.dx, self.dz = kwargs["dx"], kwargs["dz"]
            self._make_corners()
        self.n_filaments = kwargs.get("n_filaments", 1)  # Number of filaments
        self.n_turns = n_turns
        self.control = control
        self.ctype = ctype
        if name is None:
            name = "Coil"
        self.name = name
        self.sub_coils = None

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
            if self.ctype == "PF":
                self.make_size()
            if self.ctype == "CS":
                self._make_corners()
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
        self._make_corners()

    def set_dz(self, dz):
        """
        Adjusts the vertical thickness of the Coil object (meshing not handled)
        """
        self.dz = dz
        self._make_corners()

    def make_size(self, current=None):
        """
        Size the coil based on a current and a current density.
        """
        if self.flag_sizefix is False:
            if current is None:
                current = self.current

            half_width = (abs(current) / (1e6 * self.j_max)) ** 0.5 / 2
            self.dx, self.dz = half_width, half_width
            self._make_corners()
            self.sub_coils = None  # Need to re-mesh if this is what you want
        else:
            pass

    def _make_corners(self):
        """
        Makes the coil corner vectors
        """
        self.rc = np.sqrt(self.dx ** 2 + self.dz ** 2) / 2
        self.x_corner, self.z_corner = make_coil_corners(
            self.x, self.z, self.dx, self.dz
        )

    def fix_size(self):
        """
        Fixes the size of the coil
        """
        self.flag_sizefix = True

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

    def assign_material(self, material, j_max=None, b_max=None):
        """
        Assigns EM material properties to coil

        Parameters
        ----------
        material: str
            The name of the material from ['NbTi', 'Nb3Sn']
        j_max: float (default None)
            Overwrite default constant material max current density [MA/m^2]
        b_max: float (default None)
            Overwrite default constant material max field [T]
        """
        if material == "NbTi":
            self.j_max = NBTI_J_MAX
            self.b_max = NBTI_B_MAX
        elif material == "Nb3Sn":
            self.j_max = NB3SN_J_MAX
            self.b_max = NB3SN_B_MAX
        else:
            raise EquilibriaError(f"Unrecognised coil material: {material}.")
        if j_max is not None:
            self.j_max = j_max
        if b_max is not None:
            self.b_max = b_max

    def get_max_current(self):
        """
        Gets the maximum current for a coil with a specified size

        Returns
        -------
        Imax: float
            The maximum current that can be produced by the coil [A]
        """
        if self.ctype == "CS" or (self.ctype == "PF" and self.flag_sizefix is True):
            return self._get_max_current(self.dx, self.dz)  # JIC

        raise EquilibriaError(
            "Only CS coils have a max current and the size of"
            " this coil has not been fixed."
        )

    def _get_max_current(self, dx, dz):
        """
        get_max_current without the safety net.

        Returns
        -------
        Imax: float
            The maximum current that can be produced by the coil [A]

        """
        return abs(self.j_max * 1e6 * (4 * dx * dz))

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
                sub_name = self.name + "_{:1.0f}".format(i + 1)
                c = Coil(
                    xc,
                    zc,
                    current,
                    Nf=1,
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
        if self.sub_coils is None:  # Filamentary Green's coil function
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

        Note
        ----
        Los resultos serán una mierda pinchada en un palo si
        el Grid no es sufficientemente grande! Solo usar si sabes lo que estas
        haciendo.
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
            # acaba sendo zero de qualquer maneira..
            Bx = 0  # veranlasst aber geistige Gesundheit!
        else:
            Bz = coil.control_Bz(self.x, self.z)
            Bx = coil.control_Bx(self.x, self.z)  # Oh doch mein Freundchen!
        return 2 * np.pi * self.x * np.array([Bz, -Bx])  # angry B

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
        return "Coil(" + self.__str__() + ")"

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

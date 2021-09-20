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
import types
from scipy.interpolate import RectBivariateSpline
from BLUEPRINT.magnetostatics.greens import (
    greens_psi,
    greens_Bx,
    greens_Bz,
)
from BLUEPRINT.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz
from BLUEPRINT.equilibria.plotting import (
    CoilSetPlotter,
    PulsePlotter,
    CoilPlotter,
    PlasmaCoilPlotter,
)
from bluemira.base.constants import MU_0
from BLUEPRINT.equilibria.constants import (
    I_MIN,
    J_TOR_MIN,
    NBTI_B_MAX,
    NBTI_J_MAX,
    NB3SN_B_MAX,
    NB3SN_J_MAX,
    X_TOLERANCE,
)
from BLUEPRINT.equilibria.eqdsk import EQDSKInterface
from bluemira.base.look_and_feel import bluemira_warn
from BLUEPRINT.geometry.loop import Loop


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
        Coil geometric centre X coordinate [m]
    z: float
        Coil geometric centre Z coordinate [m]
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
        "_current",
        "_dx",
        "_dz",
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
        self._current = current
        self.j_max = j_max
        self.b_max = b_max

        # Default: free sizes for PF coils
        self.flag_sizefix = kwargs.get("flag_sizefix", False)

        if "dx" and "dz" not in kwargs:
            self.make_size()
        else:
            self._dx, self._dz = kwargs["dx"], kwargs["dz"]
            self._make_corners()
        self.n_filaments = kwargs.get("n_filaments", 1)  # Number of filaments
        self.n_turns = n_turns
        self.control = control
        self.ctype = ctype
        if name is None:
            name = "Coil"

        self.name = name
        self.sub_coils = None

    @property
    def current(self):
        """
        Get current.
        """
        return self._current

    @current.setter
    def current(self, val):
        """
        Sets the current in a Coil object.

        Parameters
        ----------
        val: float
            The current to set in the coil
        """
        self._current = val
        if self.sub_coils is not None:
            for coil in self.sub_coils.values():
                coil.current = self._current / len(self.sub_coils)

    @property
    def dx(self):
        """
        Get dx.
        """
        return self._dx

    @dx.setter
    def dx(self, val):
        """
        Adjusts the vertical thickness of the Coil object (meshing not handled).
        """
        self._dx = val
        self._make_corners()

    @property
    def dz(self):
        """
        Get dz.
        """
        return self._dz

    @dz.setter
    def dz(self, val):
        """
        Adjusts the radial thickness of the Coil object (meshing not handled).
        """
        self._dz = val
        self._make_corners()

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
            self.dz = dz
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

    def make_size(self, current=None):
        """
        Calculate coil corner locations (ANTI-CLOCKWISE ordered)
        """
        if self.flag_sizefix is False:
            if current is None:
                current = self.current

            half_width = (abs(current) / (1e6 * self.j_max)) ** 0.5 / 2
            self._dx, self._dz = half_width, half_width
            self._make_corners()
            self.sub_coils = None  # Need to re-mesh if this is what you want
        else:
            pass

    def _make_corners(self):
        """
        Makes the coil corner vectors
        """
        self.rc = np.sqrt(self._dx ** 2 + self._dz ** 2) / 2
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
            raise ValueError("Este material ainda não existe..")
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

        raise ValueError(
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
        coil: Coil
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

        Parameters
        ----------
        ax: axis object
            Matplotlib axis object, optional
        subcoil: bool
            Whether or not to plot the Coil subcoils
        **kwargs
            arguments passed to Matplotlib

        Returns
        -------
        CoilPlotter

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


class CoilGroup:
    """
    Abstract grouping of Coil objects
    """

    def __getitem__(self, name):
        """
        list-like behaviour for CoilSet object
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

    def circuit_expander(self):
        """
        Expand circuits into their constituent coils.

        Returns
        -------
        coils: dict

        extra_coils: dict
            A dictionary containing the number of extra CS and PF coils from splitting
        """
        coils = {}
        extra_coils = {"PF": 0, "CS": 0}
        for coil in self.coils.values():
            if hasattr(coil, "splittable") and coil.splittable == True:
                for ii, circuit_coil in enumerate(coil.split()):
                    coils[circuit_coil.name] = circuit_coil
                    if ii > 0:
                        extra_coils[circuit_coil.ctype] += 1
            else:
                coils[coil.name] = coil
        return coils, extra_coils

    def splitter(self, split_circuits):
        """
        Split circuits and add new coils to Group.

        Parameters
        ----------
        split_circuits: bool
            Split circuits or not

        """
        if split_circuits:
            self.coils, extra_coils = self.circuit_expander()
            self.n_PF += extra_coils["PF"]
            self.n_CS += extra_coils["CS"]
            self.n_coils += extra_coils["PF"] + extra_coils["CS"]

    def to_dict(self, split_circuits=True):
        """
        Convert Group to dictionary.

        Parameters
        ----------
        split_circuits: bool, optional
            split circuit coils into the constituent coils, by default True

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

        if split_circuits:
            coils, _ = self.circuit_expander()
        else:
            coils = self.coils

        for name, coil in coils.items():
            cdict[name] = coil.to_dict()
        return cdict

    def copy(self):
        """
        Get a deep copy of the CoilGroup.
        """
        return deepcopy(self)

    def to_group_vecs(self, split_circuits=True):
        """
        Convert CoilGroup properties to numpy arrays

        Parameters
        ----------
        split_circuits: bool, optional
            split circuit coils into the constituent coils, by default True.

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
        if split_circuits:
            coils, extra_coils = self.circuit_expander()
            n_coils = self.n_coils + extra_coils["PF"] + extra_coils["CS"]
        else:
            coils = self.coils
            n_coils = self.n_coils

        x, z = np.zeros(n_coils), np.zeros(n_coils)
        dx, dz = np.zeros(n_coils), np.zeros(n_coils)
        currents = np.zeros(n_coils)
        for i, coil in enumerate(coils.values()):
            x[i], z[i] = coil.x, coil.z
            dx[i], dz[i] = coil.dx, coil.dz
            currents[i] = coil.current
        return x, z, dx, dz, currents

    def add_coil(self, coil):
        """
        Adds a coil to the CoilGroup

        Parameters
        ----------
        coil: Coil object
            The coil to be added to the CoilGroup
        """
        self.coils[coil.name] = coil

    def remove_coil(self, coilname):
        """
        Removes a coil from the CoilGroup

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
        forces = np.zeros((self.n_C, 2))
        for i, coil in enumerate(self.coils.values()):
            forces[i, :] = coil.F(eqcoil)
        return forces

    def toggle_control(self, *name):
        """
        Toggles the control of a coil in a CoilGroup. Tracks control of a coil
        in the CoilGroup
        """
        for n in name:
            self.coils[n].toggle_control()
        self._classify_control()

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
            self.plasma_Bp = np.sqrt(Bx ** 2 + Bz ** 2)

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
        array = np.zeros_like(x)
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
            return np.sqrt(self.Bx(x, z) ** 2 + self.Bz(x, z) ** 2)

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


class Solenoid(CoilGroup):
    """
    Solenoid object for a vertically arranged stack of PF coils. Will default
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

    control = True  # List sort utility (also kind of true)

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
        coils: list(Coil, Coil, ..)
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
            coil.name = name_coil(coil, i + 1)

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

    def psi(self, x, z):
        """
        Poloidal magnetic flux at x, z
        """
        return self._sum_all(self.coils, "psi", x, z)

    def mesh_coils(self, d_coil):
        """
        Sub-divide coils into subcoils based on size. Coils are meshed within
        the Coil object.
        """
        for coil in self.coils:
            coil.mesh_coil(d_coil)

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


class CoilClassifier:
    """
    Coil classification tool

    Parameters
    ----------
    R_0: float
        Machine major radius [m]

    Gives coils a name and number based on their position relative to the
    major radius:
        - PF coils: clockwise ITER-like convention [1..] (some things don't change)
        - CS coils: top to bottom [1...]
    Gives coils an index for its position in the constraint vectors:
        - PF coils [0, 1, .. nPF-1]
        - CS coils [nPF, nPF+1, .. nPF+nCS-1]
    """

    def __init__(self, R_0):
        self.R_0 = R_0

    def _purge(self):
        """
        Zustandsreinigung
        """
        for attribute in ["n_PF", "n_CS", "n_C"]:
            if hasattr(self, attribute):
                delattr(self, attribute)

    def __call__(self, coils):
        """
        Returns sorted dict of coils, and n_PF, n_CS, n_C
        """
        self._purge()
        ccoils = [c for c in coils if c.control]
        ucoils = [c for c in coils if not c.control]
        coil_dict = self.classify_coils(ccoils)
        ucoil_dict = self.handle_passive(ucoils)
        all_coil_dict = {**coil_dict, **ucoil_dict}  # Rely on 3.6 dict order..
        return all_coil_dict, self.n_PF, self.n_CS, self.n_coils

    def handle_passive(self, coils):
        """
        Unpack passive coil objects
        """
        p_coils = self._sort_PF_coils(coils, self.R_0)
        return self._make_PF_dict(p_coils, ioffset=self.n_PF)

    def classify_coils(self, coils):
        """
        Unpacks Coil and Solenoid objects
        """
        pf_coils, cs_coils = [], []
        npf, ncs, j = 0, 0, 0
        solenoid = False
        for coil in coils:
            if isinstance(coil, Solenoid):
                solenoid = coil
                j += 1
            elif coil.ctype == "PF":
                npf += 1
                pf_coils.append(coil)
            elif coil.ctype == "CS":
                cs_coils.append(coil)
            else:
                bluemira_warn("Unbekannte Magnetspulart.")
        self.n_PF = npf
        pf_coils = self._sort_PF_coils(pf_coils, self.R_0)
        pf_coils = self._make_PF_dict(pf_coils)
        if len(cs_coils) != 0:
            cs_coils = sorted(cs_coils, key=lambda coil_: coil_.z)
            solenoid = Solenoid.from_coils(cs_coils)
            cs_coils = []
            j += 1
        if j < 1:
            bluemira_warn("No CS specified.")
        elif j > 1:
            raise ValueError("More than 1 CS specified.")
        if solenoid:
            for coil in solenoid.coils:
                ncs += 1
                cs_coils.append(coil)
            self.n_CS = ncs
            self.n_coils = self.n_CS + self.n_PF
            cs_coils = self._sort_CS_coils(cs_coils)
            cs_coils = self._make_CS_dict(cs_coils)
            return {**pf_coils, **cs_coils}
        else:
            self.n_CS = 0
            self.n_coils = self.n_PF
            return pf_coils

    @staticmethod
    def _make_PF_dict(pf_coils, ioffset=0):
        """
        Makes a dictionary of PF coils
        """
        pf = {}
        for i, coil in enumerate(pf_coils):
            if coil.control:
                name = name_coil(coil, i + 1 + ioffset)
                coil.number = i + 1
                coil.index = i
                coil.name = name
                pf[name] = coil
        return pf

    def _make_CS_dict(self, cs_coils):
        """
        Makes a dictionary of CS coils
        """
        cs = {}
        for i, coil in enumerate(cs_coils):
            name = name_coil(coil, i + 1)
            coil.number = i + 1
            coil.index = self.n_PF + i
            coil.name = name
            cs[name] = coil
        return cs

    @staticmethod
    def _sort_PF_coils(coils, R_0):
        """
        Sort PF coils clockwise from first top left (ITER-like)
        """
        theta = []
        for coil in coils:
            theta.append(np.arctan2(coil.z - 0, coil.x - R_0))
        b = np.argsort(theta)[::-1]
        coils = np.array(coils)
        return list(coils[b])

    @staticmethod
    def _sort_CS_coils(coils):
        """
        Sorts CS coil modules vertically from top to bottom (ITER-like)
        """
        z = []
        for coil in coils:
            z.append(coil.z)
        b = np.argsort(z)[::-1]
        coils = np.array(coils)
        return list(coils[b])


class CoilSet(CoilGroup):
    """
    Poloidal field coil set

    Parameters
    ----------
    coils: list(Coil, Coil, ..)
        The list of poloidal field coils
    R_0: float
        Major radius [m] of machine (used to order coil numbers)
    d_coil: float
        Coil mesh length [m]
    """

    def __init__(self, coils, R_0, d_coil=0.5):
        self._classifier = CoilClassifier(R_0)
        self.n_coils = None
        self.n_CS = None
        self.n_PF = None

        self.coils = self.sort_coils(coils)
        self._classify_control()

    @classmethod
    def from_eqdsk(cls, filename):
        """
        Initialises a CoilSet object from an eqdsk file.
        """
        eqdsk = EQDSKInterface()
        e = eqdsk.read(filename)
        if "equilibria" not in e["name"]:
            # SCENE or CREATE
            e["dxc"] = e["dxc"] / 2
            e["dzc"] = e["dzc"] / 2
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
                        ctype="PF",
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
                else:
                    coil = Coil(
                        groupvecs["xc"][i],
                        groupvecs["zc"][i],
                        groupvecs["Ic"][i],
                        dx=dx,
                        dz=dz,
                        ctype="PF",
                    )
                    coil.fix_size()  # Oh ja
                    pfcoils.append(coil)
        R_0 = groupvecs["xgrid1"] + groupvecs["xdim"] / 2  # Rough and ready
        coils = pfcoils
        if len(cscoils) != 0:
            solenoid = Solenoid.from_coils(cscoils)
            coils.append(solenoid)
        coils.extend(passivecoils)
        return cls(coils, R_0)

    def sort_coils(self, coils):
        """
        Sorts coils using a CoilClassifier Object which returns a dict of
        arrange PF and CS coil objects.
        Coil type-numbers also extracted and assigned to self
        """
        coils, self.n_PF, self.n_CS, self.n_coils = self._classifier(coils)
        return coils

    def reassign_coils(self, coils):
        """
        Re-set the coils in the CoilSet.
        """
        self.coils = coils
        self._classify_control()

    def add_coil(self, coil):
        """
        Add a coil to the CoilSet and re-order coil numbering.
        """
        super().add_coil(coil)
        self.coils = self.sort_coils(self.coils.values())
        self._classify_control()

    def remove_coil(self, coilname):
        """
        Remove a coil from the Coilset and re-order coil numbering.
        """
        super().remove_coil(coilname)
        self.coils = self.sort_coils(self.coils.values())
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

    def assign_coil_materials(self, name, material="Nb3Sn", j_max=None, b_max=None):
        """
        Assigns material limits to coils

        Parameters
        ----------
        name: str
            Name of the coil to assign the material to
            from ['PF', 'CS', 'PF_x', 'CS_x'] with x a valid str(int)
        material: str
            Name of the material
            from ['NbTi', 'Nb3Sn']
        j_max: float (default None)
            Overwrite default constant material max current density [MA/m^2]
        b_max: float (default None)
            Overwrite default constant material max field [T]
        """
        if name == "PF":
            names = self.get_PF_names()
            for n in names:
                self.assign_coil_materials(n, material, j_max=j_max, b_max=b_max)
        elif name == "CS":
            names = self.get_CS_names()
            for n in names:
                self.assign_coil_materials(n, material, j_max=j_max, b_max=b_max)
        else:
            self.coils[name].assign_material(material, j_max=j_max, b_max=b_max)

    def get_max_fields(self):
        """
        Returns a vector of the maximum magnetic fields

        Returns
        -------
        b_max: np.array(self.n_C)
            An array of maximum field values [T]
        """
        b_max = np.zeros(self.n_coils)
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
            coil.current += d_i

    def set_control_currents(self, currents, update_size=True):
        """
        Sets the currents in the coils being controlled
        """
        for coil, current in zip(self._ccoils, currents):
            coil.current = current
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

    def plot_pulse(self, ax=None):
        """
        Plots coilset currents throughout a pulse
        """
        return PulsePlotter(self, ax=ax)

    def generate_cross_sections(
        self,
        mesh_sizes=None,
        geometry_names=None,
        verbose=True,
        split_circuits=True,
    ):
        """
        Generate the meshed `CrossSection` for this CoilSet

        This cleans the Loop representing the coil and any sub coils based on the
        `min_length` and `min_angle` using the
        :func:`~BLUEPRINT.geometry.geomtools.clean_loop_points` algorithm.

        The clean points are then fed into a sectionproperties `CustomSection` object,
        with corresponding facets and control point. The geometry is cleaned, using the
        sectionproperties `clean_geometry` method, before creating a mesh and loading the
        mesh and geometry into a sectionproperties `CrossSection`.

        Parameters
        ----------
        mesh_sizes : List[float], optional
            The mesh sizes to use for the sectionproperties meshing algorithm,
            by default None. If None then the minimium length between nodes on the Loop
            is used.
        geometry_names : List[str], optional
            A list of names in the system's geometry dictionary to generate
            cross-sections for.
            If None then all geometries for analysis in the X-Z plane will be
            used, by default None.
        verbose : bool, optional
            Determines if verbose mesh cleaning output should be provided,
            by default True.
        split_circuits: bool, optional
            Split circuit coils into individual coils, by default True.

        Returns
        -------
        cross_sections : List[sectionproperties.analysis.cross_section.CrossSection]
            The resulting `CrossSection` objects from meshing the cleaned loops.
        loops : List[Loop]
            The loop geometries used to generate the `CrossSection` objects.
        """
        cross_sections = []
        loops = []
        if split_circuits:
            coils, _ = self.circuit_expander()
            coils = coils.values()
        else:
            coils = self.coils.values()
        if geometry_names:
            coils = [self.coils[name] for name in geometry_names]
        for coil in coils:
            loop = Loop(coil.x_corner, coil.z_corner)
            loop.close()
            (cross_section, loop) = loop.generate_cross_section(
                mesh_sizes=mesh_sizes, verbose=verbose
            )
            cross_sections += [cross_section]
            loops += [loop]
        return cross_sections, loops


class SymmetricCircuit(Coil):
    """
    Represents a set of coils, symmetric about z = 0, connected in a circuit.
    The coils are identical except for their z position.

    Parameters
    ----------
    *args
        Coil positional arguments
    **kwargs
        Coil keyword arguments

    Notes
    -----
    At the moment, only represents two symmetric coils but should be easy
    to adapt to a generic case. Possibly need to change the initialisation to take
    coil objects rather than paramaters for a single coil. The other option is to take
    a list of positional parameters to iterate over. Also consider if
    ForceField class needs adapting to work with such a case.
    Fractional current is another extension to the class.

    If any of these are implemented a subclass that is based on a generic circuit
    could be made where the generic circuit probably will contain most of the current
    functionality.

    """

    __slots__ = ["_n_coils", "_pm", "_meshed", "splittable"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.splittable = True
        self._n_coils = 2
        self._pm = "±"
        self._meshed = False

    def split(self):
        """
        Split a circuit into its individual coils.

        Returns
        -------
        list
          An single element list of the coil if not splittable

        Yields
        ------
        Coil-like
            a split circuit coil

        """
        if not self.splittable:
            return [self]
        for n_c in range(self._n_coils):
            coil = self.copy()
            coil._set_properties(n_c)
            coil.splittable = False
            yield coil

    def _set_properties(self, coil_number):
        """
        Explicitly set properties of the coil that are implicit in the circuit.

        Parameters
        ----------
        coil_number: int
            coil number within circuit (0 indexed)

        """
        if coil_number == 1:
            self._set_coil2_properties(existing_copy=True)

        self._pm = ""

        # Use Coil methods instead of circuit methods
        for method in [
            "_control_Bx_greens",
            "_control_Bx_analytical",
            "_control_Bz_greens",
            "_control_Bz_analytical",
            "_points_inside_coil",
            "control_psi",
            "mesh_coil",
            "plot",
        ]:
            setattr(self, method, types.MethodType(getattr(Coil, method), self))

    def _points_inside_coil(self, x, z):
        """
        Determine which points lie inside or on the coil boundaries.

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
        return (x >= x_min) & (x <= x_max) & (abs(z) >= z_min) & (abs(z) <= z_max)

    def control_psi(self, x, z):
        """
        Calculate poloidal flux at (x, z) due to a unit current

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinates to check
        z: Union[float, np.array]
            The z coordinates to check

        Returns
        -------
        gpsi: float or array(N, M)
            The total poloidal flux at (x, z) as a sum of contributions from symmetrical
            coils with unit current in a circuit.
        """
        if self.sub_coils is None:  # Filamentary Green's coil function
            return (
                greens_psi(self.x, self.z, x, z, self.dx, self.dz) * self.n_turns
                + greens_psi(self.x, -(self.z), x, z, self.dx, self.dz) * self.n_turns
            )

        gpsiu = [greens_psi(c.x, c.z, x, z, c.dx, c.dz) for c in self.sub_coils.values()]
        gpsil = [
            greens_psi(c.x, -(c.z), x, z, c.dx, c.dz) for c in self.sub_coils.values()
        ]
        gpsit = [sum(i) for i in zip(gpsiu, gpsil)]
        return sum(gpsit) / self.n_filaments

    def _control_Bx_greens(self, x, z):
        """
        Calculate radial magnetic field Bx respose at (x, z) due to a unit
        current using Green's functions.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinates to check
        z: Union[float, np.array]
            The z coordinates to check

        Returns
        -------
        gBx: float or array(N * M)
            The total radial magnetic field at (x, z) as a sum of contributions from
            symmetrical coils with unit current in a circuit.
        """
        if self.sub_coils is None:
            return (
                greens_Bx(self.x, self.z, x, z) * self.n_turns
                + greens_Bx(self.x, -(self.z), x, z) * self.n_turns
            )

        gxu = [greens_Bx(c.x, c.z, x, z) for c in self.sub_coils.values()]
        gxl = [greens_Bx(c.x, -(c.z), x, z) for c in self.sub_coils.values()]
        gxt = [sum(i) for i in zip(gxu, gxl)]
        return sum(gxt) / self.n_filaments

    def _control_Bz_greens(self, x, z):
        """
        Calculate vertical magnetic field Bz at (x, z) due to a unit current

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinates to check
        z: Union[float, np.array]
            The z coordinates to check

        Returns
        -------
        gBz: float or array(N * M)
            The total vertical magnetic field at (x, z) as a sum of contributions from
            symmetrical coils with unit current in a circuit.
        """
        if self.sub_coils is None:
            return (
                greens_Bz(self.x, self.z, x, z) * self.n_turns
                + greens_Bz(self.x, -(self.z), x, z) * self.n_turns
            )

        gzu = [greens_Bz(c.x, c.z, x, z) for c in self.sub_coils.values()]
        gzl = [greens_Bz(c.x, -(c.z), x, z) for c in self.sub_coils.values()]
        gzt = [sum(i) for i in zip(gzu, gzl)]
        return sum(gzt) / self.n_filaments

    def _control_Bx_analytical(self, x, z):
        """
        Calculate radial magnetic field response at (x, z) due to a unit
        current using semi-analytic method.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinates to check
        z: Union[float, np.array]
            The z coordinates to check

        Returns
        -------
        Bx: Union[float, array(N, M)]
            The radial magnetic field at (x, z) as a sum of contributions
            from symmetrical coils with unit current in a circuit. These are calculated
            semi-analytically for the purposes of x and z being inside the coils.
        """
        Bxu = semianalytic_Bx(self.x, self.z, x, z, d_xc=self.dx, d_zc=self.dz)
        Bxl = semianalytic_Bx(self.x, -(self.z), x, z, d_xc=self.dx, d_zc=self.dz)
        return Bxu + Bxl

    def _control_Bz_analytical(self, x, z):
        """
        Calculate vertical magnetic field response at (x, z) due to a unit
        current using semi-analytic method.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinates to check
        z: Union[float, np.array]
            The z coordinates to check

        Returns
        -------
        Bz: Union[float, array(N, M)]
            The vertical magnetic field at (x, z) as a sum of contributions
            from symmetrical coils with unit current in a circuit. These are calculated
            semi-analytically for the purposes of x and z being inside the coils.
        """
        Bzu = semianalytic_Bz(self.x, self.z, x, z, d_xc=self.dx, d_zc=self.dz)
        Bzl = semianalytic_Bz(self.x, -(self.z), x, z, d_xc=self.dx, d_zc=self.dz)
        return Bzu + Bzl

    def __str__(self):
        """
        Pretty circuit printing for debug.
        """
        return (
            f"X={self.x:.2f} m, Z={self._pm}{self.z:.2f} m, I={self.current/1e6:.2f} MA "
            f"control={self.control}"
        )

    def mesh_coil(self, d_coil):
        """
        Mesh a coil as done in Coil.
        """
        super().mesh_coil(d_coil)
        self._meshed = d_coil

    def _remesh(self):
        """
        Mesh a coil with the last meshing value
        if the coil has been meshed.
        """
        if self._meshed:
            self.mesh_coil(self._meshed)

    def _set_coil2_properties(self, coil_number=1, existing_copy=False):
        """
        Set the properties of the virtual coil on a copy of the
        current instance of the circuit.

        Parameters
        ----------
        coil_number: int
            number of coil
        existing_copy: bool
            make a copy of the current instance, default is False

        Returns
        -------
        coil2: instance
            instance of SymmetricCircuit

        """
        coil2 = self if existing_copy else self.copy()
        coil2.z = -coil2.z
        coil2._make_corners()
        coil2._remesh()
        coil2.name += f".{coil_number}"
        return coil2

    def plot(self, ax=None, subcoil=True, **kwargs):
        """
        Plots the Coil object onto `ax`. Should only be used for individual
        coils. Use CoilSet.plot() for CoilSet objects

        Parameters
        ----------
        ax: axis object
            Matplotlib axis object, optional
        subcoil: bool
            Whether or not to plot the Coil subcoils
        **kwargs
            arguments passed to Matplotlib

        Returns
        -------
        CoilPlotter

        """
        ret_ax = super().plot(ax=ax, subcoil=subcoil, **kwargs)
        return super(SymmetricCircuit, self._set_coil2_properties()).plot(
            ax=ret_ax(), subcoil=subcoil, **kwargs
        )


if __name__ == "__main__":
    from BLUEPRINT import test

    test()

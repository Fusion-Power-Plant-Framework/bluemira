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

import abc
from contextlib import suppress
from copy import deepcopy
from enum import Enum, auto
from re import split
from typing import Any, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.constants import I_MIN, NBTI_B_MAX, NBTI_J_MAX, X_TOLERANCE
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.plotting import CoilPlotter, CoilSetPlotter
from bluemira.magnetostatics.greens import (
    circular_coil_inductance_elliptic,
    greens_Bx,
    greens_Bz,
    greens_psi,
)
from bluemira.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz
from bluemira.utilities.tools import is_num


class CoilType(Enum):
    PF = auto()
    CS = auto()
    NONE = auto()


class CoilFieldsMixin:
    def mesh_coil(self):
        pass

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

    def control_psi(self, x, z):
        """
        Calculate poloidal flux at (x, z) due to a unit current
        """
        if self.sub_coils is None:
            return greens_psi(self.x, self.z, x, z, self.dx, self.dz) * self.n_turns

        gpsi = [greens_psi(c.x, c.z, x, z, c.dx, c.dz) for c in self.sub_coils.values()]
        return sum(gpsi) / self.n_filaments

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

    def Bp(self, x, z):
        """
        Calculate poloidal magnetic field Bp at (x, z)
        """
        return np.hypot(
            self.control_Bx(x, z) * self.current, self.control_Bz(x, z) * self.current
        )

    def _mix_control_method(self, x, z, greens_func, semianalytic_func):
        """
        Boiler-plate helper function to mixed the Green's function responses
        with the semi-analytic function responses, as a function of position
        outside/inside the coil boundary.
        """
        x, z = np.ascontiguousarray(x), np.ascontiguousarray(z)
        if np.isclose(self.dx, 0.0) or np.isclose(self.dz, 0.0):
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

    def F(self, eqcoil):  # noqa :N802
        """
        Calculate the force response at the coil centre including the coil
        self-force.

        \t:math:`\\mathbf{F} = \\mathbf{j}\\times \\mathbf{B}`\n
        \t:math:`F_x = IB_z+\\dfrac{\\mu_0I^2}{4\\pi X}\\textrm{ln}\\bigg(\\dfrac{8X}{r_c}-1+\\xi/2\\bigg)`\n
        \t:math:`F_z = -IBx`
        """  # noqa :W505
        Bx, Bz = eqcoil.Bx(self.x, self.z), eqcoil.Bz(self.x, self.z)
        if self.rc != 0:  # true divide errors for zero current coils
            a = MU_0 * self.current**2 / (4 * np.pi * self.x)
            fx = a * (np.log(8 * self.x / self.rc) - 1 + 0.25)

        else:
            fx = 0
        return np.array(
            [
                (self.current * Bz + fx) * 2 * np.pi * self.x,
                -self.current * Bx * 2 * np.pi * self.x,
            ]
        )

    def control_F(self, coil):  # noqa :N802
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
        Coil current density [A/m^2]

    Returns
    -------
    max_current: float
        Maximum current [A]
    """
    return abs(j_max * (4 * dx * dz))


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
        raise NotImplementedError("TODO vectorise")
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

        coil._flag_sizefix = self.flag_sizefix

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
        coil.x_boundary, coil.z_boundary = self._make_boundary(
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
        self.flag_sizefix = coil._flag_sizefix

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
            Maximum current for the coil. If the current density
            is not specified, this will be set to np.inf.

        Raises
        ------
        EquilibriaError:
            If the coil size is not fixed.
        """
        self.update(coil)
        if not all(self.flag_sizefix):
            raise EquilibriaError(
                "Cannot get the maximum current of a coil of an unspecified size."
            )

        if self.j_max is None:
            max_current = np.inf
        else:
            max_current = get_max_current(self.dx, self.dz, self.j_max)

        return max_current

    def _make_size(self, current=None):
        """
        Size the coil based on a current and a current density.
        """
        if current is None:
            current = self.current

        half_width = 0.5 * np.sqrt(abs(current) / self.j_max)
        return half_width, half_width

    @staticmethod
    def _make_boundary(x_c: float, z_c: float, dx: float, dz: float):
        """
        Makes the coil boundary vectors

        Parameters
        ----------
        x_c: float
            x coordinate of centre
        z_c: float
            z coordinate of centre
        dx: float
            dx of coil
        dz: float
            dz of coil

        Returns
        -------
        x_boundary, z_boundary: np.array

        Note
        ----
        Only rectangular coils

        """
        xx, zz = np.ones(4) * x_c, np.ones(4) * z_c
        x_boundary = xx + dx * np.array([-1, 1, 1, -1])
        z_boundary = zz + dz * np.array([-1, -1, 1, 1])
        return x_boundary, z_boundary


class CoilGroup(CoilFieldsMixin, abc.ABC):

    __ITERABLE_FLOAT = Union[float, Iterable[float]]
    __ITERABLE_COILTYPE = Union[str, CoilType, Iterable[Union[str, CoilType]]]

    __slots__ = (
        "__sizer",
        "_ctype",
        "_current",
        "_dx",
        "_dz",
        "_flag_sizefix",
        "_x",
        "_x_boundary",
        "_z",
        "_z_boundary",
        "b_max",
        "j_max",
        "name",
        "r_c",
    )

    def __init__(
        self,
        x: __ITERABLE_FLOAT,
        z: __ITERABLE_FLOAT,
        dx: Optional[__ITERABLE_FLOAT] = None,
        dz: Optional[__ITERABLE_FLOAT] = None,
        current: Optional[__ITERABLE_FLOAT] = 0,
        name: Optional[Union[str, Iterable[str]]] = None,
        ctype: Optional[__ITERABLE_COILTYPE] = CoilType.PF,
        j_max: Optional[__ITERABLE_FLOAT] = None,
        b_max: Optional[__ITERABLE_FLOAT] = None,
    ) -> None:

        x, z, dx, dz, current, name, ctype, j_max, b_max = self._make_iterable(
            x, z, dx, dz, current, name, ctype, j_max, b_max
        )

        self._x = x
        self._z = z
        self._dx = dx
        self._dz = dz
        self._current = current
        self.j_max = j_max
        self.b_max = b_max

        no_name_ind = np.where(name is None)[0]
        # name[no_name_ind] = CoilNamer.generate_name(self, None)
        self.name = name

        # Immutable after init
        self._ctype = tuple(
            [CoilType[ctype] if isinstance(ct, str) else ct for ct in ctype]
        )

        self._flag_sizefix = False
        self.__sizer = CoilSizer(self)
        self.__sizer(self)

    def _make_iterable(*args: Iterable[Any]) -> Iterable[Iterable[Any]]:
        """
        Converts all arguments to Iterables

        Parameters
        ----------
        *args: Any

        Returns
        -------
        Iterable

        """
        return (
            arg
            if isinstance(arg, Iterable)
            else [arg]
            if isinstance(arg, (str, type(None), CoilType))
            else np.array(arg)
            for arg in args
        )

    @property
    def x_boundary(self) -> np.array:
        """
        TODO
        """
        return self._x_boundary

    @property
    def z_boundary(self) -> np.array:
        """
        TODO
        """
        return self._z_boundary

    @property
    def area(self) -> np.array:
        """
        The cross-sectional area of the coil

        Returns
        -------
        area: float
            The cross-sectional area of the coil [m^2]
        """
        return 4 * self.dx * self.dz

    @property
    def volume(self) -> np.array:
        """
        The volume of the coil

        Returns
        -------
        volume: float
            The volume of the coil [m^3]
        """
        return self.area * 2 * np.pi * self.x

    @property
    def n_coils(self) -> int:
        return len(self.x)

    @property
    def ctype(self) -> Union[CoilType, Iterable[CoilType]]:
        """
        TODO
        """
        return self._ctype

    @property
    def current(self) -> np.array:
        """
        TODO
        """
        return self._current

    @current.setter
    def current(self, new_current: __ITERABLE_FLOAT) -> None:
        """
        TODO
        """
        pass

    def adjust_current(self, d_current: __ITERABLE_FLOAT) -> None:
        """
        TODO
        """
        pass

    @property
    def position(self) -> np.array:
        """
        TODO
        """
        return np.stack([self.x, self.z], axis=1)

    @position.setter
    def position(self, new_position: __ITERABLE_FLOAT):
        """
        TODO
        """
        pass

    def adjust_position(
        self, d_x: __ITERABLE_FLOAT, d_z: Optional[__ITERABLE_FLOAT] = None
    ):
        """
        TODO
        """
        pass

    @property
    def x(self) -> np.array:
        """
        TODO
        """
        return self._x

    @x.setter
    def x(self, new_x: __ITERABLE_FLOAT) -> None:
        """
        TODO
        """
        self._x = new_x
        self.__sizer(self)

    @property
    def z(self) -> np.array:
        """
        TODO
        """
        return self._z

    @z.setter
    def z(self, new_z: __ITERABLE_FLOAT) -> None:
        """
        TODO
        """
        self._z = new_z
        self.__sizer(self)

    @property
    def dx(self) -> np.array:
        """
        TODO
        """
        return self._dx

    @dx.setter
    def dx(self, new_dx: __ITERABLE_FLOAT) -> None:
        """
        TODO
        """
        self._dx = new_dx
        self.__sizer(self)

    @property
    def dz(self) -> np.array:
        """
        TODO
        """
        return self._dz

    @dz.setter
    def dz(self, new_dz: __ITERABLE_FLOAT) -> None:
        """
        TODO
        """
        self._dz = new_dz
        self.__sizer(self)

    def make_size(self, current: Optional[__ITERABLE_FLOAT] = None) -> None:
        """
        Size the coil based on a current and a current density.
        """
        self.__sizer(self, current)

    def fix_size(self) -> None:
        """
        TODO
        """
        self._flag_sizefix = True

    def assign_material(
        self,
        j_max: Optional[__ITERABLE_FLOAT] = NBTI_J_MAX,
        b_max: Optional[__ITERABLE_FLOAT] = NBTI_B_MAX,
    ) -> None:
        """
        Assigns EM material properties to coil

        Parameters
        ----------
        j_max: float (default None)
            Overwrite default constant material max current density [A/m^2]
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

    def get_max_current(self) -> np.array:
        """
        Gets the maximum current for a coil with a specified size

        Returns
        -------
        Imax: float
            The maximum current that can be produced by the coil [A]
        """
        return self.__sizer.get_max_current(self)

    def to_dict(self):
        """
        TODO
        """
        pass

    def to_group_vecs(self):
        """
        TODO
        """
        pass

    def plot(self):
        """
        TODO
        """
        pass

    def __str__(self) -> str:
        """
        Pretty coil printing.
        """
        ret_str = ""
        for ind in range(len(self.x)):
            ret_str += (
                f"{self.name[ind]} X={self.x[ind]:.2f} m, Z={self.z[ind]:.2f} m, I={self.current[ind]/1e6:.2f} MA "
                f"control={self.control}\n"
            )
        return ret_str[:-1]

    def __repr__(self) -> str:
        """
        Pretty console coil rendering.
        """
        return f"{self.__class__.__name__}({self.__str__()})"


class Coil(CoilGroup):

    __slots__ = ()

    def __init__(
        self,
        x: float,
        z: float,
        dx: Optional[float] = None,
        dz: Optional[float] = None,
        current: Optional[float] = 0,
        name: Optional[str] = None,
        ctype: Optional[Union[str, CoilType]] = CoilType.PF,
        j_max: Optional[float] = None,
        b_max: Optional[float] = None,
    ) -> None:
        # Only to force type check correctness
        super().__init__(x, z, dx, dz, current, name, ctype, j_max, b_max)

    def __getattribute__(self, attr: str) -> Any:
        """
        Get attribute first element if Iterable and len == 1
        """
        val = super().__getattribute__(attr)

        if isinstance(val, Iterable) and len(val) == 1:
            val = val[0]

        return val

    def __setattr__(self, attr: str, value: Any) -> None:

        with suppress(AttributeError):
            old_attr = super().__getattribute__(attr)
            if not isinstance(value, Iterable) and len(old_attr) == 1:
                if isinstance(value, str):
                    value = [value]
                else:
                    value = np.array(value)

        if isinstance(value, Iterable) and len(value) == 1:
            super().__setattr__(attr, value)
        else:
            raise ValueError(f"Length of value should be 1: {attr}={value}")

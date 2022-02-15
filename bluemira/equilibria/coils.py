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
from functools import update_wrapper, wraps
from re import split
from typing import Any, Dict, Iterable, List, Optional, Union

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

__all__ = ("CoilType", "Coil", "CoilSet", "Circuit", "SymmetricCircuit")


class CoilType(Enum):
    PF = auto()
    CS = auto()
    NONE = auto()


class CoilNumber:
    """
    Coil naming-numbering utility class. Coil naming convention is not enforced here.
    """

    __PF_counter: int = 1
    __CS_counter: int = 1
    __no_counter: int = 1

    @staticmethod
    def generate(ctype: CoilType) -> int:
        """
        Generate a coil name based on its type and indexing if specified. If no index is
        specified, an encapsulated global counter assigns an index.

        Parameters
        ----------
        coil: Any
            Object to name

        Returns
        -------
        name: str
            Coil name
        """
        if ctype == CoilType.NONE:
            idx = CoilNumber.__no_counter
            CoilNumber.__no_counter += 1
        elif ctype == CoilType.CS:
            idx = CoilNumber.__CS_counter
            CoilNumber.__CS_counter += 1
        elif ctype == CoilType.PF:
            idx = CoilNumber.__PF_counter
            CoilNumber.__PF_counter += 1

        return idx


def _sum_all(func: Optional[callable] = None, *, axis: int = 0) -> callable:
    """
    Sum all outputs of a function on a given axis
    """

    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        return ret.sum(axis=axis)

    if func is None:

        def decorator(func):
            return update_wrapper(wrapper, func)

        return decorator
    else:
        return update_wrapper(wrapper, func)


class CoilFieldsMixin:

    __slots__ = (
        "_quad_dx",
        "_quad_dz",
        "_quad_x",
        "_quad_z",
        "_quad_weighting",
        "_no_quads",
    )

    def __init__(self, weighting=None):

        # setup initial meshing
        # quadratures
        self.mesh_coil(weighting)

    def _set_quad_weighting(self, weighting=None):
        self._quad_weighting = (
            np.ones(self.x.shape[0]) if weighting is None else weighting
        )

    def mesh_coil(self, weighting=None):
        # each quadrature array = (quadrature, (x,z))

        if weighting is None:
            self._quad_x = self.x.copy()
            self._quad_dx = self.dx.copy()
            self._quad_z = self.z.copy()
            self._quad_dz = self.x.copy()

            self._no_quads = np.arange(self.x.shape[0])

        else:
            raise NotImplementedError("TODO meshing")

        self._set_quad_weighting(weighting)

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
        gpsi = greens_psi(self._quad_x, self._quad_z, x, z, self._quad_dx, self._quad_dz)

        # number of quadratures 5 for the first coil 6 for the second
        # self._noquads = [0, 5, 11]
        return np.add.reduceat(gpsi * self._quad_weighting, self._no_quads)

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
        return np.hypot(self.Bx(x, z), self.Bz(x, z))

    def _mix_control_method(self, x, z, greens_func, semianalytic_func):
        """
        Boiler-plate helper function to mixed the Green's function responses
        with the semi-analytic function responses, as a function of position
        outside/inside the coil boundary.
        """
        x, z = np.ascontiguousarray(x), np.ascontiguousarray(z)

        lg_or = np.logical_or(self._quad_dx == 0, self._quad_dz == 0)

        if False in lg_or:
            # if dx or dz is not 0 and x,z inside coil
            # TODO improve to remove inside coil calc if already known
            inside = np.logical_and(self._points_inside_coil(x, z), not lg_or)

            response = np.zeros(x.shape[1])

            if np.any(~inside):
                response[~inside] = greens_func(x[~inside], z[~inside])
            if np.any(inside):
                response[inside] = semianalytic_func(x[inside], z[inside])
        else:
            response = greens_func(x, z)

        return response

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
        inside: np.array(dtype=bool)
            The Boolean array of point indices inside/outside the coil boundary
        """
        x, z = np.ascontiguousarray(x), np.ascontiguousarray(z)
        # Add an offset, to ensure points very near the edge are counted as
        # being on the edge of a coil
        atol = X_TOLERANCE
        x_min, x_max = (
            self._quad_x - self._quad_dx - atol,
            self._quad_x + self._quad_dx + atol,
        )
        z_min, z_max = (
            self._quad_z - self._quad_dz - atol,
            self._quad_z + self._quad_dz + atol,
        )
        return (x >= x_min) & (x <= x_max) & (z >= z_min) & (z <= z_max)

    def _control_Bx_greens(self, x, z):
        """
        Calculate radial magnetic field Bx respose at (x, z) due to a unit
        current using Green's functions.
        """
        return np.add.reduceat(
            greens_Bx(self._quad_x, self._quad_z, x, z) * self._quad_weighting,
            self._no_quads,
        )

        # return sum(gx) / self.n_filaments

    def _control_Bz_greens(self, x, z):
        """
        Calculate vertical magnetic field Bz at (x, z) due to a unit current
        """
        return np.add.reduceat(
            greens_Bz(self._quad_x, self._quad_z, x, z) * self._quad_weighting,
            self._no_quads,
        )

        # return sum(gz) / self.n_filaments

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
        self.update(coil)

        dx_specified = np.array([is_num(dx) for dx in self.dx], dtype=bool)
        dz_specified = np.array([is_num(dz) for dz in self.dz], dtype=bool)
        dxdz_specified = dx_specified and dz_specified

        if not any(np.logical_and(dxdz_specified, dx_specified != dz_specified)):
            # Check that we don't have dx = None and dz = float or vice versa
            raise EquilibriaError("Must specify either dx and dz or neither.")

        if any(dxdz_specified):
            if not self.flag_sizefix:
                # If dx and dz are specified, we presume the coil size should remain fixed
                self.flag_sizefix = True

            self._set_coil_attributes(coil)

        else:
            if any(self.j_max == np.nan):
                # Check there is a viable way to size the coil
                raise EquilibriaError("Must specify either dx and dz or j_max.")

            if self.flag_sizefix:
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
        coil._x_boundary, coil._z_boundary = self._make_boundary(
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
        if not self.flag_sizefix:
            raise EquilibriaError(
                "Cannot get the maximum current of a coil of an unspecified size."
            )

        return np.where(
            self.j_max == np.nan, np.inf, get_max_current(self.dx, self.dz, self.j_max)
        )

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
        xx, zz = np.ones((1, 4)) * x_c, np.ones((1, 4)) * z_c
        x_boundary = xx + dx * np.atleast_2d([-1, 1, 1, -1])
        z_boundary = zz + dz * np.atleast_2d([-1, -1, 1, 1])
        return x_boundary, z_boundary


class CoilGroup(CoilFieldsMixin, abc.ABC):
    """
    Abstract base class for all groups of coils

    A group of coils is defined as shaing a property eg current

    Parameters
    ----------
    x: Union[float, Iterable[float]]
        Coil geometric centre x coordinate [m]
    z: Union[float, Iterable[float]]
        Coil geometric centre z coordinate [m]
    dx: Optional[Union[float, Iterable[float]]]
        Coil radial half-width [m] from coil centre to edge (either side)
    dz: Optional[Union[float, Iterable[float]]]
        Coil vertical half-width [m] from coil centre to edge (either side)
    current: Optional[Union[float, Iterable[float]] (default = 0)
        Coil current [A]
    name: Optional[Union[str, Iterable[str]]]]
        The name of the coil
    ctype: Optional[Union[str, CoilType, Iterable[Union[str, CoilType]]]
        Type of coil see CoilType enum
    j_max: Optional[Union[float, Iterable[float]]]
        Maximum current density in the coil [MA/m^2]
    b_max: Optional[Union[float, Iterable[float]]]
        Maximum magnetic field at the coil [T]

    Notes
    -----
    This class is not designed to be used directly as there are few
    protections on input variables

    """

    __ITERABLE_FLOAT = Union[float, Iterable[float]]
    __ITERABLE_COILTYPE = Union[str, CoilType, Iterable[Union[str, CoilType]]]
    __ANY_ITERABLE = Union[__ITERABLE_COILTYPE, __ITERABLE_FLOAT]

    __slots__ = (
        "__sizer",
        "_b_max",
        "_ctype",
        "_current",
        "_dx",
        "_dz",
        "_flag_sizefix",
        "_index",
        "_j_max",
        "_name_map",
        "_rc",
        "_x",
        "_x_boundary",
        "_z",
        "_z_boundary",
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

        _inputs = {
            "x": x,
            "z": z,
            "dx": dx,
            "dz": dz,
            "current": current,
            "name": name,
            "ctype": ctype,
            "j_max": j_max,
            "b_max": b_max,
        }

        _inputs = self._make_iterable(_inputs)

        self._lengthcheck(_inputs)

        self._x = _inputs["x"]
        self._z = _inputs["z"]
        self._dx = _inputs["dx"]
        self._dz = _inputs["dz"]
        self._current = _inputs["current"]
        self._j_max = _inputs["j_max"]
        self._b_max = _inputs["b_max"]

        self._ctype = [
            ct if isinstance(ct, CoilType) else CoilType[ctype]
            for ct in _inputs["ctype"]
        ]
        self._index = [CoilNumber.generate(ct) for ct in self.ctype]

        self._name_map = {
            f"{self._ctype[en].name}_{ind}" if n is None else n: ind
            for en, n, ind in enumerate(zip(_inputs["name"], self._index))
        }

        self._flag_sizefix = False
        self.__sizer = CoilSizer(self)
        self.__sizer(self)

        # Meshing
        super().__init__(None)

    @staticmethod
    def _make_iterable(
        **kwargs: __ANY_ITERABLE,
    ) -> Dict[str, Iterable[Union[str, float, CoilType]]]:
        """
        Converts all arguments to Iterables

        Parameters
        ----------
        *args: Any

        Returns
        -------
        Iterable

        Notes
        -----
        Assumes init is specified correctly. No protection against singular None.
        A singular None will fail on lengthcheck.
        String and CoilType are converted to lists everything else is a np.array
        with dtype=float.
        """
        return {
            name: (
                arg
                if isinstance(arg, Iterable)
                else [arg]
                if isinstance(arg, (str, CoilType))
                else np.atleast_2d(arg, dtype=float)
            )
            for name, arg in kwargs.items()
        }

    @staticmethod
    def _lengthcheck(ignore: Optional[List] = None, **kwargs: __ANY_ITERABLE) -> None:
        """
        Check length of iterables

        Parameters
        ----------
        ignore: list
            list of variables to ignore
        **kwargs: dict
            dictionary of arguments to check length of

        Raises
        ------
        ValueError if not in ignore list and different length to 'x'

        """
        if ignore is None:
            ignore = []

        len_first = len(kwargs["x"])
        for name, value in kwargs.items():
            if name in ignore:
                continue

            if len(value) != len_first:
                raise ValueError("lengthcheck")

    @property
    def x_boundary(self) -> np.array:
        """
        Get x boundary of coil
        """
        return self._x_boundary

    @property
    def z_boundary(self) -> np.array:
        """
        Get z boundary of coil
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
        """
        Get number of coils in group
        """
        return len(self.x)

    @property
    def ctype(self) -> Union[CoilType, Iterable[CoilType]]:
        """
        Get CoilType of coils
        """
        return self._ctype

    @property
    def name(self):
        """
        Get names of coils
        """
        return self._name_map.keys()

    @property
    def rc(self):
        """
        TODO
        """
        return self._rc

    @property
    def j_max(self):
        """
        Get coil current density
        """
        return self._j_max

    @j_max.setter
    def j_max(self, new_j_max):
        """
        Set new coil current density
        """
        self._j_max[:] = new_j_max

    @property
    def b_max(self):
        """
        Get maximum magnetic field at each coil [T]
        """
        return self._b_max

    @b_max.setter
    def b_max(self, new_b_max):
        """
        Set maximum magnetic field at each coil [T]
        """
        self._b_max[:] = new_b_max

    @property
    def current(self) -> np.array:
        """
        Get coil current
        """
        return self._current

    @current.setter
    def current(self, new_current: __ITERABLE_FLOAT) -> None:
        """
        Set coil current
        """
        self._current[:] = new_current

    def adjust_current(self, d_current: __ITERABLE_FLOAT) -> None:
        """
        Modify current in each coil
        """
        self.current += d_current

    @property
    def position(self) -> np.array:
        """
        Set position of each coil
        """
        return np.stack([self.x, self.z], axis=1)

    @position.setter
    def position(self, new_position: __ITERABLE_FLOAT):
        """
        Set position of each coil
        """
        self._x[:] = new_position[:, 0]
        self._z[:] = new_position[:, 1]
        self.__sizer(self)

    def adjust_position(self, d_xz: __ITERABLE_FLOAT):
        """
        Adjust position of each coil
        """
        self.position = np.stack([self.x + d_xz[:, 0], self.z + d_xz[:, 1]], axis=1)
        self.__sizer(self)

    @property
    def x(self) -> np.array:
        """
        Get x coordinate of each coil
        """
        return self._x

    @x.setter
    def x(self, new_x: __ITERABLE_FLOAT) -> None:
        """
        Set x coordinate of each coil
        """
        self._x[:] = new_x
        self.__sizer(self)

    @property
    def z(self) -> np.array:
        """
        Get z coordinate of each coil
        """
        return self._z

    @z.setter
    def z(self, new_z: __ITERABLE_FLOAT) -> None:
        """
        Set z coordinate of each coil
        """
        self._z[:] = new_z
        self.__sizer(self)

    @property
    def dx(self) -> np.array:
        """
        Get dx coordinate of each coil
        """
        return self._dx

    @dx.setter
    def dx(self, new_dx: __ITERABLE_FLOAT) -> None:
        """
        Set dx coordinate of each coil
        """
        self._dx[:] = new_dx
        self.__sizer(self)

    @property
    def dz(self) -> np.array:
        """
        Get dz coordinate of each coil
        """
        return self._dz

    @dz.setter
    def dz(self, new_dz: __ITERABLE_FLOAT) -> None:
        """
        Set dz coordinate of each coil
        """
        self._dz[:] = new_dz
        self.__sizer(self)

    def make_size(self, current: Optional[__ITERABLE_FLOAT] = None) -> None:
        """
        Size the coil based on a current and a current density.
        """
        self.__sizer(self, current)

    def fix_size(self) -> None:
        """
        Fixes the size of all coils
        """
        self._flag_sizefix = True
        self.__sizer.update(self)

    def assign_material(
        self,
        j_max: __ITERABLE_FLOAT = NBTI_J_MAX,
        b_max: __ITERABLE_FLOAT = NBTI_B_MAX,
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
        for jm, bm in zip(j_max, b_max):
            if not is_num(j_max):
                raise EquilibriaError(f"j_max must be specified as a number, not: {jm}")
            if not is_num(b_max):
                raise EquilibriaError(f"b_max must be specified as a number, not: {bm}")

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

    def to_group_vecs(self) -> Iterable[np.array]:
        """
        Collect CoilGroup Properties

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
            self.x,
            self.z,
            self.dx,
            self.dz,
            self.current,
        )

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

    # def _wrap_properties(self):
    # pass
    # If attribute property wrap function to produce numbered output

    # def __getattribute__(self, attr: str) -> Any:
    #     """
    #     Get attribute first element if Iterable and len == 1
    #     """
    #     val = super().__getattribute__(attr)

    #     if isinstance(val, Iterable) and len(val) == 1:
    #         val = val[0]

    #     return val

    def __setattr__(self, attr: str, value: Any) -> None:

        with suppress(AttributeError):
            old_attr = super().__getattribute__(attr)
            if not isinstance(value, Iterable) and len(old_attr) == 1:
                if isinstance(value, (str, CoilType)):
                    value = [value]
                else:
                    value = np.atleast_2d(value, dtype=float)

        if isinstance(value, Iterable) and len(value) == 1:
            super().__setattr__(attr, value)
        else:
            raise ValueError(f"Length of value should be 1: {attr}={value}")

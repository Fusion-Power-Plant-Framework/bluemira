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
from __future__ import annotations

import abc

# from copy import deepcopy
from enum import Enum, EnumMeta, auto
from functools import update_wrapper

# from re import split
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

# import matplotlib.pyplot as plt
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
from bluemira.utilities.tools import is_num_array

# from scipy.interpolate import RectBivariateSpline


__all__ = ["CoilType", "Coil", "CoilSet", "Circuit", "SymmetricCircuit"]


class CoilTypeEnumMeta(EnumMeta):
    """
    Allow override of KeyError error string
    """

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError(f"Unknown CoilType {name}") from None


class CoilType(Enum, metaclass=CoilTypeEnumMeta):
    """
    CoilType Enum
    """

    PF = auto()
    CS = auto()
    NONE = auto()


__ITERABLE_FLOAT = Union[float, Iterable[float]]
__ITERABLE_COILTYPE = Union[str, CoilType, Iterable[Union[str, CoilType]]]
__ANY_ITERABLE = Union[__ITERABLE_COILTYPE, __ITERABLE_FLOAT]


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
        "_einsum_str",
    )

    def __init__(self, weighting=None):
        if type(self) == CoilFieldsMixin:
            raise TypeError("Can't be initialised directly")

        self.discretise(weighting)

    def _pad_discretisation(
        self,
        _quad_x: List[np.ndarray],
        _quad_z: List[np.ndarray],
        _quad_dx: List[np.ndarray],
        _quad_dz: List[np.ndarray],
    ):
        """
        Convert quadrature list of array to rectuangualr arrays.
        Padding quadrature arrays with zeros to allow array operations
        on rectangular matricies.

        Parameters
        ----------
        _quad_x: List[np.ndarray]
            x quadratures
        _quad_z: List[np.ndarray]
            z quadratures
        _quad_dx: List[np.ndarray]
            dx quadratures
        _quad_dz: List[np.ndarray]
            dz quadratures

        Notes
        -----
        In reality this is just a nicety for storing as padding only
        exists for multiple coils in a :class:CoilGroup that are different shapes.
        There is no extra calculation on the elements set to zero because
        of the mechanics of the :func:_combined_control method.

        """
        all_len = np.array([len(q) for q in _quad_x])
        max_len = max(all_len)
        diff = max_len - all_len

        for i, d in enumerate(diff):
            for val in [_quad_x, _quad_z, _quad_dx, _quad_dz]:
                val[i] = np.pad(val[i], (0, d))

        self._quad_x = np.array(_quad_x)
        self._quad_z = np.array(_quad_z)
        self._quad_dx = np.array(_quad_dx)
        self._quad_dz = np.array(_quad_dz)
        weighting = np.ones((self._x.shape[0], max_len)) / all_len[:, None]
        weighting[self._quad_dx == 0] = 0

        return weighting

    def _rectangular_discretisation(self, d_coil):
        """
        Discretise a coil into smaller rectangles based on fraction of
        coil dimensions

        Parameters
        ----------
        d_coil: float
            Target discretisation fraction

        Returns
        -------
        weighting: np.ndarray

        """
        nx = np.maximum(1, np.ceil(self._dx * 2 / d_coil))
        nz = np.maximum(1, np.ceil(self._dz * 2 / d_coil))

        if not all(nx * nz == 1):
            dx, dz = self._dx / nx, self._dz / nz
            _quad_x = []
            _quad_z = []
            _quad_dx = []
            _quad_dz = []

            for i, (coil_x, coil_z, sc_dx, sc_dz, _nx, _nz) in enumerate(
                zip(self._x, self._z, dx, dz, nx, nz)
            ):

                # Calculate sub-coil centroids
                x_sc = (coil_x - self._dx[i]) + sc_dx * np.arange(1, 2 * _nx, 2)
                z_sc = (coil_z - self._dz[i]) + sc_dz * np.arange(1, 2 * _nz, 2)
                x_sc, z_sc = np.meshgrid(x_sc, z_sc)

                _quad_x += [x_sc.flat]
                _quad_z += [z_sc.flat]
                _quad_dx += [np.ones(x_sc.size) * sc_dx]
                _quad_dz += [np.ones(x_sc.size) * sc_dz]

            return self._pad_discretisation(_quad_x, _quad_z, _quad_dx, _quad_dz)

    def discretise(self, d_coil=None):
        """
        Discretise a coil for greens function magnetic field calculations

        Parameters
        ----------
        d_coil: float
            Target discretisation fraction

        Notes
        -----
        Only discretisation method currently implemented is rectangular fraction

        """
        weighting = None
        self._quad_x = self._x.copy()
        self._quad_dx = self._dx.copy()
        self._quad_z = self._z.copy()
        self._quad_dz = self._dz.copy()

        if d_coil is not None:
            # How fancy do we want the mesh or just smaller rectangles?
            weighting = self._rectangular_discretisation(d_coil)

        if weighting is None:
            self._einsum_str = "..., ...j -> ..."
            self._quad_weighting = np.ones((self._x.shape[0], 1))
        else:
            self._einsum_str = "...j, ...j -> ..."
            self._quad_weighting = weighting

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
        x, z = np.ascontiguousarray(x), np.ascontiguousarray(z)

        return np.einsum(
            self._einsum_str,
            greens_psi(
                self._quad_x[None],
                self._quad_z[None],
                x[..., None],
                z[..., None],
                self._quad_dx[None],
                self._quad_dz[None],
            ),
            self._quad_weighting[None],
        )

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

        Parameters
        ----------
        x,z: Union[float, int, np.array]
            Points to calculate field at
        greens_func: Callable
            greens function
        semianalytic_func: Callable
            semianalytic function

        Returns
        -------
        response: np.ndarray

        """
        x, z = np.ascontiguousarray(x), np.ascontiguousarray(z)

        zero_coil_size = np.logical_or(self._dx == 0, self._dz == 0)

        if False in zero_coil_size:
            # if dx or dz is not 0 and x,z inside coil
            inside = np.logical_and(
                self._points_inside_coil(x, z), ~zero_coil_size[None]
            )
            if np.all(~inside):
                return greens_func(x, z)
            elif np.all(inside):
                # Not called for circuits as they will always be a mixture
                return semianalytic_func(x, z)
            else:
                return self._combined_control(
                    inside, x, z, greens_func, semianalytic_func
                )
        else:
            return greens_func(x, z)

    def _combined_control(self, inside, x, z, greens_func, semianalytic_func):
        """
        Combine semianalytic and greens function calculation of magnetic field

        Used for situation where there are calculation points both inside and
        outside the coil boundaries.

        Parameters
        ----------
        inside: np.ndarray[bool]
            array of if the point is inside a coil
        x,z: Union[float, int, np.array]
            Points to calculate field at
        greens_func: Callable
            greens function
        semianalytic_func: Callable
            semianalytic function

        Returns
        -------
        response: np.ndarray

        """
        response = np.zeros_like(inside, dtype=float)
        for coil, (points, qx, qz, qw, cx, cz, cdx, cdz) in enumerate(
            zip(
                inside.T,
                self._quad_x,
                self._quad_z,
                self._quad_weighting,
                self._x,
                self._z,
                self._dx,
                self._dz,
            )
        ):
            if np.any(~points):
                response[~points, coil] = np.squeeze(
                    greens_func(x[~points], z[~points], True, qx, qz, qw)
                )
            if np.any(points):
                response[points, coil] = np.squeeze(
                    semianalytic_func(x[points], z[points], True, cx, cz, cdx, cdz)
                )

        return response

    def _points_inside_coil(
        self,
        x: Union[float, np.array],
        z: Union[float, np.array],
        *,
        atol: float = X_TOLERANCE,
    ):
        """
        Determine which points lie inside or on the coil boundary.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinates to check
        z: Union[float, np.array]
            The z coordinates to check
        atol: Optional[float]
            Add an offset, to ensure points very near the edge are counted as
            being on the edge of a coil

        Returns
        -------
        inside: np.array(dtype=bool)
            The Boolean array of point indices inside/outside the coil boundary
        """
        x, z = np.ascontiguousarray(x)[..., None], np.ascontiguousarray(z)[..., None]

        x_min, x_max = (
            self._x - self._dx - atol,
            self._x + self._dx + atol,
        )
        z_min, z_max = (
            self._z - self._dz - atol,
            self._z + self._dz + atol,
        )
        return (
            (x >= x_min[None])
            & (x <= x_max[None])
            & (z >= z_min[None])
            & (z <= z_max[None])
        )

    def _control_B_greens(
        self, greens, x, z, split=False, _quad_x=None, _quad_z=None, _quad_weight=None
    ):
        """
        Calculate radial magnetic field B respose at (x, z) due to a unit
        current using Green's functions.

        Parameters
        ----------
        greens: Callable
            greens function
        x,z: Union[float, int, np.array]
            Points to calculate field at
        split: bool
            Flag for if :func:_combined_control is used
        _quad_x: Optional[np.ndarray]
            :func:_combined_control x positions
        _quad_z: Optional[np.ndarray]
            :func:_combined_control z positions
        _quad_weight: Optional[np.ndarray]
            :func:_combined_control weighting

        Returns
        -------
        response: np.ndarray

        """
        if not split:
            _quad_x = self._quad_x
            _quad_z = self._quad_z
            _quad_weight = self._quad_weighting

        return np.einsum(
            self._einsum_str,
            greens(
                _quad_x[None],
                _quad_z[None],
                x[..., None],
                z[..., None],
            ),
            _quad_weight[None],
        )

    def _control_Bx_greens(
        self, x, z, split=False, _quad_x=None, _quad_z=None, _quad_weight=None
    ):
        """
        Calculate radial magnetic field Bx respose at (x, z) due to a unit
        current using Green's functions.

        Parameters
        ----------
        greens: Callable
            greens function
        x,z: Union[float, int, np.array]
            Points to calculate field at
        split: bool
            Flag for if :func:_combined_control is used
        _quad_x: Optional[np.ndarray]
            :func:_combined_control x positions
        _quad_z: Optional[np.ndarray]
            :func:_combined_control z positions
        _quad_weight: Optional[np.ndarray]
            :func:_combined_control weighting

        Returns
        -------
        response: np.ndarray

        """
        return self._control_B_greens(
            greens_Bx, x, z, split, _quad_x, _quad_z, _quad_weight
        )

    def _control_Bz_greens(
        self, x, z, split=False, _quad_x=None, _quad_z=None, _quad_weight=None
    ):
        """
        Calculate vertical magnetic field Bz at (x, z) due to a unit current

        Parameters
        ----------
        greens: Callable
            greens function
        x,z: Union[float, int, np.array]
            Points to calculate field at
        split: bool
            Flag for if :func:_combined_control is used
        _quad_x: Optional[np.ndarray]
            :func:_combined_control x positions
        _quad_z: Optional[np.ndarray]
            :func:_combined_control z positions
        _quad_weight: Optional[np.ndarray]
            :func:_combined_control weighting

        Returns
        -------
        response: np.ndarray

        """
        return self._control_B_greens(
            greens_Bz, x, z, split, _quad_x, _quad_z, _quad_weight
        )

    def _control_B_analytical(
        self,
        semianalytic,
        x,
        z,
        split=False,
        coil_x=None,
        coil_z=None,
        coil_dx=None,
        coil_dz=None,
    ):
        """
        Calculate radial magnetic field Bx response at (x, z) due to a unit
        current using semi-analytic method.

        Parameters
        ----------
        semianalytic: Callable
            semianalytic function
        x,z: Union[float, int, np.array]
            Points to calculate field at
        split: bool
            Flag for if :func:_combined_control is used
        coil_x: Optional[np.ndarray]
            :func:_combined_control x positions
        coil_z: Optional[np.ndarray]
            :func:_combined_control z positions
        coil_dx: Optional[np.ndarray]
            :func:_combined_control x positions
        coil_dz: Optional[np.ndarray]
            :func:_combined_control z positions

        Returns
        -------
        response: np.ndarray

        """
        if not split:
            coil_x = self._x
            coil_z = self._z
            coil_dx = self._dx
            coil_dz = self._dz

        return semianalytic(
            coil_x[None],
            coil_z[None],
            x[..., None],
            z[..., None],
            d_xc=coil_dx[None],
            d_zc=coil_dz[None],
        )

    def _control_Bx_analytical(
        self, x, z, split=False, coil_x=None, coil_z=None, coil_dx=None, coil_dz=None
    ):
        """
        Calculate vertical magnetic field Bx response at (x, z) due to a unit
        current using semi-analytic method.

        Parameters
        ----------
        x,z: Union[float, int, np.array]
            Points to calculate field at
        split: bool
            Flag for if :func:_combined_control is used
        coil_x: Optional[np.ndarray]
            :func:_combined_control x positions
        coil_z: Optional[np.ndarray]
            :func:_combined_control z positions
        coil_dx: Optional[np.ndarray]
            :func:_combined_control x positions
        coil_dz: Optional[np.ndarray]
            :func:_combined_control z positions

        Returns
        -------
        response: np.ndarray
        """
        return self._control_B_analytical(
            semianalytic_Bx, x, z, split, coil_x, coil_z, coil_dx, coil_dz
        )

    def _control_Bz_analytical(
        self, x, z, split=False, coil_x=None, coil_z=None, coil_dx=None, coil_dz=None
    ):
        """
        Calculate vertical magnetic field Bz response at (x, z) due to a unit
        current using semi-analytic method.

        Parameters
        ----------
        x,z: Union[float, int, np.array]
            Points to calculate field at
        split: bool
            Flag for if :func:_combined_control is used
        coil_x: Optional[np.ndarray]
            :func:_combined_control x positions
        coil_z: Optional[np.ndarray]
            :func:_combined_control z positions
        coil_dx: Optional[np.ndarray]
            :func:_combined_control x positions
        coil_dz: Optional[np.ndarray]
            :func:_combined_control z positions

        Returns
        -------
        response: np.ndarray
        """
        return self._control_B_analytical(
            semianalytic_Bz, x, z, split, coil_x, coil_z, coil_dx, coil_dz
        )

    def F(self, eqcoil):  # noqa :N802
        """
        Calculate the force response at the coil centre including the coil
        self-force.

        \t:math:`\\mathbf{F} = \\mathbf{j}\\times \\mathbf{B}`\n
        \t:math:`F_x = IB_z+\\dfrac{\\mu_0I^2}{4\\pi X}\\textrm{ln}\\bigg(\\dfrac{8X}{r_c}-1+\\xi/2\\bigg)`\n
        \t:math:`F_z = -IBx`
        """  # noqa :W505
        Bx, Bz = eqcoil.Bx(self._x, self._z), eqcoil.Bz(self._x, self._z)
        if self.rc != 0:  # true divide errors for zero current coils
            a = MU_0 * self.current**2 / (4 * np.pi * self._x)
            fx = a * (np.log(8 * self._x / self.rc) - 1 + 0.25)

        else:
            fx = 0
        return np.array(
            [
                (self.current * Bz + fx) * 2 * np.pi * self._x,
                -self.current * Bx * 2 * np.pi * self._x,
            ]
        )

    def control_F(self, coil: CoilGroup = None):  # noqa :N802
        """
        Returns the Green's matrix element for the coil mutual force.

        \t:math:`Fz_{i,j}=-2\\pi X_i\\mathcal{G}(X_j,Z_j,X_i,Z_i)`
        """

        if coil._x == self._x and coil._z == self._z:
            # self inductance
            if self.rc != 0:
                a = MU_0 / (4 * np.pi * self._x)
                Bz = a * (np.log(8 * self._x / self.rc) - 1 + 0.25)
            else:
                Bz = 0
            Bx = 0  # Should be 0 anyway

        else:
            Bz = coil.control_Bz(self._x, self._z)
            Bx = coil.control_Bx(self._x, self._z)

        return 2 * np.pi * self._x * np.array([Bz, -Bx])  # 1 cross B


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

        dx_specified = np.array([is_num_array(self._dx)], dtype=bool)
        dz_specified = np.array([is_num_array(self._dz)], dtype=bool)
        dxdz_specified = np.logical_and(dx_specified, dz_specified)

        if any(
            np.logical_and(
                ~dxdz_specified, np.logical_xor(dx_specified, dz_specified)
            ).flatten()
        ):
            # Check that we don't have dx = None and dz = float or vice versa
            raise EquilibriaError("Must specify either dx and dz or neither.")

        if any(dxdz_specified.flatten()):
            if not self.flag_sizefix:
                # If dx and dz are specified, we presume the coil size should
                # remain fixed
                self.flag_sizefix = True

            self._set_coil_attributes(coil)

        else:
            if any(~is_num_array(self.j_max)):
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
            coil._dx, coil._dz = self._make_size(current)
            self._set_coil_attributes(coil)

    def _set_coil_attributes(self, coil):
        coil._rc = 0.5 * np.hypot(coil._dx, coil._dz)
        coil._x_boundary, coil._z_boundary = self._make_boundary(
            coil._x, coil._z, coil._dx, coil._dz
        )

    def update(self, coil):
        """
        Update the CoilSizer

        Parameters
        ----------
        coil: Coil
            Coil to size
        """
        self._dx = coil._dx
        self._dz = coil._dz
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
            self.j_max == np.nan, np.inf, get_max_current(self._dx, self._dz, self.j_max)
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
        xx, zz = (np.ones((4, 1)) * x_c).T, (np.ones((4, 1)) * z_c).T
        x_boundary = xx + (dx * np.array([-1, 1, 1, -1])[:, None]).T
        z_boundary = zz + (dz * np.array([-1, -1, 1, 1])[:, None]).T
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

    __slots__ = (
        "_sizer",
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
        d_coil: Optional[int] = None,
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

        _inputs = self._make_iterable(**_inputs)

        self._lengthcheck(**_inputs)

        self._x = _inputs["x"]
        self._z = _inputs["z"]
        self._dx = _inputs["dx"]
        self._dz = _inputs["dz"]
        self._current = _inputs["current"]
        self._j_max = _inputs["j_max"]
        self._b_max = _inputs["b_max"]

        self._ctype = [
            ct
            if isinstance(ct, CoilType)
            else CoilType["NONE"]
            if isinstance(ct, np.ndarray) or ct is None
            else CoilType[ct.upper()]
            for ct in _inputs["ctype"]
        ]
        self._index = [CoilNumber.generate(ct) for ct in self.ctype]

        self._name_map = {
            f"{self._ctype[en].name}_{ind}" if n is None else n: ind
            for en, (n, ind) in enumerate(zip(_inputs["name"], self._index))
        }

        self._flag_sizefix = False
        self._sizer = CoilSizer(self)
        self._sizer(self)

        # Meshing
        super().__init__(d_coil)

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
        ret = {}

        n_c = len(kwargs["x"]) if isinstance(kwargs["x"], Iterable) else 1
        for name, arg in kwargs.items():
            if name in ["name", "ctype"]:
                if isinstance(arg, (str, type(None), CoilType)):
                    arg = [arg for _ in range(n_c)]
            elif isinstance(arg, Iterable) and isinstance(
                arg[0], (int, float, complex, CoilType)
            ):
                arg = np.array(arg)
            else:
                arg = np.array([arg for _ in range(n_c)], dtype=float)
            ret[name] = arg

        return ret

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
                raise ValueError(
                    f"{name} does not contain {len_first} elements: {value}"
                )

    def _define_subgroup(self, *groups):
        """
        Create groups enum

        be careful will make all previous uses uncomparible
        """
        groups = ["_all"] + list(groups)
        self._SubGroup = Enum("SubGroup", {g: auto() for g in groups})

    @property
    def x_boundary(self) -> np.ndarray:
        """
        Get x boundary of coil
        """
        return self._x_boundary

    @property
    def z_boundary(self) -> np.ndarray:
        """
        Get z boundary of coil
        """
        return self._z_boundary

    @property
    def area(self) -> np.ndarray:
        """
        The cross-sectional area of the coil

        Returns
        -------
        area: float
            The cross-sectional area of the coil [m^2]
        """
        return 4 * self.dx * self.dz

    @property
    def volume(self) -> np.ndarray:
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
        return list(self._name_map.keys())

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
    def current(self) -> np.ndarray:
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
    def position(self) -> np.ndarray:
        """
        Set position of each coil
        """
        return self.x.ravel(), self.z.ravel()

    @position.setter
    def position(self, new_position: __ITERABLE_FLOAT):
        """
        Set position of each coil
        """
        self.x = new_position[:, 0]
        self.z = new_position[:, 1]

    def adjust_position(self, d_xz: __ITERABLE_FLOAT):
        """
        Adjust position of each coil
        """
        d_xz = np.atleast_2d(d_xz)
        self.position = np.stack(
            [self.x.ravel() + d_xz[:, 0], self.z.ravel() + d_xz[:, 1]], axis=1
        )

    @property
    def x(self) -> tuple:
        """
        Get x coordinate of each coil
        """
        _x = self._x[:]
        _x.flags.writeable = False
        return _x

    @x.setter
    def x(self, new_x: __ITERABLE_FLOAT) -> None:
        """
        Set x coordinate of each coil
        """
        self._x[:] = np.atleast_2d(new_x.T).T
        self._sizer(self)

    @property
    def z(self) -> tuple:
        """
        Get z coordinate of each coil
        """
        _z = self._z[:]
        _z.flags.writeable = False
        return _z

    @z.setter
    def z(self, new_z: __ITERABLE_FLOAT) -> None:
        """
        Set z coordinate of each coil
        """
        self._z[:] = np.atleast_2d(new_z.T).T
        self._sizer(self)

    @property
    def dx(self) -> tuple:
        """
        Get dx coordinate of each coil
        """
        _dx = self._dx[:]
        _dx.flags.writeable = False
        return _dx

    @dx.setter
    def dx(self, new_dx: __ITERABLE_FLOAT) -> None:
        """
        Set dx coordinate of each coil
        """
        self._dx[:] = np.atleast_2d(new_dx.T).T
        self._sizer(self)

    @property
    def dz(self) -> np.ndarray:
        """
        Get dz coordinate of each coil
        """
        _dz = self._dz[:]
        _dz.flags.writeable = False
        return _dz

    @dz.setter
    def dz(self, new_dz: __ITERABLE_FLOAT) -> None:
        """
        Set dz coordinate of each coil
        """
        self._dz[:] = np.atleast_2d(new_dz.T).T
        self._sizer(self)

    def make_size(self, current: Optional[__ITERABLE_FLOAT] = None) -> None:
        """
        Size the coil based on a current and a current density.
        """
        self._sizer(self, current)

    def fix_size(self) -> None:
        """
        Fixes the size of all coils
        """
        self._flag_sizefix = True
        self._sizer.update(self)

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
            if not is_num_array(j_max):
                raise EquilibriaError(f"j_max must be specified as a number, not: {jm}")
            if not is_num_array(b_max):
                raise EquilibriaError(f"b_max must be specified as a number, not: {bm}")

        self.j_max = j_max
        self.b_max = b_max
        self._sizer.update(self)

    def get_max_current(self) -> np.ndarray:
        """
        Gets the maximum current for a coil with a specified size

        Returns
        -------
        Imax: float
            The maximum current that can be produced by the coil [A]
        """
        return self._sizer.get_max_current(self)

    def to_dict(self):
        """
        TODO
        """
        raise NotImplementedError

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

    @classmethod
    def from_group_vecs(cls, groupvecs):
        raise NotImplementedError

    @classmethod
    def from_coils(cls, coils):
        raise NotImplementedError

    def plot(self):
        """
        TODO
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        Pretty coil printing.
        """
        x = self._x.flatten()
        z = self._z.flatten()
        c = self._current.flatten()
        return " | ".join(
            (
                f"{name}: X={x[ind]:.2f} m,"
                f" Z={z[ind]:.2f} m,"
                f" I={c[ind]/1e6:.2f} MA"
            )
            for ind, name in enumerate(self._name_map.keys())
        )

    def __repr__(self) -> str:
        """
        Pretty console coil rendering.
        """
        return f"{self.__class__.__name__}({self.__str__()})"


class Coil(CoilGroup):
    """
    Singular coil

    Parameters
    ----------
    x: float
        Coil geometric centre x coordinate [m]
    z: float
        Coil geometric centre z coordinate [m]
    dx: Optional[float]
        Coil radial half-width [m] from coil centre to edge (either side)
    dz: Optional[float]
        Coil vertical half-width [m] from coil centre to edge (either side)
    current: Optional[float] (default = 0)
        Coil current [A]
    name: Optional[str]
        The name of the coil
    ctype: Optional[Union[str, CoilType]]
        Type of coil see CoilType enum
    j_max: Optional[float]
        Maximum current density in the coil [MA/m^2]
    b_max: Optional[float]
        Maximum magnetic field at the coil [T]

    """

    __slots__ = ()

    __safe_attrs = ("_flag_sizefix", "_sizer")

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
        d_coil: Optional[int] = None,
    ) -> None:
        # Only to force type check correctness
        super().__init__(x, z, dx, dz, current, name, ctype, j_max, b_max, d_coil)

    @property
    def name(self):
        """
        Name of coil
        """
        return list(self._name_map.keys())[0]

    # def __setattr__(self, attr: str, value: Any) -> None:
    #     """
    #     Set attribute with some protection for singular values
    #     """
    #     with suppress(AttributeError):
    #         old_attr = super().__getattribute__(attr)
    #         if attr not in self.__safe_attrs:
    #             if not isinstance(value, Iterable) and len(old_attr) == 1:
    #                 if isinstance(value, (str, CoilType)):
    #                     value = [value]
    #                 else:
    #                     value = np.atleast_2d(value, dtype=float)

    #     if attr in self.__safe_attrs or (
    #         isinstance(value, Iterable) and len(value) == 1
    #     ):
    #         super().__setattr__(attr, value)
    #     else:
    #         raise ValueError(f"Length of value should be 1: {attr}={value}")


class Circuit(CoilGroup):
    """
    Base circuit class
    """

    __num_circuit = 0

    __slots__ = "_circuit_name"

    @staticmethod
    def _namer():
        Circuit.__num_circuit += 1
        return f"CIRC_{Circuit.__num_circuit}"

    @CoilGroup.current.setter
    def current(self, new_current: float) -> None:
        """
        Set coil current
        """
        if isinstance(new_current, Iterable):
            new_current = new_current[0]
        self._current[:] = new_current


class SymmetricCircuit(Circuit):
    """
    Positionally symmetric coils everything else the same


    Parameters
    ----------
    symmetry_line: np.ndarray[[float, float], [float, float]]:
        two points making a symmetry line
    x: float
        Coil geometric centre x coordinate [m]
    z: float
        Coil geometric centre z coordinate [m]
    dx: Optional[float]
        Coil radial half-width [m] from coil centre to edge (either side)
    dz: Optional[float]
        Coil vertical half-width [m] from coil centre to edge (either side)
    current: Optional[float] (default = 0)
        Coil current [A]
    name: Optional[str]
        The name of the coil
    ctype: Optional[Union[str, CoilType]]
        Type of coil see CoilType enum
    j_max: Optional[float]
        Maximum current density in the coil [MA/m^2]
    b_max: Optional[float]
        Maximum magnetic field at the coil [T]

    """

    __slots__ = ("_uv", "_symmetry_point", "_point")

    def __init__(
        self,
        symmetry_line: np.ndarray[[float, float], [float, float]],
        x: float,
        z: float,
        dx: Optional[float] = None,
        dz: Optional[float] = None,
        current: Optional[float] = 0,
        name: Optional[str] = None,
        ctype: Optional[Union[str, CoilType]] = CoilType.PF,
        j_max: Optional[float] = None,
        b_max: Optional[float] = None,
        d_coil: Optional[int] = None,
    ) -> None:

        self._circuit_name = self._namer()
        self._point = np.array([x, z])
        x, z = self._setup_symmetry(symmetry_line)
        ones = np.ones(2)
        current *= ones
        ctype = [ctype, ctype]

        if dx is not None:
            dx *= ones
        if dz is not None:
            dz *= ones
        if name is not None:
            name = [f"{name}.1", f"{name}.2"]
        if j_max is not None:
            j_max *= ones
        if b_max is not None:
            b_max *= ones

        super().__init__(x, z, dx, dz, current, name, ctype, j_max, b_max, d_coil)

    @property
    def name(self):
        """
        Name of circuit
        """
        return self._circuit_name

    def modify_symmetry(self, symmetry_line: np.ndarray[[float, float], [float, float]]):
        """
        Create a unit vector for the symmetry of the coil

        Parameters
        ----------
        symmetry_line: np.ndarray[[float, float], [float, float]]
            two points making a symmetry line

        """
        self._uv = (symmetry_line[1] - symmetry_line[0]) / np.linalg.norm(
            symmetry_line[1] - symmetry_line[0]
        )
        self._symmetry_point = symmetry_line[0]

    def _setup_symmetry(self, symmetry_line):
        """
        Setup the symmetry of the coil

        Parameters
        ----------
        symmetry_line: np.ndarray[[float, float], [float, float]]
            two points making a symmetry line

        Returns
        -------
        x, z of the two coils

        """
        self.modify_symmetry(symmetry_line)
        return np.array([self._point, self._point - self._symmetrise()]).T

    def _symmetrise(self):
        """
        Calculate the change in position to the symmetric coil,
        twice the distance to the line of symmetry.
        """
        return 2 * (
            (self._point - self._symmetry_point)
            - (np.dot(self._point - self._symmetry_point, self._uv) * self._uv)
        )

    def _resymmetrise_x(self):
        self._point[0] = self._x[0]
        self._x[1] = self._point[0] - self._symmetrise()[0]

    def _resymmetrise_z(self):
        self._point[1] = self._z[0]
        self._z[1] = self._point[1] - self._symmetrise()[1]

    @Circuit.x.setter
    def x(self, new_x: float) -> None:
        """
        Set x coordinate of each coil
        """
        self._x[0] = self._point[0] = new_x
        self._x[1] = self._point[0] - self._symmetrise()[0]
        self._sizer(self)

    @Circuit.z.setter
    def z(self, new_z: float) -> None:
        """
        Set z coordinate of each coil
        """
        self._z[0] = self._point[1] = new_z
        self._z[1] = self._point[1] - self._symmetrise()[1]
        self._sizer(self)

    @Circuit.position.setter
    def position(self, new_position: __ITERABLE_FLOAT):
        """
        Set position of each coil
        """
        self.x = new_position[0, 0]
        self.z = new_position[0, 1]


class CoilSet(CoilGroup):
    """
    Coilset is the main interface for groups of coils in bluemira

    """

    __slots__ = ("__coilgroups", "_circuits")

    def __init__(self, *coils: Union[CoilGroup, List, Dict], d_coil=None):

        if not coils:
            raise ValueError("No coils provided")

        attributes = self._process_coilgroups(self._convert_to_coilgroup(coils))

        for k, v in attributes.items():
            setattr(self, k, v)

        self.discretise(d_coil)
        # TODO deal with sizing
        # TODO think whether this is the best way forward

    def __init_subclass__(cls, *args, **kwargs):
        """
        Subclassing protection
        """
        raise EquilibriaError("class not designed to be subclassed")

    def __str__(self) -> str:
        """
        Pretty pront Coilset
        """
        return ", ".join(
            sorted(
                [f"{v.__class__.__name__}({v})" for v in self.__coilgroups.values()],
                key=lambda k: k.split("(")[1].split(":")[0],
            )
        )

    # @CoilGroup.x.setter
    # def x(self, new_x: __ITERABLE_FLOAT):
    #     self._x[:] = np.atleast_2d(new_x.T).T
    #     self._ensure_symmetry('x')

    # @CoilGroup.z.setter
    # def z(self, new_z: __ITERABLE_FLOAT):
    #     self._z[:] = np.atleast_2d(new_z.T).T
    #     self._ensure_symmetry('z')
    @staticmethod
    def _sizer(self):
        for cg in self.__coilgroups.values():
            cg._sizer(cg)

    def _ensure_symmetry(self, prop):
        for no, cg in enumerate(self.__coilgroups.values()):
            if self._circuits[no]:
                getattr(cg, f"_resymmetrise_{prop}")()
            cg._sizer(cg)

    @staticmethod
    def _convert_to_coilgroup(
        coils: Tuple[Union[CoilGroup, List, Dict]]
    ) -> List[CoilGroup]:
        # Overly complex data structure of coils not dealt with
        # eg Tuple(List(CoilGroup), List(List), Dict(List))
        for i, coil in enumerate(coils):
            if isinstance(coil, List):
                coils[i] = Coil(*coil)
            elif isinstance(coil, Dict):
                coils[i] = Coil(**coil)
            elif not isinstance(coil, CoilGroup):
                raise TypeError(f"Conversion to Coil unknown for type '{type(coil)}'")
        return coils

    def _process_coilgroups(self, coilgroups: List[CoilGroup]):
        self.__coilgroups = {cg.name: cg for cg in coilgroups}
        self._circuits = [isinstance(cg, Circuit) for cg in coilgroups]

        # filters = {
        #     group.name: partial(
        #         lambda name, coilgroup: np.array(
        #             [c_n == name for c_n in coilgroup.name], dtype=bool
        #         ),
        #         group.name,
        #     )
        #     for group in coilgroups
        # }

        # self.define_subset(filters)
        # self._finalise_groups()

        names = [
            "_x",
            "_z",
            "_dx",
            "_dz",
            "_current",
            "_j_max",
            "_b_max",
            "_ctype",
        ]
        attributes = {k: [] for k in names}
        indexes = {}
        for name, attr_list in attributes.items():
            no_coils = 0
            for no, group in enumerate(coilgroups):
                child_attr = getattr(group, name)
                old_coils = no_coils
                no_coils += 1 if isinstance(child_attr, str) else len(child_attr)
                indexes[no] = (old_coils, no_coils)
                if (
                    len(child_attr) > 1 and not isinstance(child_attr, str)
                ) or isinstance(child_attr, list):
                    attributes[name].extend(child_attr)
                else:
                    attributes[name].append(child_attr)

            if isinstance(getattr(group, name), np.ndarray) and (
                attributes[name][0].dtype == float
                if isinstance(attributes[name][0], np.ndarray)
                else True
            ):
                attributes[name] = np.squeeze(np.array(attributes[name], dtype=float))
            else:
                attributes[name] = np.array(attributes[name], dtype=object)

            for no, group in enumerate(coilgroups):
                index_slice = slice(indexes[no][0], indexes[no][1])
                setattr(group, name, attributes[name][index_slice])

        return attributes

    @property
    def name(self):
        """
        Names of Coilset
        """
        return list(self.__coilgroups.keys())

    def get_coil(self, name_or_id):
        """
        Get an individual coil
        """
        # Actually all coils could just be attributes eg coilset.PF_1
        # all groups coilset.PF.current = 5
        pass

    def _define_subset(self, filters: Dict[str, Callable]):
        # Create new subgroup of coils

        self._filters = {
            "PF": lambda coilgroup: np.array(
                [ct is CoilType.PF for ct in coilgroup.ctype], dtype=bool
            ),
            "CS": lambda coilgroup: np.array(
                [ct is CoilType.CS for ct in coilgroup.ctype], dtype=bool
            ),
            **filters,
        }

    def add_subset(self, filters: Dict[str, Callable]):
        """
        Subset filtering
        """
        self._filters = {**self._filters, **filters}

        self._finalise_groups()

    def _finalise_groups(self):
        self._define_subgroup(self._filters.keys())

        self._group_ind = {
            self._SubGroup._all: slice(None),
            **{self._SubGroup[f_k]: filt(self) for f_k, filt in self._filters.items()},
        }

    def __getattribute__(self, attr):
        """
        Get attribute with extra for subgroups
        """
        try:
            return super().__getattribute__(attr)
        except AttributeError as ae:
            if attr != "__coilgroups":
                try:
                    return self.__coilgroups[attr]
                except KeyError:
                    # try:
                    #     return self.__coilgroups[self._group_ind[self._SubGroup[attr]]]
                    # except KeyError:
                    raise ae
            else:
                raise ae

    def psi(self, x, z):
        return np.sum(super().psi(x, z), axis=-1)


# TODO or To remove (for imports)


class PlasmaCoil:
    """
    Dummy
    """

    pass


class Solenoid:
    """
    Dummy
    """

    pass


def symmetrise_coilset():
    """
    Dummy
    """
    pass


def check_coilset_symmetric():
    """
    Dummy
    """
    pass


def make_mutual_inductance_matrix():
    """
    Dummy
    """
    pass


CS_COIL_NAME = "{}"  # noqa: F401
PF_COIL_NAME = "{}"  # noqa: F401
NO_COIL_NAME = "{}"  # noqa: F401

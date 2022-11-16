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
from functools import update_wrapper, wraps

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
from bluemira.utilities.tools import is_num


class CoilFieldsMixin:

    __slots__ = (
        "_quad_dx",
        "_quad_dz",
        "_quad_x",
        "_quad_z",
        "_quad_weighting",
        "_einsum_str",
    )

    def __init__(self):
        if type(self) == CoilFieldsMixin:
            raise TypeError("Can't be initialised directly")

    def psi(self, x, z):
        """
        Calculate poloidal flux at (x, z)
        """
        return self.unit_psi(x, z) * self.current

    def psi_greens(self, pgreen):
        """
        Calculate plasma psi from Greens functions and current
        """
        return self.current * pgreen

    def unit_psi(self, x, z):
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
        return self.unit_Bx(x, z) * self.current

    def Bx_greens(self, bgreen):
        """
        Uses the Greens mapped dict to quickly compute the Bx
        """
        return self.current * bgreen

    def unit_Bx(self, x, z):
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
            x, z, self._unit_Bx_greens, self._unit_Bx_analytical
        )

    def Bz(self, x, z):
        """
        Calculate vertical magnetic field Bz at (x, z)
        """
        return self.unit_Bz(x, z) * self.current

    def Bz_greens(self, bgreen):
        """
        Uses the Greens mapped dict to quickly compute the Bx
        """
        return self.current * bgreen

    def unit_Bz(self, x, z):
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
            x, z, self._unit_Bz_greens, self._unit_Bz_analytical
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

        zero_coil_size = np.logical_or(np.isclose(self.dx, 0), np.isclose(self.dz, 0))

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
                self.x,
                self.z,
                self.dx,
                self.dz,
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
            self.x - self.dx - atol,
            self.x + self.dx + atol,
        )
        z_min, z_max = (
            self.z - self.dz - atol,
            self.z + self.dz + atol,
        )
        return (
            (x >= x_min[None])
            & (x <= x_max[None])
            & (z >= z_min[None])
            & (z <= z_max[None])
        )

    def _unit_B_greens(
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

    def _unit_Bx_greens(
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
        return self._unit_B_greens(
            greens_Bx, x, z, split, _quad_x, _quad_z, _quad_weight
        )

    def _unit_Bz_greens(
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
        return self._unit_B_greens(
            greens_Bz, x, z, split, _quad_x, _quad_z, _quad_weight
        )

    def _unit_B_analytical(
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
            coil_x = self.x
            coil_z = self.z
            coil_dx = self.dx
            coil_dz = self.dz

        return semianalytic(
            coil_x[None],
            coil_z[None],
            x[..., None],
            z[..., None],
            d_xc=coil_dx[None],
            d_zc=coil_dz[None],
        )

    def _unit_Bx_analytical(
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
        return self._unit_B_analytical(
            semianalytic_Bx, x, z, split, coil_x, coil_z, coil_dx, coil_dz
        )

    def _unit_Bz_analytical(
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
        return self._unit_B_analytical(
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
        Bx, Bz = eqcoil.Bx(self.x, self.z), eqcoil.Bz(self.x, self.z)
        if self.rc != 0:  # true divide errors for zero current coils
            a = MU_0 * self.current**2 / (4 * np.pi * self.x)
            fx = a * (np.log(8 * self.x / self._current_radius) - 1 + 0.25)

        else:
            fx = 0
        return np.array(
            [
                (self.current * Bz + fx) * 2 * np.pi * self.x,
                -self.current * Bx * 2 * np.pi * self.x,
            ]
        )

    def control_F(self, coil: CoilGroup = None):  # noqa :N802
        """
        Returns the Green's matrix element for the coil mutual force.

        \t:math:`Fz_{i,j}=-2\\pi X_i\\mathcal{G}(X_j,Z_j,X_i,Z_i)`
        """

        if coil.x == self.x and coil.z == self.z:
            # self inductance
            if self._current_radius != 0:
                a = MU_0 / (4 * np.pi * self.x)
                Bz = a * (np.log(8 * self.x / self._current_radius) - 1 + 0.25)
            else:
                Bz = 0
            Bx = 0  # Should be 0 anyway

        else:
            Bz = coil.unit_Bz(self.x, self.z)
            Bx = coil.unit_Bx(self.x, self.z)

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

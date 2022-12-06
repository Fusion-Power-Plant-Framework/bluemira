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

from enum import Enum, EnumMeta, auto
from typing import Iterable, Optional, Union

import numpy as np

from bluemira.base.constants import EPS
from bluemira.equilibria.coils._field import CoilFieldsMixin
from bluemira.equilibria.coils._tools import get_max_current
from bluemira.equilibria.constants import NBTI_B_MAX, NBTI_J_MAX
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.plotting import CoilGroupPlotter
from bluemira.utilities.tools import is_num

__all__ = ["CoilType", "Coil"]


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
        else:
            raise ValueError(f"Unknown coil type {ctype}")

        return idx


class Coil(CoilFieldsMixin):
    """
    Coil Object

    For use with PF/CS/passive coils. All coils have a rectangular cross section.

    Parameters
    ----------
    x: float
        Coil geometric centre x coordinate [m]
    z: float
        Coil geometric centre z coordinate [m]
    dx: float
        Coil radial half-width [m] from coil centre to edge (either side)
    dz: float
        Coil vertical half-width [m] from coil centre to edge (either side)
    name: str
        The name of the coil
    ctype: Union[str, CoilType]
        Type of coil as defined in CoilType
    current: float (default = 0)
        Coil current [A]
    j_max: float
        Maximum current density in the coil [A/m^2]
    b_max: float
        Maximum magnetic field at the coil [T]
    discretisation: float
        discretise the coil. The value (between 0 and 1) is the fractional value of
        the width and the height of the coils to discretise over.
        For example 0.5 will result in 4 magnetic filaments.
    n_turns: int
        Number of turns

    """

    def __init__(
        self,
        x: float,
        z: float,
        dx: float = None,
        dz: float = None,
        name: Optional[str] = None,
        ctype: Union[str, CoilType] = CoilType.NONE,
        current: float = 0,
        j_max: float = np.nan,
        b_max: float = np.nan,
        discretisation: float = 1,
        n_turns: int = 1,
    ):
        self._dx = None
        self._dz = None
        self._discretisation = 1
        self._flag_sizefix = None not in (dx, dz)

        self.x = x
        self.z = z
        self.dx = dx
        self.dz = dz
        self.discretisation = discretisation
        self.current = current
        self.j_max = j_max
        self.b_max = b_max
        self.ctype = ctype
        self.name = name
        self.n_turns = n_turns

        self._number = CoilNumber.generate(self.ctype)
        if self.name is None:
            self.name = f"{self._ctype.name}_{self._number}"

        # check if dx and not dz set
        # check of j max set
        self._validate_size()
        if not self._flag_sizefix and None in (self.dx, self.dz):
            self._dx, self._dz = 0, 0
            self._discretise()
            self._set_coil_attributes()

    def __repr__(self):
        """
        Pretty printing
        """
        return (
            f"{type(self).__name__}({self.name} ctype={self.ctype.name} x={self.x:.2g}"
            f" z={self.z:.2g} dx={self.dx:.2g} dz={self.dz:.2g} current={self.current:.2g}"
            f" j_max={self.j_max:.2g} b_max={self.b_max:.2g}"
            f" discretisation={self.discretisation:.2g})"
        )

    def plot(
        self,
        ax=None,
        subcoil: bool = True,
        label: bool = False,
        force: Optional[Iterable] = None,
        **kwargs,
    ):
        """
        Plot a Coil

        Parameters
        ----------
        ax: Optional[Axes]
            Matplotlib axis object
        subcoil: bool
            plot coil discretisations
        label: bool
            show coil labels on plot
        force: Optional[Iterable]
            force arrows iterable
        kwargs:
            passed to matplotlib plotting

        """
        return CoilGroupPlotter(
            self, ax=ax, subcoil=subcoil, label=label, force=force, **kwargs
        )

    def n_coils(self):
        """
        Number of coils in coil

        Notes
        -----
        Allows n_coils to be accessed if an individual coil or a CoilGroup
        """
        return 1

    @property
    def x(self) -> float:
        """Get coil x position"""
        return self._x

    @property
    def z(self) -> float:
        """Get coil z position"""
        return self._z

    @property
    def ctype(self) -> CoilType:
        """Get coil type"""
        return self._ctype

    @property
    def dx(self) -> float:
        """Get coil width (half)"""
        return self._dx

    @property
    def dz(self) -> float:
        """Get coil height (half)"""
        return self._dz

    @property
    def current(self) -> float:
        """Get coil current"""
        return self._current

    @property
    def j_max(self) -> float:
        """Get coil max current density"""
        return self._j_max

    @property
    def b_max(self) -> float:
        """Get coil max field"""
        return self._b_max

    @property
    def discretisation(self) -> float:
        """Get coil discretisation"""
        return self._discretisation

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
    def x_boundary(self):
        """Get coil x coordinate boundary"""
        if getattr(self, "_x_boundary") is not None:
            return self._x_boundary
        return self._make_boundary(self.x, self.z, self.dx, self.dz)[0]

    @property
    def z_boundary(self):
        """Get coil z coordinate boundary"""
        if getattr(self, "_z_boundary") is not None:
            return self._z_boundary
        return self._make_boundary(self.x, self.z, self.dx, self.dz)[1]

    @property
    def _quad_boundary(self):
        """Get coil quadrature x,z coordinate boundary"""
        return self._make_boundary(
            self._quad_x, self._quad_z, self._quad_dx, self._quad_dz
        )

    @x.setter
    def x(self, value: float):
        """Set coil x position"""
        self._x = float(value)
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @z.setter
    def z(self, value: float):
        """Set coil z position"""
        self._z = float(value)
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @ctype.setter
    def ctype(self, value: Union[str, CoilType]):
        """Set coil type"""
        self._ctype = (
            value
            if isinstance(value, CoilType)
            else CoilType[value[0] if isinstance(value, np.ndarray) else value]
        )

    @dx.setter
    def dx(self, value: float):
        """Set coil dx size"""
        self._dx = None if value is None else float(value)
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @dz.setter
    def dz(self, value: float):
        """Set coil dz size"""
        self._dz = None if value is None else float(value)
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @current.setter
    def current(self, value: float):
        """Set coil current"""
        self._current = float(value)
        if None not in (self.dx, self.dz) and not self._flag_sizefix:
            self.resize()

    @j_max.setter
    def j_max(self, value: float):
        """Set coil max current density"""
        self._j_max = float(value)
        if None not in (self.dx, self.dz) and not self._flag_sizefix:
            self.resize()

    @b_max.setter
    def b_max(self, value: float):
        """Set coil max field"""
        self._b_max = float(value)

    @discretisation.setter
    def discretisation(self, value: float):
        """Set coil discretisation"""
        self._discretisation = np.clip(float(value), EPS, 1)
        self._discretise()

    def assign_material(
        self,
        j_max=NBTI_J_MAX,
        b_max=NBTI_B_MAX,
    ) -> None:
        """
        Assigns EM material properties to coil

        Parameters
        ----------
        j_max: float
            Overwrite default constant material max current density [A/m^2]
        b_max: float
            Overwrite default constant material max field [T]

        Notes
        -----
        Will always modify both j_max and b_max of the coil with the either the default
        or specified values.

        """
        self.j_max = j_max
        self.b_max = b_max

    def get_max_current(self):
        """Get max current"""
        return (
            np.infty
            if np.isnan(self.j_max)
            else get_max_current(self.dx, self.dz, self.j_max)
        )

    def _discretise(self):
        """
        Discretise a coil for greens function magnetic field calculations

        Notes
        -----
        Only discretisation method currently implemented is rectangular fraction
        The discretisation value (between 0 and 1) is the fractional value of
        the width and the height of the coils to discretise over.
        For example 0.5 will result in 4 magnetic filaments.

        Possible improvement: multiple discretisations for different coils

        """
        self._quad_x = np.array([self.x])
        self._quad_z = np.array([self.z])
        self._quad_dx = np.array([self.dx])
        self._quad_dz = np.array([self.dz])
        self._quad_weighting = np.ones_like(self._quad_x)
        self._einsum_str = "...j, ...j -> ..."

        if self.discretisation < 1:
            # How fancy do we want the mesh or just smaller rectangles?
            self._rectangular_discretisation()

    def _validate_size(self):
        dx_spec = is_num(self.dx)
        dz_spec = is_num(self.dz)
        dxdz_spec = dx_spec and dz_spec

        if (dx_spec ^ dz_spec) and not dxdz_spec:
            # Check that we don't have dx = None and dz = float or vice versa
            raise EquilibriaError("Must specify either dx and dz or neither.")
        if dxdz_spec:
            # If dx and dz are specified, we presume the coil size should
            # remain fixed
            if not self._flag_sizefix:
                self._flag_sizefix = True

            self._set_coil_attributes()
            self._discretise()
        else:
            if not is_num(self.j_max):
                # Check there is a viable way to size the coil
                raise EquilibriaError("Must specify either dx and dz or j_max.")

            if self._flag_sizefix:
                self._flag_sizefix = False

    def _set_coil_attributes(self):
        self._current_radius = 0.5 * np.hypot(self.dx, self.dz)
        self._x_boundary, self._z_boundary = self._make_boundary(
            self.x, self.z, self.dx, self.dz
        )

    def _rectangular_discretisation(self):
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
        nx = np.maximum(1, np.ceil(self.dx * 2 / self.discretisation))
        nz = np.maximum(1, np.ceil(self.dz * 2 / self.discretisation))

        if not nx * nz == 1:
            sc_dx, sc_dz = self.dx / nx, self.dz / nz

            # Calculate sub-coil centroids
            x_sc = (self.x - self.dx) + sc_dx * np.arange(1, 2 * nx, 2)
            z_sc = (self.z - self.dz) + sc_dz * np.arange(1, 2 * nz, 2)
            x_sc, z_sc = np.meshgrid(x_sc, z_sc)

            self._quad_x = x_sc.flatten()
            self._quad_z = z_sc.flatten()
            self._quad_dx = np.full(x_sc.size, sc_dx)
            self._quad_dz = np.full(x_sc.size, sc_dz)

            self._quad_weighting = np.ones(x_sc.size) / x_sc.size

    def fix_size(self):
        """
        Fixes the size of all coils
        """
        self._flag_sizefix = True

    def resize(self, current: Optional[float] = None):
        """Resize coil given a current"""
        if not self._flag_sizefix:
            # Adjust the size of the coil
            self.dx, self.dz = self._make_size(current)
            self._set_coil_attributes()

    def _make_size(self, current=None):
        """
        Size the coil based on a current and a current density.
        """
        if current is None:
            current = self.current
        if not np.isnan(self.j_max):
            half_width = 0.5 * np.sqrt(abs(current) / self.j_max)
            return half_width, half_width
        else:
            return self.dx, self.dz

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
        x_boundary: np.ndarray
        z_boundary: np.ndarray

        Note
        ----
        Only rectangular coils

        """
        xx, zz = (np.ones((4, 1)) * x_c).T, (np.ones((4, 1)) * z_c).T
        x_boundary = xx + (dx * np.array([-1, 1, 1, -1])[:, None]).T
        z_boundary = zz + (dz * np.array([-1, -1, 1, 1])[:, None]).T
        return x_boundary, z_boundary

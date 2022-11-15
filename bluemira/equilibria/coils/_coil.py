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
from dataclasses import dataclass, field

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


@dataclass
class Coil:
    x: float
    z: float
    ctype: Union[str, CoilType]
    dx: float
    dz: float
    current: float
    j_max: float
    b_max: float
    discretisation: float
    name: Optional[str] = None

    _x: float = field(init=False, repr=False)
    _z: float = field(init=False, repr=False)
    _ctype: Union[str, CoilType] = field(init=False, repr=False, default=CoilType.NONE)
    _dx: float = field(init=False, repr=False, default=None)
    _dz: float = field(init=False, repr=False, default=None)
    _current: float = field(init=False, repr=False, default=0)
    _j_max: float = field(init=False, repr=False, default=np.nan)
    _b_max: float = field(init=False, repr=False, default=np.nan)
    _discretisation: float = field(init=False, repr=False, default=1)
    _quad_x: np.ndarray = field(init=False, repr=False)
    _quad_z: np.ndarray = field(init=False, repr=False)
    _quad_dx: np.ndarray = field(init=False, repr=False)
    _quad_dz: np.ndarray = field(init=False, repr=False)
    _number: int = field(init=False, repr=False)
    _current_radius: float = field(init=False, repr=False)
    _x_boundary: float = field(init=False, repr=False)
    _z_boundary: float = field(init=False, repr=False)
    _flag_sizefix: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self._number = CoilNumber.generate(self.ctype)
        if self.name is None:
            self.name = f"{self._ctype.name}_{self._number}"

        # check if dx and not dz set
        # check of j max set
        self.validate_size()

    @property
    def x(self) -> float:
        return self._x

    @property
    def z(self) -> float:
        return self._z

    @property
    def ctype(self) -> CoilType:
        return self._ctype

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def dz(self) -> float:
        return self._dz

    @property
    def current(self):
        return self._current

    @property
    def j_max(self) -> float:
        return self._j_max

    @property
    def b_max(self) -> float:
        return self._b_max

    @property
    def discretisation(self) -> float:
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
        return self._make_boundary(self.x, self.z, self.dx, self.dz)[0]

    @property
    def z_boundary(self):
        return self._make_boundary(self.x, self.z, self.dx, self.dz)[1]

    @x.setter
    def x(self, value: float):
        if type(value) is property:
            raise TypeError("__init__() missing required positional argument 'x'")
        self._x = value
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @z.setter
    def z(self, value: float):
        if type(value) is property:
            raise TypeError("__init__() missing required positional argument 'z'")
        self._z = value
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @ctype.setter
    def ctype(self, value: Union[str, CoilType]):
        if type(value) is property:
            # initial value not specified, use default
            value = type(self)._ctype
        import ipdb

        ipdb.set_trace()
        self._ctype = value if isinstance(value, CoilType) else CoilType[value]

    @dx.setter
    def dx(self, value: float):
        import ipdb

        ipdb.set_trace()
        if type(value) is property:
            # initial value not specified, use default
            value = type(self)._dx
        self._dx = value
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @dz.setter
    def dz(self, value: float):
        if type(value) is property:
            # initial value not specified, use default
            value = type(self)._dz
        self._dz = value
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @current.setter
    def current(self, value: Union[str, CoilType]):
        if type(value) is property:
            # initial value not specified, use default
            value = type(self)._current
        self._current = value

    @j_max.setter
    def j_max(self, value: float):
        if type(value) is property:
            # initial value not specified, use default
            value = type(self)._j_max
        self._j_max = value
        self.resize()

    @b_max.setter
    def b_max(self, value: float):
        if type(value) is property:
            # initial value not specified, use default
            value = type(self)._b_max
        self._b_max = value

    @discretisation.setter
    def discretisation(self, value: float):
        if type(value) is property:
            # initial value not specified, use default
            value = type(self)._discretisation
        self._discretisation = value
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
        j_max: float (default None)
            Overwrite default constant material max current density [A/m^2]
        b_max: float (default None)
            Overwrite default constant material max field [T]

        """
        if not is_num(j_max):
            raise EquilibriaError(f"j_max must be specified as a number, not: {jm}")
        if not is_num_(b_max):
            raise EquilibriaError(f"b_max must be specified as a number, not: {bm}")

        self.j_max = j_max
        self.b_max = b_max

    def _discretise(self):
        """
        Discretise a coil for greens function magnetic field calculations

        Notes
        -----
        Only discretisation method currently implemented is rectangular fraction

        Possible improvement: multiple discretisations for different coils

        """
        weighting = None
        self._quad_x = np.array([self.x])
        self._quad_z = np.array([self.z])
        self._quad_dx = np.array([self.dx])
        self._quad_dz = np.array([self.dz])

        if self.discretisation < 1:
            # How fancy do we want the mesh or just smaller rectangles?
            self._rectangular_discretisation()

    def validate_size(self):
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

        if not all(nx * nz == 1):
            sc_dx, sc_dz = self.dx / nx, self.dz / nz

            # Calculate sub-coil centroids
            x_sc = (self.x - self.dx) + sc_dx * np.arange(1, 2 * nx, 2)
            z_sc = (self.z - self.dz) + sc_dz * np.arange(1, 2 * nz, 2)
            x_sc, z_sc = np.meshgrid(x_sc, z_sc)

            self._quad_x = x_sc.flatten()
            self._quad_z = z_sc.flatten()
            self._quad_dx = np.ones(x_sc.size) * sc_dx
            self._quad_dz = np.ones(x_sc.size) * sc_dz

    def fix_size(self):
        """
        Fixes the size of all coils
        """
        self._flag_sizefix = True

    def resize(self, current: Optional[float] = None):
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
        x_boundary, z_boundary: np.array

        Note
        ----
        Only rectangular coils

        """
        xx, zz = (np.ones((4, 1)) * x_c).T, (np.ones((4, 1)) * z_c).T
        x_boundary = xx + (dx * np.array([-1, 1, 1, -1])[:, None]).T
        z_boundary = zz + (dz * np.array([-1, -1, 1, 1])[:, None]).T
        return x_boundary, z_boundary

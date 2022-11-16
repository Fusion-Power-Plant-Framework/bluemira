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
from operator import attrgetter

# from re import split
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

# import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.coils._field import CoilFieldsMixin
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
from bluemira.utilities.tools import flatten_iterable, is_num

# from scipy.interpolate import RectBivariateSpline


class CoilGroup(CoilFieldsMixin):
    def __init__(self, *coils: Union[Coil, CoilGroup[Coil]]) -> None:
        self._coils = coils
        self._pad_discretisation(self.__list_getter("_quad_x"))

    def __list_getter(self, attr):
        return np.frompyfunc(attrgetter(attr), 1, 1)(self._coils)

    def __getter(self, attr):
        return np.array([*flatten_iterable(self.__list_getter(attr))])

    def __quad_getter(self, attr):
        _quad_list = self.__list_getter(attr)

        for i, d in enumerate(self._pad_size):
            _quad_list[i] = np.pad(_quad_list[i], (0, d))

        return np.array(_quad_list.tolist())

    def __setter(self, attr, values):
        no = 0
        for coil in flatten_iterable(self._coils):
            end_no = no + coil.n_coils
            setattr(coil, attr, values[no:end_no])
            no = end_no

    def _pad_discretisation(
        self,
        _to_pad: List[np.ndarray],
    ):
        """
        Convert quadrature list of array to rectuangualr arrays.
        Padding quadrature arrays with zeros to allow array operations
        on rectangular matricies.

        Parameters
        ----------
        _to_pad: List[np.ndarray]
            x quadratures

        Notes
        -----
        Padding exists for coils with different discretisations or sizes within a coilgroup.
        There are a few extra calculations of the greens functions where padding exists in
        the :func:_combined_control method.

        """
        all_len = np.array([len(q) for q in _to_pad])
        max_len = max(all_len)
        self._pad_size = max_len - all_len

        self._einsum_str = (
            "..., ...j -> ..." if all(self._pad_size == 0) else "...j, ...j -> ..."
        )

    @property
    def n_coils(self):
        n = 0
        for cg in flatten_iterable(self._coils):
            n += cg.n_coils
        return n

    @property
    def name(self):
        return self.__getter("name")

    @property
    def x(self) -> np.ndarray:
        return self.__getter("x")

    @property
    def z(self) -> float:
        return self.__getter("z")

    @property
    def ctype(self):
        return self.__getter("ctype")

    @property
    def dx(self) -> float:
        return self.__getter("dx")

    @property
    def dz(self) -> float:
        return self.__getter("dz")

    @property
    def current(self) -> float:
        return self.__getter("current")

    @property
    def j_max(self) -> float:
        return self.__getter("j_max")

    @property
    def b_max(self) -> float:
        return self.__getter("b_max")

    @property
    def discretisation(self) -> float:
        return self.__getter("discretisation")

    @property
    def area(self) -> np.ndarray:
        return self.__getter("area")

    @property
    def volume(self) -> np.ndarray:
        return self.__getter("volume")

    @property
    def x_boundary(self) -> np.ndarray:
        return self.__getter("x_boundary")

    @property
    def z_boundary(self) -> np.ndarray:
        return self.__getter("z_boundary")

    @x.setter
    def x(self, values):
        self.__setter("x", values)

    @z.setter
    def z(self, values):
        self.__setter("z", values)

    @ctype.setter
    def ctype(self, values):
        values = np.array(values, dtype=object)
        self.__setter("ctype", values)

    @dx.setter
    def dx(self, values):
        self.__setter("dx", values)

    @dz.setter
    def dz(self, values):
        self.__setter("dz", values)

    @current.setter
    def current(self, values):
        self.__setter("current", values)

    @j_max.setter
    def j_max(self, values):
        self.__setter("j_max", values)

    @b_max.setter
    def b_max(self, values):
        self.__setter("b_max", values)

    @discretisation.setter
    def discretisation(self, values):
        self.__setter("discretisation", values)
        self._pad_discretisation(self.__list_getter("_quad_x"))

    def assign_material(self, j_max, b_max):
        self.j_max = j_max
        self.b_max = b_max

    @property
    def _current_radius(self):
        return self.__getter("_current_radius")

    @property
    def _quad_x(self):
        return self.__quad_getter("_quad_x")

    @property
    def _quad_z(self):
        return self.__quad_getter("_quad_z")

    @property
    def _quad_dx(self):
        return self.__quad_getter("_quad_dx")

    @property
    def _quad_dz(self):
        return self.__quad_getter("_quad_dz")

    @property
    def _quad_weighting(self):
        return self.__quad_getter("_quad_weighting")


class _CoilGroup(abc.ABC):
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
        "_current_radius",
        "_x",
        "_x_boundary",
        "_z",
        "_z_boundary",
    )

    def __init__(self, *coils: Union[Coil, Iterable[Coil]]) -> None:

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

        self._flag_sizefix = False
        self._sizer = CoilSizer(self)
        self._sizer(self)

        # Meshing
        super().__init__(d_coil)

        # Lastly number coils to minimise incrementing coil numbers
        # on failing initialisation.
        self._index = [CoilNumber.generate(ct) for ct in self.ctype]

        self._name_map = {
            f"{self._ctype[en].name}_{ind}" if n is None else n: ind
            for en, (n, ind) in enumerate(zip(_inputs["name"], self._index))
        }

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
    def n_coils(self) -> int:
        """
        Get number of coils in group
        """
        return len(self.x)

    @property
    def name(self):
        """
        Get names of coils
        """
        return list(self._name_map.keys())

    @property
    def current_radius(self):
        """
        TODO
        """
        return self._current_radius

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


class _Coil(CoilGroup):
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
        # ctype: Optional[Union[str, CoilType]] = CoilType.PF,
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


class _Circuit(CoilGroup):
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


class _SymmetricCircuit(_Circuit):
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
        # ctype: Optional[Union[str, CoilType]] = CoilType.PF,
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

    @_Circuit.x.setter
    def x(self, new_x: float) -> None:
        """
        Set x coordinate of each coil
        """
        self._x[0] = self._point[0] = new_x
        self._x[1] = self._point[0] - self._symmetrise()[0]
        self._sizer(self)

    @_Circuit.z.setter
    def z(self, new_z: float) -> None:
        """
        Set z coordinate of each coil
        """
        self._z[0] = self._point[1] = new_z
        self._z[1] = self._point[1] - self._symmetrise()[1]
        self._sizer(self)

    # @_Circuit.position.setter
    # def position(self, new_position: __ITERABLE_FLOAT):
    #     """
    #     Set position of each coil
    #     """
    #     self.x = new_position[0, 0]
    #     self.z = new_position[0, 1]


class _CoilSet(CoilGroup):
    """
    Coilset is the main interface for groups of coils in bluemira

    """

    __slots__ = (
        "__coilgroups",
        "_circuits",
        "_control",
        "_has_circuits",
    )

    def __init__(self, *coils: Union[CoilGroup, List, Dict], d_coil=None):

        if not coils:
            raise ValueError("No coils provided")

        attributes = self._process_coilgroups(self._convert_to_coilgroup(coils))

        for k, v in attributes.items():
            setattr(self, k, v)

        self._circuit_mechanics()

        self.discretise(d_coil)

        self._control = np.ones_like(self._x, dtype=bool)

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

    def _circuit_mechanics(self):
        self._circuits = np.array(
            [isinstance(cg, Circuit) for cg in self.__coilgroups.values()], dtype=bool
        )

        self._has_circuits = any(self._circuits)

        return
        no_coils = len(self._x)
        _circuit_index = [0]
        for c in range(no_coils):
            if not (self._circuits[c] and c):
                _circuit_index.append(no + 1)
            else:
                pass
        self._circuit_index = np.array(_circuit_index)

    @CoilGroup.x.setter
    def x(self, new_x):
        """
        https://stackoverflow.com/questions/10810369/python-super-and-setting-parent-class-property
        """
        if self._has_circuits:
            raise NotImplementedError
        super(CoilSet, self.__class__).x.fset(self, new_x)

    @CoilGroup.z.setter
    def z(self, new_z):
        if self._has_circuits:
            raise NotImplementedError
        super(CoilSet, self.__class__).z.fset(self, new_z)

    @CoilGroup.dx.setter
    def dx(self, new_dx):
        if self._has_circuits:
            raise NotImplementedError
        super(CoilSet, self.__class__).dx.fset(self, new_dx)

    @CoilGroup.dz.setter
    def dz(self, new_dz):
        if self._has_circuits:
            raise NotImplementedError
        super(CoilSet, self.__class__).dz.fset(self, new_dz)

    @CoilGroup.current.setter
    def current(self, new_current):
        if self._has_circuits:
            raise NotImplementedError
        super(CoilSet, self.__class__).current.fset(self, new_current)

    @property
    def control_current(self) -> np.ndarray:
        """
        Get coil current
        """
        return self.current[self._control]

    @control_current.setter
    def control_current(self, new_current: __ITERABLE_FLOAT) -> None:
        """
        Set coil current
        """
        if self._has_circuits:
            raise NotImplementedError
        self.current[self._control] = new_current

    @property
    def control_x(self) -> np.ndarray:
        """
        Get control coil x positions
        """
        return self.x[self._control]

    @control_x.setter
    def control_x(self, new_x: __ITERABLE_FLOAT) -> None:
        """
        Set coil control_x
        """
        if self._has_circuits:
            raise NotImplementedError
        self.x[self._control] = new_x

    @property
    def control_z(self) -> np.ndarray:
        """
        Get control coil z positions
        """
        return self.z[self._control]

    @control_z.setter
    def control_z(self, new_z: __ITERABLE_FLOAT) -> None:
        """
        Set coil control_z
        """
        if self._has_circuits:
            raise NotImplementedError
        self.z[self._control] = new_z

    def _controller(func):
        @wraps(func)
        def _control_wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)[..., self._control]

        return _control_wrapper

    # control_Bx = _controller(CoilGroup.unit_Bx)
    # control_Bz = _controller(CoilGroup.unit_Bz)
    # control_psi = _controller(CoilGroup.unit_psi)

    def _summer(func):
        @wraps(func)
        def _sum_wrapper(self, *args, **kwargs):
            return np.squeeze(np.sum(func(self, *args, **kwargs), axis=-1))

        return _sum_wrapper

    # Bx = _summer(CoilGroup.Bx)
    # Bz = _summer(CoilGroup.Bz)
    # psi = _summer(CoilGroup.psi)
    # Bx_greens = _summer(CoilGroup.Bx_greens)
    # Bz_greens = _summer(CoilGroup.Bz_greens)
    # psi_greens = _summer(CoilGroup.psi_greens)
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

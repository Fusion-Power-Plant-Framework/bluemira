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

from copy import deepcopy
from operator import attrgetter
from typing import Iterable, List, Optional, Tuple, Type, Union

import numpy as np

from bluemira.base.constants import MU_0
from bluemira.equilibria.coils._coil import Coil, CoilType
from bluemira.equilibria.coils._field import CoilGroupFieldsMixin
from bluemira.equilibria.constants import I_MIN, NBTI_B_MAX, NBTI_J_MAX, X_TOLERANCE
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.plotting import CoilPlotter, CoilSetPlotter
from bluemira.utilities.tools import flatten_iterable, yintercept


class CoilGroup(CoilGroupFieldsMixin):
    def __init__(self, *coils: Union[Coil, CoilGroup[Coil]]):
        self._coils = coils
        self._pad_discretisation(self.__list_getter("_quad_x"))

    def add_coil(self, *coils: Union[Coil, CoilGroup[Coil]]):
        self._coils = (*self._coils, *coils)
        self._pad_discretisation(self.__list_getter("_quad_x"))

    def remove_coil(self, *coil_name, _top_level=True):
        # have to navigate nested
        names = self.name

        if _top_level and (remainder := set(coil_name) - set(names)):
            raise EquilibriaError(f"Unknown coils {remainder}")

        c_names = []
        removed_names = []

        for c in self._coils:
            if isinstance(c, CoilGroup):
                removed_names.extend(c.remove_coil(*coil_name, _top_level=False))
            elif c.name in coil_name:
                c_names.append(c.name)

        to_remove = [names.index(c_n) for c_n in c_names]

        coils = np.array(self._coils, dtype=object)
        mask = np.full(coils.size, True, dtype=bool)
        mask[to_remove] = False

        for no, c in enumerate(self._coils):
            if c.name == []:
                mask[no] = False

        self._coils = tuple(coils[mask])
        self._pad_discretisation(self.__list_getter("_quad_x"))

        if not _top_level:
            return removed_names + to_remove

    def __list_getter(self, attr: str):
        return np.frompyfunc(attrgetter(attr), 1, 1)(self._coils)

    def __getter(self, attr: str):
        return np.array([*flatten_iterable(self.__list_getter(attr))])

    def __quad_getter(self, attr: str):
        _quad_list = self.__list_getter(attr)

        for i, d in enumerate(self._pad_size):
            _quad_list[i] = np.pad(_quad_list[i], (0, d))

        return np.array(_quad_list.tolist())

    def __setter(
        self,
        attr: str,
        values: Union[CoilType, float, Iterable[Union[CoilType, float]]],
        dtype: Union[Type, None] = None,
    ):
        values = np.array([values], dtype=dtype)
        no_val = values.size
        no = 0
        for coil in flatten_iterable(self._coils):
            end_no = no + coil.n_coils
            if end_no > no_val:
                if no_val == 1:
                    setattr(coil, attr, np.repeat(values[0], end_no - no))
                else:
                    raise ValueError(
                        "The number of elements is less than the number of coils"
                    )
            else:
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
            "...j, ...ij -> ...i" if len(_to_pad) == 1 else "...ij, ...ij -> ...i"
        )

    def assign_material(
        self, j_max: Union[float, Iterable[float]], b_max: Union[float, Iterable[float]]
    ):
        self.j_max = j_max
        self.b_max = b_max

    @property
    def n_coils(self) -> int:
        n = 0
        for cg in flatten_iterable(self._coils):
            n += cg.n_coils
        return n

    @property
    def name(self) -> np.ndarray:
        return self.__getter("name").tolist()

    @property
    def x(self) -> np.ndarray:
        return self.__getter("x")

    @property
    def z(self) -> np.ndarray:
        return self.__getter("z")

    @property
    def ctype(self) -> np.ndarray:
        return self.__getter("ctype").tolist()

    @property
    def dx(self) -> np.ndarray:
        return self.__getter("dx")

    @property
    def dz(self) -> np.ndarray:
        return self.__getter("dz")

    @property
    def current(self) -> np.ndarray:
        return self.__getter("current")

    @property
    def j_max(self) -> np.ndarray:
        return self.__getter("j_max")

    @property
    def b_max(self) -> np.ndarray:
        return self.__getter("b_max")

    @property
    def discretisation(self) -> np.ndarray:
        return self.__getter("discretisation")

    @property
    def area(self) -> np.ndarray:
        return self.__getter("area")

    @property
    def volume(self) -> np.ndarray:
        return self.__getter("volume")

    @property
    def x_boundary(self) -> np.ndarray:
        xb = self.__getter("x_boundary")
        if self.n_coils > 1:
            return xb.reshape(-1, 4)
        else:
            return xb

    @property
    def z_boundary(self) -> np.ndarray:
        zb = self.__getter("z_boundary")
        if self.n_coils > 1:
            return zb.reshape(-1, 4)
        else:
            return zb

    @property
    def _current_radius(self) -> np.ndarray:
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

    @x.setter
    def x(self, values: Union[float, Iterable[float]]):
        self.__setter("x", values)

    @z.setter
    def z(self, values: Union[float, Iterable[float]]):
        self.__setter("z", values)

    @ctype.setter
    def ctype(self, values: Union[CoilType, Iterable[CoilType]]):
        self.__setter("ctype", values, dtype=object)

    @dx.setter
    def dx(self, values: Union[float, Iterable[float]]):
        self.__setter("dx", values)

    @dz.setter
    def dz(self, values: Union[float, Iterable[float]]):
        self.__setter("dz", values)

    @current.setter
    def current(self, values: Union[float, Iterable[float]]):
        self.__setter("current", values)

    @j_max.setter
    def j_max(self, values: Union[float, Iterable[float]]):
        self.__setter("j_max", values)

    @b_max.setter
    def b_max(self, values: Union[float, Iterable[float]]):
        self.__setter("b_max", values)

    @discretisation.setter
    def discretisation(self, values: Union[float, Iterable[float]]):
        self.__setter("discretisation", values)
        self._pad_discretisation(self.__list_getter("_quad_x"))

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class Circuit(CoilGroup):
    def __init__(
        self, *coils: Union[Coil, CoilGroup[Coil]], current: Optional[float] = None
    ):
        super().__init__(*coils)
        self.current = self._coils[0].current if current is None else current

    def add_coil(self, *coils: Union[Coil, CoilGroup[Coil]]):
        super().add_coil(coils)
        self.current = self._coils[0].current

    @CoilGroup.current.setter
    def current(self, values: Union[float, Iterable[float]]):
        if isinstance(values, Iterable):
            # Force the same value of current for all coils
            values = values[0]
        self._CoilGroup__setter("current", values)


class SymmetricCircuit(Circuit):
    def __init__(
        self,
        *coils: Union[Coil, CoilGroup[Coil]],
        symmetry_line: Union[Tuple, np.ndarray] = ((0, 0), (1, 0)),
    ):
        if len(coils) != 2:
            raise EquilibriaError(
                f"Wrong number of coils to create a {type(self).__name__}"
            )

        super().__init__(*coils)

        self.modify_symmetry(symmetry_line)
        diff = self._symmetrise()
        self._coils[1].x -= diff[0]
        self._coils[1].z -= diff[1]

    def modify_symmetry(self, symmetry_line: np.ndarray):
        """
        Create a unit vector for the symmetry of the circuit

        Parameters
        ----------
        symmetry_line: np.ndarray[[float, float], [float, float]]
            two points making a symmetry line

        """
        if isinstance(symmetry_line, tuple):
            self._symmetry_line = np.array(symmetry_line)

        self._symmetry_matrix()

    def _symmetry_matrix(self):
        self._shift, grad = yintercept(self._symmetry_line)
        grad2 = grad**2
        mgrad2 = 1 - grad2
        gradsq = 2 * grad
        self.sym_mat = 1 / (1 + grad2) * np.array([[mgrad2, gradsq], [gradsq, -mgrad2]])

    def _symmetrise(self):
        """
        Calculate the change in position to the symmetric coil,
        twice the distance to the line of symmetry.
        """
        cp = self._get_group_centre()
        cp[1] -= self._shift
        mirror = np.dot(self.sym_mat, cp.T)
        mirror[1] += self._shift
        return np.array([np.mean(self._coils[1].x), np.mean(self._coils[1].z)]) - mirror

    @Circuit.x.setter
    def x(self, new_x: float):
        """
        Set x coordinate of each coil
        """
        self._coils[0].x += np.mean(new_x) - self._get_group_x_centre()
        self._coils[1].x -= self._symmetrise()[0]

    @Circuit.z.setter
    def z(self, new_z: float):
        """
        Set z coordinate of each coil
        """
        self._coils[0].z += np.mean(new_z) - self._get_group_z_centre()
        self._coils[1].z -= self._symmetrise()[1]

    def _get_group_x_centre(self):
        return np.mean(self._coils[0].x)

    def _get_group_z_centre(self):
        return np.mean(self._coils[0].z)

    def _get_group_centre(self):
        return np.array([self._get_group_x_centre(), self._get_group_z_centre()])


# TODO or To remove (for imports)


class _Solenoid(CoilGroup):
    """
    Dummy
    """

    pass

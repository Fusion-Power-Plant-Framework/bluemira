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
Module containing the base Component class.
"""

from bluemira.base.components import (
    Component,
    MagneticComponent
)
from typing import Any

class Plasma(MagneticComponent):

    def __init__(
        self,
        name: str,
        shape: Any,
        material: Any = None,
        conductor: Any = None,
        mhd_solver: Any = None,
        gs_solver: Any = None,
        parent: Component = None,
        children: Component = None,
    ):
        super().__init__(name, material, conductor, parent, children)
        self._mhd_solver = mhd_solver
        self._gs_solver = gs_solver

        def set_mhd_solver(self,solver):
            self._mhd_solver = solver

        def set_gs_solver(self,solver):
            self._gs_solver = solver

    @property
    def _pprime(self):
        return self._mhd_solver.pprime

    @property
    def _ffprime(self):
        return self._mhd_solver.ffprime

    @property
    def _psi(self):
        def wrapper(points):
            return self._gs_solver.psi(points)
        return wrapper

    @property
    def psi_ax(self):
        return self._gs_solver.psi_max

    def curr_density(self, j0=0):
        """Toroidal plasma current density"""
        def wrapper(points):
            r = points[0]
            a = 0
            b = 0
            if self.psi_ax > 0:
                psi_norm = (self.psi_ax - self._psi(points))/self.psi_ax
                if self._pprime is not None:
                    a = -const.MU_0*r*self._pprime(psi_norm)
                if self._ffprime is not None:
                    b = - 1/r*self._ffprime(psi_norm)
            return j0 - 1/const.MU_0*(a + b)
        return wrapper



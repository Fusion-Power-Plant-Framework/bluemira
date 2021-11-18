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

        def _pprime(self):
            return self._mhd_solver.get_pprime()

        def _ffprime(self):
            return self._mhd_solver.get_ffprime()

        def curr_density(self):
            """Toroidal plasma current density"""






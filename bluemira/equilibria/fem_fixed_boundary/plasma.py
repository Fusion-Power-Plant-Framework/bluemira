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
import math

import bluemira.base.constants as const
from bluemira.mesh import meshing, msh2xdmf

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
        super().__init__(name, shape, material, conductor, parent, children)
        self._mhd_solver = mhd_solver
        self._gs_solver = gs_solver
        self.mesh_dict = {}

    def set_mhd_solver(self, solver):
        self._mhd_solver = solver

    def set_gs_solver(self, solver):
        self._gs_solver = solver

    def _pprime(self):
        return self._mhd_solver.get_pprime()

    def _ffprime(self):
        return self._mhd_solver.get_ffprime()

    def _psi(self):
        def wrapper(points):
            return self._mhd_solver.psi(points)
        return wrapper

    def curr_density(self):
        """Toroidal plasma current density"""
        def wrapper(points):
            r = points[0]
            return -1/const.MU_0*(-const.MU_0*r*self._pprime(self._psi(points))
                                  - 1/r*self._ffprime(self._psi(points)))
        return wrapper

    def calculate_mesh(self):
        m = meshing.Mesh()
        buffer = m(self.shape)
        msh2xdmf.msh2xdmf("Mesh.msh", dim=2, directory=".")
        mesh, boundaries, subdomains, labels = msh2xdmf.import_mesh(
            prefix="Mesh",
            dim=2,
            directory=".",
            subdomains=True,
        )
        self.mesh_dict = {'mesh': mesh, 'boundaries': boundaries,
                          'subdomains': subdomains, 'labels': labels}

    def calculate_plasma_parameters(self):
        self.lp = self.shape.length
        self.Ap = self.shape.area
        self.Sp = 2*math.pi*self.shape.center_of_mass[0]*self.lp
        self.Vp = 2*math.pi*self.shape.center_of_mass[0]*self.Ap

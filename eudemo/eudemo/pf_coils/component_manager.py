# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
"""Component Manager for PF coils."""
from bluemira.base.builder import ComponentManager


class PFCoil(ComponentManager):
    """
    Wrapper around the PF Coil component tree.
    """

    def __init__(self, component, coilset):
        super().__init__(component)
        self._coilset = coilset

    @property
    def coilset(self):
        """
        The poloidal coilset
        """
        return self._coilset

    def xz_boundary(self):
        """
        Boundaries of the coils in xz
        """
        return self.PF_xz_boundary() + self.CS_xz_boundary()

    def PF_xz_boundary(self):
        """
        Boundaries of the PF coils in xz
        """
        return [
            pf.get_component("Casing").shape.boundary[0]
            for pf in self.component()
            .get_component("PF coils")
            .get_component("xz", first=False)
        ]

    def CS_xz_boundary(self):
        """
        Boundaries of the CS coils in xz
        """
        return [
            pf.get_component("Casing").shape.boundary[0]
            for pf in self.component()
            .get_component("CS coils")
            .get_component("xz", first=False)
        ]

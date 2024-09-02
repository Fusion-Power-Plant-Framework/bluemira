# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Component Manager for PF coils."""

from bluemira.base.components import PhysicalComponent
from bluemira.base.reactor import ComponentManager
from bluemira.equilibria.coils._grouping import CoilSet


class PFCoil(ComponentManager):
    """
    Wrapper around the PF Coil component tree.
    """

    def __init__(self, component, coilset):
        super().__init__(component)
        self._coilset = coilset

    @property
    def coilset(self) -> CoilSet:
        """
        Returns
        -------
        :
            The poloidal coilset
        """
        return self._coilset

    def xz_boundary(self) -> list[PhysicalComponent]:
        """
        Boundaries of the coils in xz

        Returns
        -------
        :
            The boundaries of the PF an CS coils
        """
        return self.PF_xz_boundary() + self.CS_xz_boundary()

    def PF_xz_boundary(self) -> list[PhysicalComponent]:
        """
        Returns
        -------
        :
            Boundaries of the PF coils in xz
        """
        return [
            pf.get_component("Casing").shape.boundary[0]
            for pf in self.component()
            .get_component("PF coils")
            .get_component("xz", first=False)
        ]

    def CS_xz_boundary(self) -> list[PhysicalComponent]:
        """
        Returns
        -------
        :
            Boundaries of the CS coils in xz
        """
        return [
            pf.get_component("Casing").shape.boundary[0]
            for pf in self.component()
            .get_component("CS coils")
            .get_component("xz", first=False)
        ]

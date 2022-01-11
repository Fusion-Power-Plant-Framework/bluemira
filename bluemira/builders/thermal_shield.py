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
Built-in build steps for making a parameterised thermal shield.
"""

from typing import List

import numpy as np

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Builder
from bluemira.base.components import Component


class ThermalShieldBuilder(Builder):
    """
    Builder for the thermal shield
    """

    _required_params: List[str] = [
        None,
    ]

    def build(self, label: str = "Thermal Shield", **kwargs) -> Component:
        """
        Build the PF Coils component.
        Returns
        -------
        component: PFCoilsComponent
            The Component built by this builder.
        """
        super().build(**kwargs)

        component = Component(name=label)
        component.add_child(self.build_xz())
        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())
        return component

    def build_xz(self):
        """
        Build the x-z components of the thermal shield.
        """
        pass

    def build_xy(self):
        """
        Build the x-y components of the thermal shield.
        """
        pass

    def build_xyz(self):
        """
        Build the x-y-z components of the thermal shield.
        """
        pass

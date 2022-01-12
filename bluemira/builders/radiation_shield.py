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
Radiation shield builder
"""

from typing import List

import numpy as np

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.builders.EUDEMO.tools import circular_pattern_component
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, make_polygon, revolve_shape


class RadiationShieldBuilder(Builder):
    """
    Builder for the cryostat
    """

    required_params: List[str] = [
        "tk_rs",
        "g_cr_rs",
        "o_p_rs",
        "n_rs_lab",
        "rs_l_d",
        "rs_l_gap",
        "n_TF",
    ]

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        # Seems we need to override this so it isn't an abstract method
        return super().reinitialise(params, **kwargs)

    def build(self, label: str, cryostat_vv, **kwargs) -> Component:
        """
        Build the radiation shield component.

        Returns
        -------
        component: Component
            The Component built by this builder.
        """
        super().build(**kwargs)

        self._cryo_vv = cryostat_vv

        component = Component(name=label)
        component.add_child(self.build_xz())
        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())
        return component

    def build_xz(self):
        """
        Build the x-z components of the radiation shield.
        """
        pass

    def build_xy(self):
        """
        Build the x-y components of the radiation shield.
        """
        pass

    def build_xyz(self):
        """
        Build the x-y-z components of the radiation shield.
        """
        pass

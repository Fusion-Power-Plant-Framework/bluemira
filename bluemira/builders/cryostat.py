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
Cryostat builder
"""

from typing import List

import numpy as np

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.builders.EUDEMO.tools import circular_pattern_component
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    make_circle,
    make_polygon,
    offset_wire,
    revolve_shape,
    slice_shape,
)


class CryostatBuilder(Builder):
    required_params: List[str] = [
        "tk_cr_vv",
        "g_cr_ts",
        "o_p_cr",
        "n_cr_lab",
        "cr_l_d",
        "n_TF",
    ]

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        # Seems we need to override this so it isn't an abstract method
        return super().reinitialise(params, **kwargs)

    def build(self, label: str, cryostat_ts, **kwargs) -> Component:
        """
        Build the cryostat component.

        Returns
        -------
        component: Component
            The Component built by this builder.
        """
        super().build(**kwargs)

        self._cts = cryostat_ts

        component = Component(name=label)
        component.add_child(self.build_xz())
        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())
        return component

    def build_xz(self):
        """
        Build the x-z components of the cryostat.
        """
        shape = None
        cryostat_vv = PhysicalComponent("Cryostat VV", shape)
        component = Component("xz", children=[cryostat_vv])
        bm_plot_tools.set_component_plane(component, "xz")
        return component

    def build_xz(self):
        """
        Build the x-y components of the cryostat.
        """
        shape = None
        cryostat_vv = PhysicalComponent("Cryostat VV", shape)
        component = Component("xy", children=[cryostat_vv])
        bm_plot_tools.set_component_plane(component, "xy")
        return component

    def build_xyz(self):
        """
        Build the x-y-z components of the cryostat.
        """
        component = Component("xyz")

        shape = None
        cryostat_vv = PhysicalComponent("Cryostat TS", shape)
        cryostat_vv.display_cad_options.color = BLUE_PALETTE["TS"][0]
        sectors = circular_pattern_component(cryostat_vv, self._params.n_TF.value)
        component.add_children(sectors, merge_trees=True)

        return component

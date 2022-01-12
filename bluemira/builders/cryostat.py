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
        "x_g_support",
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
        # Cryostat VV
        x_in = 0
        x_out, z_top = self._get_extrema()
        z_gs = -15  # TODO: Get from gravity support
        x_gs_kink = self.params.x_g_support - 2  # TODO: Get from a parameter
        well_depth = 5  # TODO: Get from a parameter
        z_mid = z_gs - self.params.g_cr_ts
        z_bot = z_mid - well_depth
        tk = self.params.tk_cr_vv.value

        x_inner = [x_in, x_out, x_out, x_gs_kink, x_gs_kink, x_in]
        z_inner = [z_top, z_top, z_mid, z_mid, z_bot, z_bot]
        x_outer = [x_in, x_gs_kink + tk, x_out + tk, x_out + tk, x_in]
        z_outer = [
            z_bot - tk,
            z_bot - tk,
            z_mid - tk,
            z_mid - tk,
            z_top + tk,
            z_top + tk,
        ]
        x = np.concatenate([x_inner, x_outer])
        z = np.concatenate([z_inner, z_outer])

        shape = BluemiraFace(make_polygon({"x": x, "y": 0, "z": z}, closed=True))
        self._cryo_vv = shape
        cryostat_vv = PhysicalComponent("Cryostat VV", shape)
        cryostat_vv.plot_options.face_options["color"] = BLUE_PALETTE["CR"][0]
        component = Component("xz", children=[cryostat_vv])
        bm_plot_tools.set_component_plane(component, "xz")
        return component

    def build_xy(self):
        """
        Build the x-y components of the cryostat.
        """
        r_in, _ = self._get_extrema()
        r_out = r_in + self.params.tk_cr_vv
        inner = make_circle(radius=r_in)
        outer = make_circle(radius=r_out)

        shape = BluemiraFace([outer, inner])
        cryostat_vv = PhysicalComponent("Cryostat VV", shape)
        cryostat_vv.plot_options.face_options["color"] = BLUE_PALETTE["CR"][0]
        component = Component("xy", children=[cryostat_vv])
        bm_plot_tools.set_component_plane(component, "xy")
        return component

    def build_xyz(self):
        """
        Build the x-y-z components of the cryostat.
        """
        component = Component("xyz")
        vv_face = self._cts_face.deepcopy()
        base = (0, 0, 0)
        direction = (0, 0, 1)
        vv_face.rotate(base=base, direction=direction, degree=-180 / self.params.n_TF)
        shape = revolve_shape(
            vv_face, base=base, direction=direction, degree=360 / self.params.n_TF
        )

        cryostat_vv = PhysicalComponent("Cryostat TS", shape)
        cryostat_vv.display_cad_options.color = BLUE_PALETTE["CR"][0]
        sectors = circular_pattern_component(cryostat_vv, self._params.n_TF.value)
        component.add_children(sectors, merge_trees=True)

        return component

    def _get_extrema(self):
        bound_box = self._cts.bounding_box
        z_max = bound_box.z_max
        x_max = bound_box.x_max
        x_out = x_max + self.params.g_cr_ts
        z_top = z_max + self.params.g_cr_ts
        return x_out, z_top

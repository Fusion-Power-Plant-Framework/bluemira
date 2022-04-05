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

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.builders.EUDEMO.tools import circular_pattern_component
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    make_circle,
    make_polygon,
    offset_wire,
    revolve_shape,
)


class RadiationShieldBuilder(Builder):
    """
    Builder for the radiation shield
    """

    _required_params: List[str] = [
        "tk_rs",
        "g_cr_rs",
        "o_p_rs",
        "n_rs_lab",
        "rs_l_d",
        "rs_l_gap",
        "n_TF",
    ]
    _params: Configuration
    _cryo_vv_xz: BluemiraFace

    def __init__(
        self,
        params,
        build_config: BuildConfig,
        cryo_vv_xz: BluemiraFace,
    ):
        super().__init__(
            params,
            build_config,
            cryo_vv_xz=cryo_vv_xz,
        )

    def reinitialise(self, params, cryo_vv_xz) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        super().reinitialise(params)
        self._cryo_vv_xz = cryo_vv_xz

    def build(self) -> Component:
        """
        Build the radiation shield component.

        Returns
        -------
        component: Component
            The Component built by this builder.
        """
        super().build()

        component = Component(name=self.name)
        component.add_child(self.build_xz())
        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())
        return component

    def build_xz(self):
        """
        Build the x-z components of the radiation shield.
        """
        cryo_vv = self._cryo_vv_xz
        base = (0, 0, 0)
        direction = (0, 0, 1)
        cryo_vv_rot = cryo_vv.deepcopy()
        cryo_vv_rot.rotate(base, direction, degree=180)
        full_cryo_vv = boolean_fuse([cryo_vv, cryo_vv_rot])
        cryo_vv_outer = full_cryo_vv.boundary[0]
        rs_inner = offset_wire(cryo_vv_outer, self.params.g_cr_rs)
        rs_outer = offset_wire(rs_inner, self.params.tk_rs)

        rs_full = BluemiraFace([rs_outer, rs_inner])
        # Now we slice in half
        bound_box = rs_outer.bounding_box
        x_min = bound_box.x_min - 1.0
        z_min, z_max = bound_box.z_min - 1.0, bound_box.z_max + 1.0
        x = [0, 0, x_min, x_min]
        z = [z_min, z_max, z_max, z_min]
        cutter = BluemiraFace(make_polygon({"x": x, "y": 0, "z": z}, closed=True))
        rs_half = boolean_cut(rs_full, cutter)[0]
        self._rs_face = rs_half

        shield_body = PhysicalComponent("Body", rs_half)
        shield_body.plot_options.face_options["color"] = BLUE_PALETTE["RS"][0]
        component = Component("xz", children=[shield_body])
        bm_plot_tools.set_component_view(component, "xz")
        return component

    def build_xy(self):
        """
        Build the x-y components of the radiation shield.
        """
        x_max = self._cryo_vv_xz.bounding_box.x_max
        r_in = x_max + self.params.g_cr_rs
        r_out = r_in + self.params.tk_rs
        inner = make_circle(radius=r_in)
        outer = make_circle(radius=r_out)

        shape = BluemiraFace([outer, inner])
        shield_body = PhysicalComponent("Body", shape)
        shield_body.plot_options.face_options["color"] = BLUE_PALETTE["RS"][0]
        component = Component("xy", children=[shield_body])
        bm_plot_tools.set_component_view(component, "xy")
        return component

    def build_xyz(self, degree=360.0):
        """
        Build the x-y-z components of the radiation shield.
        """
        n_rs_draw = max(1, int(degree // (360 // self._params.n_TF.value)))
        degree = (360.0 / self._params.n_TF.value) * n_rs_draw

        component = Component("xyz")
        rs_face = self._rs_face.deepcopy()
        base = (0, 0, 0)
        direction = (0, 0, 1)
        shape = revolve_shape(
            rs_face, base=base, direction=direction, degree=360 / self._params.n_TF.value
        )

        rs_body = PhysicalComponent("Body", shape)
        rs_body.display_cad_options.color = BLUE_PALETTE["RS"][0]
        sectors = circular_pattern_component(rs_body, n_rs_draw, degree=degree)
        component.add_children(sectors, merge_trees=True)

        return component

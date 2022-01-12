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
from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
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
        cryo_vv = self._cryo_vv
        base = (0, 0, 0)
        direction = (0, 0, 1)
        cryo_vv_rot = cryo_vv.deepcopy().rotate(base, direction, degree=180)
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
        cutter = make_polygon({"x": x, "y": 0, "z": z}, closed=True)
        rs_half = boolean_cut(rs_full, cutter)[0]
        self._rs_face = rs_half

        shield_body = PhysicalComponent("Body", rs_half)
        shield_body.plot_options.face_options["color"] = BLUE_PALETTE["RS"][0]
        component = Component("xz", children=[shield_body])
        bm_plot_tools.set_component_plane(component, "xz")
        return component

    def build_xy(self):
        """
        Build the x-y components of the radiation shield.
        """
        x_max = self._cryo_vv.bounding_box.x_max
        r_in = x_max + self.params.g_cr_rs
        r_out = r_in + self.params.tk_rs
        inner = make_circle(radius=r_in)
        outer = make_circle(radius=r_out)

        shape = BluemiraFace([outer, inner])
        shield_body = PhysicalComponent("Body", shape)
        shield_body.plot_options.face_options["color"] = BLUE_PALETTE["RS"][0]
        component = Component("xy", children=[shield_body])
        bm_plot_tools.set_component_plane(component, "xy")
        return component

    def build_xyz(self):
        """
        Build the x-y-z components of the radiation shield.
        """
        component = Component("xyz")
        rs_face = self._rs_face.deepcopy()
        base = (0, 0, 0)
        direction = (0, 0, 1)
        rs_face.rotate(base=base, direction=direction, degree=-180 / self.params.n_TF)
        shape = revolve_shape(
            rs_face, base=base, direction=direction, degree=360 / self.params.n_TF
        )

        rs_body = PhysicalComponent("Body", shape)
        rs_body.display_cad_options.color = BLUE_PALETTE["RS"][0]
        sectors = circular_pattern_component(rs_body, self._params.n_TF.value)
        component.add_children(sectors, merge_trees=True)

        return component

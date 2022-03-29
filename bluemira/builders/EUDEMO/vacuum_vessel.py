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
Builder for making a parameterised EU-DEMO vacuum vessel.
"""

from typing import List

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.builders.EUDEMO.tools import circular_pattern_component, varied_offset
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, offset_wire, revolve_shape
from bluemira.geometry.wire import BluemiraWire


class VacuumVesselBuilder(Builder):
    """
    Builder for the vacuum vessel
    """

    _required_params: List[str] = [
        "r_vv_ib_in",
        "r_vv_ob_in",
        "tk_vv_in",
        "tk_vv_out",
        "g_vv_bb",
        "n_TF",
    ]
    _params: Configuration
    _fw_koz: BluemiraWire

    def __init__(
        self,
        params,
        build_config: BuildConfig,
        fw_koz: BluemiraWire,
    ):
        super().__init__(
            params,
            build_config,
            fw_koz=fw_koz,
        )
        self._vv_face = None

    def build(self) -> Component:
        """
        Build the vacuum vessel component.

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
        Build the x-z components of the vacuum vessel.
        """
        inner_vv = offset_wire(
            self._fw_koz, self._params.g_vv_bb.value, join="arc", open_wire=False
        )
        angle_1 = 80
        angle_2 = 160
        outer_vv = varied_offset(
            inner_vv,
            self._params.tk_vv_in.value,
            self._params.tk_vv_out.value,
            angle_1,
            angle_2,
            num_points=300,
        )
        face = BluemiraFace([outer_vv, inner_vv])
        self._vv_face = face

        body = PhysicalComponent("body", face)
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]

        component = Component("xz", children=[body])
        bm_plot_tools.set_component_plane(component, "xz")
        return component

    def build_xy(self):
        """
        Build the x-y components of the vacuum vessel.
        """
        center = (0, 0, 0)
        axis = (0, 0, 1)
        degree = 360
        r_ib_in = self._params.r_vv_ib_in.value
        r_ib_out = r_ib_in + self._params.tk_vv_in.value
        r_ob_in = self._params.r_vv_ob_in.value
        r_ob_out = r_ob_in + self._params.tk_vv_out.value

        ib_inner = make_circle(r_ib_in, center=center, axis=axis, end_angle=degree)
        ib_outer = make_circle(r_ib_out, center=center, axis=axis, end_angle=degree)
        inboard = BluemiraFace([ib_outer, ib_inner])
        vv_inboard = PhysicalComponent("inboard", inboard)
        vv_inboard.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]

        ob_inner = make_circle(r_ob_in, center=center, axis=axis, end_angle=degree)
        ob_outer = make_circle(r_ob_out, center=center, axis=axis, end_angle=degree)
        outboard = BluemiraFace([ob_outer, ob_inner])
        vv_outboard = PhysicalComponent("outboard", outboard)
        vv_outboard.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]

        component = Component("xy", children=[vv_inboard, vv_outboard])
        bm_plot_tools.set_component_plane(component, "xy")
        return component

    def build_xyz(self, degree=360.0) -> Component:
        """
        Build the x-y-z components of the vacuum vessel.
        """
        n_vv_draw = max(1, int(degree // (360 // self._params.n_TF.value)))
        degree = (360.0 / self._params.n_TF.value) * n_vv_draw

        vv_face = self._vv_face.deepcopy()
        base = (0, 0, 0)
        direction = (0, 0, 1)
        vv_body = revolve_shape(
            vv_face,
            base=base,
            direction=direction,
            degree=360 / self._params.n_TF.value,
        )

        vv = PhysicalComponent("Body", vv_body)
        vv.display_cad_options.color = BLUE_PALETTE["VV"][0]

        sectors = circular_pattern_component([vv], n_vv_draw, degree=degree)
        component = Component("xyz")
        component.add_children(sectors, merge_trees=True)

        return component

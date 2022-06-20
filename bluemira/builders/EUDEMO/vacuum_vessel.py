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

from copy import deepcopy
from typing import List

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.builders.EUDEMO.tools import (
    find_xy_plane_radii,
    make_circular_xy_ring,
    varied_offset,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import offset_wire, revolve_shape
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
    _ivc_koz: BluemiraWire

    def __init__(
        self,
        params,
        build_config: BuildConfig,
        ivc_koz: BluemiraWire,
    ):
        super().__init__(
            params,
            build_config,
            ivc_koz=ivc_koz,
        )

    def reinitialise(self, params, ivc_koz) -> None:
        """
        Reinitialise the parameters and boundary.

        Parameters
        ----------
        params: dict
            The new parameter values to initialise this builder against.
        """
        super().reinitialise(params)

        if not ivc_koz.is_closed():
            ivc_koz = deepcopy(ivc_koz)
            ivc_koz.close()
        self._ivc_koz = ivc_koz

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
            self._ivc_koz, self._params.g_vv_bb.value, join="arc", open_wire=False
        )
        # TODO: Calculate these / get them from params
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

        body = PhysicalComponent("Body", face)
        body.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]

        component = Component("xz", children=[body])
        bm_plot_tools.set_component_view(component, "xz")
        return component

    def build_xy(self):
        """
        Build the x-y components of the vacuum vessel.
        """
        xy_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 1, 0])
        r_ib_out, r_ob_out = find_xy_plane_radii(self._vv_face.boundary[0], xy_plane)
        r_ib_in, r_ob_in = find_xy_plane_radii(self._vv_face.boundary[1], xy_plane)

        inboard = make_circular_xy_ring(r_ib_in, r_ib_out)
        vv_inboard = PhysicalComponent("inboard", inboard)
        vv_inboard.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]

        outboard = make_circular_xy_ring(r_ob_in, r_ob_out)
        vv_outboard = PhysicalComponent("outboard", outboard)
        vv_outboard.plot_options.face_options["color"] = BLUE_PALETTE["VV"][0]

        component = Component("xy", children=[vv_inboard, vv_outboard])
        bm_plot_tools.set_component_view(component, "xy")
        return component

    def build_xyz(self, degree=360.0) -> Component:
        """
        Build the x-y-z components of the vacuum vessel.
        """
        vv_face = self._vv_face.deepcopy()
        base = (0, 0, 0)
        direction = (0, 0, 1)
        vv_body = revolve_shape(
            vv_face,
            base=base,
            direction=direction,
            degree=degree
            - 1,  # TODO: Put back `degree/ self._params.n_TF.value,` (#902)
        )

        vv = PhysicalComponent("Body", vv_body)
        vv.display_cad_options.color = BLUE_PALETTE["VV"][0]
        component = Component("xyz", children=[vv])
        # TODO: Put back sector segmentation (see #902 for details)
        # n_vv_draw = max(1, int(degree // (360 // self._params.n_TF.value)))
        # degree = (360.0 / self._params.n_TF.value) * n_vv_draw
        # sectors = circular_pattern_component(vv, n_vv_draw, degree=degree)
        # component = Component("xyz")
        # component.add_children(sectors, merge_trees=True)

        return component

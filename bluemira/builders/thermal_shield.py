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

from typing import List, Optional

import numpy as np
from scipy.spatial import ConvexHull

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.builders.EUDEMO.tools import (
    circular_pattern_component,
    find_xy_plane_radii,
    make_circular_xy_ring,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    make_polygon,
    offset_wire,
    revolve_shape,
)
from bluemira.geometry.wire import BluemiraWire


class VacuumVesselThermalShieldBuilder(Builder):
    """
    Build for the vacuum vessel thermal shield (VVTS).
    """

    _required_params: List[str] = [
        "tk_ts",
        "g_vv_ts",
        "n_TF",
    ]
    _params: Configuration
    _vv_koz: Optional[BluemiraWire]

    def reinitialise(self, params, vv_koz) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        super().reinitialise(params)

        if vv_koz:
            vv_koz.close()

        self._vv_koz = vv_koz

    def build(self) -> Component:
        """
        Build the vacuum vessel thermal shield component.

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
        Build the x-z components of the vacuum vessel thermal shield.
        """
        vvts_inner_wire = offset_wire(
            self._vv_koz,
            self._params.g_vv_ts.value,
            join="intersect",
            open_wire=False,
            ndiscr=600,
        )
        vvts_outer_wire = offset_wire(
            vvts_inner_wire,
            self._params.tk_ts.value,
            join="intersect",
            open_wire=False,
            ndiscr=600,
        )
        vvts_face = BluemiraFace([vvts_outer_wire, vvts_inner_wire])
        self._vvts_face = vvts_face
        vvts = PhysicalComponent("VVTS", vvts_face)

        vvts.plot_options.face_options["color"] = BLUE_PALETTE["TS"][0]

        component = Component("xz", children=[vvts])
        bm_plot_tools.set_component_view(component, "xz")
        return component

    def build_xy(self):
        """
        Build the x-y components of the vacuum vessel thermal shield.
        """
        xy_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 1, 0])
        r_ib_out, r_ob_out = find_xy_plane_radii(self._vvts_face.boundary[0], xy_plane)
        r_ib_in, r_ob_in = find_xy_plane_radii(self._vvts_face.boundary[1], xy_plane)

        inboard = make_circular_xy_ring(r_ib_in, r_ib_out)
        vvts_ib = PhysicalComponent("inboard", inboard)
        vvts_ib.plot_options.face_options["color"] = BLUE_PALETTE["TS"][0]

        outboard = make_circular_xy_ring(r_ob_in, r_ob_out)
        vvts_ob = PhysicalComponent("outboard", outboard)
        vvts_ob.plot_options.face_options["color"] = BLUE_PALETTE["TS"][0]

        component = Component("xy", children=[vvts_ib])
        bm_plot_tools.set_component_view(component, "xy")
        return component

    def build_xyz(self, degree=360.0):
        """
        Build the x-y-z components of the vacuum vessel thermal shield
        """
        n_ts_draw = max(1, int(degree // (360 // self._params.n_TF.value)))
        degree = (360.0 / self._params.n_TF.value) * n_ts_draw

        component = Component("xyz")
        vvts_face = self._vvts_face
        base = (0, 0, 0)
        direction = (0, 0, 1)
        shape = revolve_shape(
            vvts_face,
            base=base,
            direction=direction,
            degree=360 / self._params.n_TF.value,
        )

        vvts_body = PhysicalComponent("VVTS", shape)
        vvts_body.display_cad_options.color = BLUE_PALETTE["TS"][0]
        sectors = circular_pattern_component(vvts_body, n_ts_draw, degree=degree)
        component.add_children(sectors, merge_trees=True)

        return component


class CryostatThermalShieldBuilder(Builder):
    """
    Builder for the thermal shield
    """

    _required_params: List[str] = [
        "tk_ts",
        "g_ts_pf",
        "g_ts_tf",
        "g_vv_ts",
        "n_TF",
    ]
    _params: Configuration
    _pf_kozs: List[BluemiraWire]
    _tf_koz: BluemiraWire
    _cts_face: BluemiraFace

    def __init__(
        self,
        params,
        build_config: BuildConfig,
        pf_coils_xz_kozs: List[BluemiraWire],
        tf_xz_koz: BluemiraWire,
        vv_xz_koz: Optional[BluemiraWire] = None,
    ):
        super().__init__(
            params,
            build_config,
            pf_coils_xz_kozs=pf_coils_xz_kozs,
            tf_xz_koz=tf_xz_koz,
        )
        self._cts_face = None

    def reinitialise(self, params, pf_coils_xz_kozs, tf_xz_koz) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        super().reinitialise(params)
        self._pf_kozs = pf_coils_xz_kozs
        self._tf_koz = tf_xz_koz

    def build(self) -> Component:
        """
        Build the thermal shield component.

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
        Build the x-z components of the thermal shield.
        """
        # Cryostat thermal shield
        pf_xz = self._pf_kozs
        x, z = [], []
        for coil in pf_xz:
            bound_box = coil.bounding_box
            x_corner = [
                bound_box.x_min,
                bound_box.x_max,
                bound_box.x_max,
                bound_box.x_min,
            ]
            z_corner = [
                bound_box.z_min,
                bound_box.z_min,
                bound_box.z_max,
                bound_box.z_max,
            ]
            x.extend(x_corner)
            z.extend(z_corner)

        # Project extrema slightly beyond axis (might be bad for NT) - will get cut later
        x.extend([-0.5, -0.5])  # [m]
        z.extend([np.min(z), np.max(z)])
        x, z = np.array(x), np.array(z)
        hull_idx = ConvexHull(np.array([x, z]).T).vertices
        wire = make_polygon({"x": x[hull_idx], "y": 0, "z": z[hull_idx]}, closed=True)
        wire = offset_wire(wire, self.params.g_ts_pf, open_wire=False, ndiscr=600)
        pf_o_wire = offset_wire(wire, self.params.tk_ts, open_wire=False, ndiscr=600)

        tf_o_wire = offset_wire(
            self._tf_koz,
            self.params.g_ts_tf,
            join="arc",
            open_wire=False,
            ndiscr=600,
        )

        try:
            cts_inner = boolean_fuse(
                [BluemiraFace(pf_o_wire), BluemiraFace(tf_o_wire)]
            ).boundary[0]
        except GeometryError:
            # TODO: boolean_fuse probably shouldn't throw an error here...
            # the TF offset face is probably enclosed by the PF offset face
            cts_inner = pf_o_wire

        cts_outer = offset_wire(cts_inner, self.params.tk_ts, ndiscr=600)
        cts_face = BluemiraFace([cts_outer, cts_inner])
        bound_box = cts_face.bounding_box
        z_min, z_max = bound_box.z_min, bound_box.z_max
        x_in, x_out = 0, -bound_box.x_max
        x = [x_in, x_out, x_out, x_in]
        z = [z_min, z_min, z_max, z_max]
        cutter = BluemiraFace(make_polygon({"x": x, "y": 0, "z": z}, closed=True))

        cts = boolean_cut(cts_face, cutter)[0]
        self._cts_face = cts
        cryostat_ts = PhysicalComponent("Cryostat TS", cts)
        cryostat_ts.plot_options.face_options["color"] = BLUE_PALETTE["TS"][0]

        component = Component("xz", children=[cryostat_ts])
        bm_plot_tools.set_component_view(component, "xz")
        return component

    def build_xy(self):
        """
        Build the x-y components of the thermal shield.
        """
        # Cryostat thermal shield
        mid_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 1, 0])
        r_in, r_out = find_xy_plane_radii(self._cts_face.boundary[0], mid_plane)

        cts = make_circular_xy_ring(r_in, r_out)
        cryostat_ts = PhysicalComponent("Cryostat TS", cts)
        cryostat_ts.plot_options.face_options["color"] = BLUE_PALETTE["TS"][0]

        component = Component("xy", children=[cryostat_ts])
        bm_plot_tools.set_component_view(component, "xy")
        return component

    def build_xyz(self, degree=360.0) -> Component:
        """
        Build the x-y-z components of the thermal shield.
        """
        component = Component("xyz")
        cts_face = self._cts_face.deepcopy()
        base = (0, 0, 0)
        direction = (0, 0, 1)
        cts = revolve_shape(
            cts_face,
            base=base,
            direction=direction,
            degree=degree
            - 1,  # TODO: Put back `degree/ self._params.n_TF.value,` (#902)
        )
        cryostat_ts = PhysicalComponent("Cryostat TS", cts)
        cryostat_ts.display_cad_options.color = BLUE_PALETTE["TS"][0]
        component.add_child(cryostat_ts)
        # TODO: Put back sector segmentation (see #902 for details)
        # n_ts_draw = max(1, int(degree // (360 // self._params.n_TF.value)))
        # degree = (360.0 / self._params.n_TF.value) * n_ts_draw
        # cryostat_ts.display_cad_options.color = BLUE_PALETTE["TS"][0]
        # sectors = circular_pattern_component(cryostat_ts, n_ts_draw, degree=degree)
        # component.add_children(sectors, merge_trees=True)

        return component

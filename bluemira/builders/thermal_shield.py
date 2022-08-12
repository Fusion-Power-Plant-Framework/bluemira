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
Thermal shield builders
"""

from typing import Dict, List, Tuple, Type, Union

import numpy as np
from scipy.spatial import ConvexHull

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.base.parameter_frame import parameter_frame
from bluemira.builders.tools import (
    circular_pattern_component,
    directional_component_tree,
    find_xy_plane_radii,
    get_n_sectors,
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


class VacuumVesselThermalShield:
    """
    VacuumVesselThermalShield Component Manager TODO
    """

    def __init__(self, component: Component):
        super().__init__()
        self._component = component

    def component(self) -> Component:
        """
        Return component
        """
        return self._component


@parameter_frame
class VVTSBuilderParams:
    """
    VVTS builder parameters
    """

    g_vv_ts: Parameter[float]
    n_TF: Parameter[int]
    tk_ts: Parameter[float]


class VVTSDesigner(Designer[BluemiraWire]):
    params_cls = None

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        vv_koz: BluemiraWire,
    ):
        super().__init__(params)
        self.vv_koz = vv_koz

    def run(self):
        """
        Vacuum vessel thermal shield designer run method
        """
        return self.vv_koz


class VVTSBuilder(Builder):
    VVTS = "VVTS"
    param_cls: Type[VVTSBuilderParams] = VVTSBuilderParams

    def build(self) -> VacuumVesselThermalShield:
        """
        Build the vacuum vessel thermal shield component.
        """
        koz = self.designer.run()
        xz_vvts = self.build_xz(koz)
        vvts_face = xz_vvts.get_component_properties("shape")

        component = super().build()

        directional_component_tree(
            component,
            xz=[xz_vvts],
            xy=self.build_xy(vvts_face),
            xyz=self.build_xyz(vvts_face),
        )

        return VacuumVesselThermalShield(component)

    def build_xz(self, koz) -> PhysicalComponent:
        """
        Build the x-z components of the vacuum vessel thermal shield.

        Parameters
        ----------
        koz: BluemiraWire
            keep out zone for the thermal shield
        """
        # This split hack works round #1319
        ex_args = dict(
            join="intersect",
            open_wire=False,
            ndiscr=600,
        )
        vvts_inner_wire = offset_wire(
            offset_wire(koz, self.params.g_vv_ts.value / 2, **ex_args),
            -self.params.g_vv_ts.value / 2,
            **ex_args
        )
        vvts_outer_wire = offset_wire(
            offset_wire(
                koz, (self.params.tk_ts.value + self.params.g_vv_ts.value) / 2, **ex_args
            ),
            -(self.params.tk_ts.value + self.params.g_vv_ts.value) / 2,
            **ex_args
        )

        vvts_face = BluemiraFace([vvts_outer_wire, vvts_inner_wire])
        self.vvts_face = vvts_face

        vvts = PhysicalComponent(self.VVTS, vvts_face)
        vvts.plot_options.face_options["color"] = BLUE_PALETTE["TS"][0]
        return vvts

    def build_xy(self, vvts_face) -> PhysicalComponent:
        """
        Build the x-y components of the vacuum vessel thermal shield.

        Parameters
        ----------
        vvts_face: BluemiraFace
            xz face to build vvts
        """
        xy_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 1, 0])

        r_ib_out, r_ob_out = find_xy_plane_radii(vvts_face.boundary[0], xy_plane)
        r_ib_in, r_ob_in = find_xy_plane_radii(vvts_face.boundary[1], xy_plane)

        sections = []
        for name, r_in, r_out in [
            ["inboard", r_ib_in, r_ib_out],
            ["outboard", r_ob_in, r_ob_out],
        ]:
            board = make_circular_xy_ring(r_in, r_out)
            section = PhysicalComponent(name, board)
            section.plot_options.face_options["color"] = BLUE_PALETTE["TS"][0]
            sections.append(section)

        return sections

    def build_xyz(self, vvts_face, degree=360) -> List[PhysicalComponent]:
        """
        Build the x-y-z components of the vacuum vessel thermal shield

        Parameters
        ----------
        vvts_face: BluemiraFace
            xz face to build vvts
        """
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)

        shape = revolve_shape(
            vvts_face,
            base=(0, 0, 0),
            direction=(0, 0, 1),
            degree=sector_degree,
        )

        vvts_body = PhysicalComponent(self.VVTS, shape)
        vvts_body.display_cad_options.color = BLUE_PALETTE["TS"][0]
        return circular_pattern_component(
            vvts_body, n_sectors, degree=sector_degree * n_sectors
        )


class CryostatThermalShield:
    """
    CryostatThermalShield Component Manager TODO
    """

    def __init__(self, component: Component):
        super().__init__()
        self._component = component

    def component(self) -> Component:
        """
        Return component
        """
        return self._component


@parameter_frame
class CryostatTSBuilderParams:
    """
    Cryostat thermal shield builder parameters
    """

    g_ts_pf: Parameter[float]
    g_ts_tf: Parameter[float]
    n_TF: Parameter[int]
    tk_ts: Parameter[float]


class CryostatTSDesigner(Designer[BluemiraWire]):
    params_cls = None

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        pf_coils_xz_kozs: List[BluemiraWire],
        tf_xz_koz: BluemiraWire,
    ):
        super().__init__(params)
        self.pf_coils_xz_kozs = pf_coils_xz_kozs
        self.tf_xz_koz = tf_xz_koz

    def run(self) -> Tuple[List[BluemiraWire], BluemiraWire]:
        """
        Vacuum vessel thermal shield designer run method
        """
        return self.pf_coils_xz_kozs, self.tf_xz_koz


class CryostatTSBuilder(Builder):
    CRYO_TS = "Cryostat TS"

    param_cls: Type[CryostatTSBuilderParams] = CryostatTSBuilderParams

    def build(self) -> CryostatThermalShield:
        """
        Build the cryostat thermal shield component.
        """
        pf_kozs, tf_koz = self.designer.run()
        xz_cts = self.build_xz(pf_kozs, tf_koz)
        cts_face = xz_cts.get_component_properties("shape")

        component = super().build()

        directional_component_tree(
            component,
            xz=[xz_cts],
            xy=[self.build_xy(cts_face)],
            xyz=self.build_xyz(cts_face),
        )

        return CryostatThermalShield(component)

    def build_xz(
        self, pf_kozs: List[BluemiraWire], tf_koz: BluemiraWire
    ) -> PhysicalComponent:
        """
        Build the x-z components of the thermal shield.
        """
        x, z = [], []
        for coil in pf_kozs:
            bb = coil.bounding_box
            x.extend([bb.x_min, bb.x_max, bb.x_max, bb.x_min])
            z.extend([bb.z_min, bb.z_min, bb.z_max, bb.z_max])

        # Project extrema slightly beyond axis (might be bad for NT) - will get cut later
        x.extend([-0.5, -0.5])  # [m]
        z.extend([np.min(z), np.max(z)])
        xz = np.array(x), np.array(z)
        hull_idx = ConvexHull(np.array([x, z]).T).vertices

        wire = make_polygon({"x": x[hull_idx], "y": 0, "z": z[hull_idx]}, closed=True)
        wire = offset_wire(wire, self.params.g_ts_pf, open_wire=False, ndiscr=600)
        pf_o_wire = offset_wire(wire, self.params.tk_ts, open_wire=False, ndiscr=600)

        pf_o_wire2 = offset_wire(
            make_polygon({"x": x[hull_idx], "y": 0, "z": z[hull_idx]}, closed=True),
            self.params.g_ts_pf + self.params.tk_ts,
            open_wire=False,
            ndiscr=600,
        )
        # TODO: is there any differnece between pf_o_wire and pf_o_wire2?
        # import ipdb
        # ipdb.set_trace()
        tf_o_wire = offset_wire(
            tf_koz,
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
        bb = cts_face.bounding_box
        x_in, x_out = 0, -bb.x_max
        cutter = BluemiraFace(
            make_polygon(
                {
                    "x": [x_in, x_out, x_out, x_in],
                    "y": 0,
                    "z": [bb.z_min, bb.z_min, bb.z_max, bb.z_max],
                },
                closed=True,
            )
        )

        cts = boolean_cut(cts_face, cutter)[0]
        cryostat_ts = PhysicalComponent(self.CRYO_TS, cts)
        cryostat_ts.plot_options.face_options["color"] = BLUE_PALETTE["TS"][0]
        return cryostat_ts

    def build_xy(self, cts_face) -> PhysicalComponent:
        """
        Build the x-y components of the thermal shield.
        """
        mid_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 1, 0])
        r_in, r_out = find_xy_plane_radii(cts_face.boundary[0], mid_plane)

        cryostat_ts = PhysicalComponent(self.CRYO_TS, make_circular_xy_ring(r_in, r_out))
        cryostat_ts.plot_options.face_options["color"] = BLUE_PALETTE["TS"][0]

        return cryostat_ts

    def build_xyz(self, cts_face, degree=360) -> List[PhysicalComponent]:
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)

        shape = revolve_shape(
            cts_face,
            base=(0, 0, 0),
            direction=(0, 0, 1),
            degree=sector_degree,
        )
        cryostat_ts = PhysicalComponent(self.CRYO_TS, shape)
        cryostat_ts.display_cad_options.color = BLUE_PALETTE["TS"][0]
        return circular_pattern_component(
            cryostat_ts, n_sectors, degree=sector_degree * n_sectors
        )

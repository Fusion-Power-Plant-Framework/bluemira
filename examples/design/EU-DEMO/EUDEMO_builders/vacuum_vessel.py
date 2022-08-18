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
from typing import Dict, List, Type, Union

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import parameter_frame
from bluemira.builders.tools import (  # circular_pattern_component,
    find_xy_plane_radii,
    get_n_sectors,
    make_circular_xy_ring,
    varied_offset,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import offset_wire, revolve_shape
from bluemira.geometry.wire import BluemiraWire


class VacuumVessel(ComponentManager):
    """
    Wrapper around a Vacuum Vessel component tree.
    """


@parameter_frame
class VacuumVesselBuilderParams:
    """
    Vacuum Vessel builder parameters
    """

    n_TF: Parameter[int]
    r_vv_ib_in: Parameter[float]
    r_vv_ob_in: Parameter[float]
    tk_vv_in: Parameter[float]
    tk_vv_out: Parameter[float]
    g_vv_bb: Parameter[float]


class VacuumVesselDesigner(Designer[BluemiraWire]):
    """
    Vacuum Vessel designer
    """

    param_cls = None

    def __init__(
        self,
        ivc_koz: BluemiraWire,
    ):
        super().__init__()
        self.ivc_koz = ivc_koz

    def run(self):
        """
        Vacuum Vessel designer run method
        """
        return self.ivc_koz


class VacuumVesselBuilder(Builder):
    """
    Vacuum Vessel builder
    """

    VV = "VV"
    BODY = "Body"
    param_cls: Type[VacuumVesselBuilderParams] = VacuumVesselBuilderParams

    def __init__(
        self,
        params: Union[VacuumVesselBuilderParams, Dict],
        build_config: Dict,
        designer: Designer[BluemiraFace],
    ):
        super().__init__(params, build_config, designer)

    def build(self) -> VacuumVessel:
        """
        Build the vacuum vessel component.
        """
        ivc_koz = self.designer.run()
        xz_vv = self.build_xz(ivc_koz)
        vv_face = xz_vv.get_component_properties("shape")

        return VacuumVessel(
            self.component_tree(
                xz=[xz_vv],
                xy=self.build_xy(vv_face),
                xyz=self.build_xyz(vv_face),
            )
        )

    def build_xz(
        self,
        ivc_koz: BluemiraWire,
        inboard_offset_degree: float = 80,
        outboard_offset_degree: float = 160,
    ) -> PhysicalComponent:
        """
        Build the x-z components of the vacuum vessel.
        """
        inner_vv = offset_wire(
            ivc_koz, self.params.g_vv_bb.value, join="arc", open_wire=False
        )
        outer_vv = varied_offset(
            inner_vv,
            self.params.tk_vv_in.value,
            self.params.tk_vv_out.value,
            # TODO: Calculate these / get them from params
            inboard_offset_degree,
            outboard_offset_degree,
            num_points=300,
        )
        face = BluemiraFace([outer_vv, inner_vv])

        body = PhysicalComponent(self.BODY, face)
        body.plot_options.face_options["color"] = BLUE_PALETTE[self.VV][0]

        return body

    def build_xy(self, vv_face: BluemiraFace) -> List[PhysicalComponent]:
        """
        Build the x-y components of the vacuum vessel.
        """
        xy_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 1, 0])

        r_ib_out, r_ob_out = find_xy_plane_radii(vv_face.boundary[0], xy_plane)
        r_ib_in, r_ob_in = find_xy_plane_radii(vv_face.boundary[1], xy_plane)

        sections = []
        for name, r_in, r_out in [
            ["inboard", r_ib_in, r_ib_out],
            ["outboard", r_ob_in, r_ob_out],
        ]:
            board = make_circular_xy_ring(r_in, r_out)
            section = PhysicalComponent(name, board)
            section.plot_options.face_options["color"] = BLUE_PALETTE[self.VV][0]
            sections.append(section)

        return sections

    def build_xyz(
        self, vv_face: BluemiraFace, degree: float = 360.0
    ) -> PhysicalComponent:
        """
        Build the x-y-z components of the vacuum vessel.
        """
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)

        shape = revolve_shape(
            vv_face,
            base=(0, 0, 0),
            direction=(0, 0, 1),
            degree=degree - 1
            # degree=sector_degree,
        )

        vv_body = PhysicalComponent(self.BODY, shape)
        vv_body.display_cad_options.color = BLUE_PALETTE[self.VV][0]
        return [vv_body]
        # this is currently broken because of #1319 and related Topological naming issues
        # return circular_pattern_component(
        #     vv_body, n_sectors, degree=sector_degree * n_sectors
        # )

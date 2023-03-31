# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
EU-DEMO Lower Port
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Union

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire


@dataclass
class LowerPortDesignerParams(ParameterFrame):
    """LowerPort ParameterFrame"""

    lower_port_angle: Parameter[float]
    divertor_padding: Parameter[float]


def intersection_points(
    wire_a: BluemiraGeo,
    wire_b: BluemiraGeo,
) -> List[Tuple]:
    dist, vects, _ = wire_a.shape.distToShape(wire_b.shape)
    if dist > 0.0001:  # not intersecting
        return []
    pois = []
    for vect_pair in vects:
        v = vect_pair[0]
        pois.append((v.x, v.y, v.z))
    return pois


class LowerPortDesigner(Designer):
    """
    Lower Port Designer

    Notes
    -----
    Retrictions on angle is ±45° because there are no protections
    keeping the port dimensions constant

    """

    param_cls: Type[ParameterFrame] = LowerPortDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        divertor_xz: BluemiraFace,
        tf_coil_xz_boundary: BluemiraWire,
        boundary_pullout_factor: float,
        straight_leg_extension_extent: float,
        z_duct: float,
    ):
        super().__init__(params, build_config)
        self.divertor_face = divertor_xz
        self.tf_coil_xz_boundary = tf_coil_xz_boundary
        self.boundary_pullout_extent = boundary_pullout_factor
        self.straight_leg_extension_extent = straight_leg_extension_extent
        self.z_duct = z_duct

    def run(self) -> Tuple[BluemiraFace, BluemiraWire]:
        """Run method of Designer"""
        if not -90 <= self.params.lower_port_angle.value <= 0:
            raise ValueError(
                f"{self.params.lower_port_angle.name}"
                " must be within the range [-90, 0] degrees:"
                f" {self.params.lower_port_angle.value}"
            )

        div_z_top = self.divertor_face.bounding_box.z_max
        div_z_bot = self.divertor_face.bounding_box.z_min

        div_x_inner = self.divertor_face.bounding_box.x_min
        div_x_outer = self.divertor_face.bounding_box.x_max

        grad = np.tan(np.deg2rad(self.params.lower_port_angle.value))  # z/x
        r_search = 20

        # bottom inner
        bi_intc_search_wire = LowerPortDesigner._make_distance_xz_wire(
            (div_x_inner, div_z_bot),
            grad,
            r_search,
        )
        bi_intc_point = intersection_points(
            self.tf_coil_xz_boundary,
            bi_intc_search_wire,
        )
        bi_intc_point = bi_intc_point[0]

        # top outer
        to_intc_search_wire = LowerPortDesigner._make_distance_xz_wire(
            (div_x_outer, div_z_top),
            grad,
            r_search,
        )
        to_intc_point = intersection_points(
            self.tf_coil_xz_boundary,
            to_intc_search_wire,
        )[0]

        # decide which to align to
        delta_z_for_move_in_x = (to_intc_point[0] - bi_intc_point[0]) * grad
        delta_x_for_move_in_z = (to_intc_point[2] - bi_intc_point[2]) / grad
        if abs(delta_z_for_move_in_x) <= abs(delta_x_for_move_in_z):
            # align vertically (equal x's)
            # set x's equal to top outer (always larger x)
            bi_join_point = (
                to_intc_point[0],
                0,
                bi_intc_point[2] + delta_z_for_move_in_x,
            )
            to_join_point = to_intc_point
        else:
            # align horizontally (equal z's)
            bi_join_point = bi_intc_point
            to_join_point = (
                to_intc_point[0] + delta_x_for_move_in_z,
                0,
                bi_intc_point[2],
            )

        koz_lower_duct = make_polygon(
            {
                "x": np.array(
                    [
                        div_x_inner,
                        div_x_outer,
                        to_join_point[0],
                        bi_join_point[0],
                    ]
                ),
                "y": 0,
                "z": np.array(
                    [
                        div_z_bot,
                        div_z_top,
                        to_join_point[2],
                        bi_join_point[2],
                    ]
                ),
            },
            closed=True,
        )
        koz_lower_duct = BluemiraFace(koz_lower_duct)

        return None

    @staticmethod
    def _make_distance_xz_wire(
        starting_xz_point: Tuple,
        gradient: float,
        distance: float,
    ) -> BluemiraWire:
        s_x = starting_xz_point[0]
        s_z = starting_xz_point[1]
        sqrt_value = np.sqrt(distance**2 / (1 + gradient**2))
        f_x = sqrt_value + s_x
        f_z = gradient * sqrt_value + s_z
        return make_polygon(
            {
                "x": np.array([s_x, f_x]),
                "y": 0,
                "z": np.array([s_z, f_z]),
            },
        )

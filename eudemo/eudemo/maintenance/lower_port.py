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
from typing import Dict, Tuple, Type, Union

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon, offset_wire
from bluemira.geometry.wire import BluemiraWire


@dataclass
class LowerPortDesignerParams(ParameterFrame):
    """LowerPort ParameterFrame"""

    lower_port_angle: Parameter[float]
    divertor_padding: Parameter[float]


class LowerPortDesigner(Designer):
    """Lower Port Designer"""

    param_cls: Type[ParameterFrame] = LowerPortDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        divertor_xz: BluemiraFace,
        x_wall_inner: float,
        x_wall_outer: float,
        x_extrema: float,
    ):
        super().__init__(params, build_config)
        self.divertor_xz = divertor_xz
        self.x_wall_inner = x_wall_inner
        self.x_wall_outer = x_wall_outer
        self.x_extrema = x_extrema

    def run(self) -> Tuple[BluemiraFace, BluemiraWire]:
        """Run method of Designer"""
        if not -90 < self.params.lower_port_angle.value < 90:
            raise ValueError(
                f"{self.params.lower_port_angle.name}"
                " must be within the range -90 < x < 90 degrees:"
                f" {self.params.lower_port_angle.value}"
            )

        z_start = self.divertor_xz.bounding_box.z_min
        z_outer = z_start + (self.x_wall_inner - self.x_wall_outer) * np.tan(
            np.deg2rad(self.params.lower_port_angle.value)
        )
        x_start = self.divertor_xz.center_of_mass[0]

        x_path = np.array(
            [x_start, self.x_wall_inner, self.x_wall_outer, self.x_extrema]
        )
        z_lower_path = np.array([z_start, z_start, z_outer, z_outer])

        traj = make_polygon(
            Coordinates(
                {
                    "x": x_path,
                    "z": z_lower_path - self.params.divertor_padding.value,
                }
            )
        )

        port = make_polygon(
            Coordinates(
                {
                    "x": np.append(x_path, x_path[::-1]),
                    "z": np.append(
                        z_lower_path,
                        (z_lower_path + self.divertor_xz.bounding_box.z_max)[::-1],
                    ),
                }
            ),
            closed=True,
        )

        return BluemiraFace(offset_wire(port, self.params.divertor_padding.value)), traj

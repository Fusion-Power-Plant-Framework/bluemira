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
from typing import Dict, Union

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.face import BluemiraFace


@dataclass
class LowerPortDesignerParams(ParameterFrame):
    lower_port_angle: Parameter[float]


class LowerPortDesigner(Designer):
    """Lower Port Designer"""

    param_cls = LowerPortDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        divertor_xz: BluemiraFace,
        x_extrema: float = 10,
    ):
        super().__init__(params, build_config)
        self.divertor_xz = divertor_xz
        self.x_extrema = x_extrema

    def run(self):
        """Run method of Designer"""

        # TODO return port koz

        # Task 1 create trajectory
        # step 1, what angle is the divertor taken out at

        # step 2, trace path through reactor
        # step 3, return to horizontal
        #         (at what level or just immediately outside reactor)
        # x_start = com_divertor
        # z_start = z_min@x_com

        z_outer = z_inner + (x_inner - x_outer) * np.sin(self.params.lower_port_angle)

        traj = np.array(
            [[x_start, x_inner, x_outer, x_stop], [z_start, z_inner, z_outer, z_fin]]
        )

        # Task 2 size of port along trajectory

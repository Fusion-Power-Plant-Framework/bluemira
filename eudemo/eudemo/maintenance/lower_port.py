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
from typing import Dict, Union

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.geometry.face import BluemiraFace


class LowerPortDesigner(Designer):
    """Lower Port Designer"""

    param_cls = None

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        divertor_xz: BluemiraFace,
    ):
        super().__init__(params, build_config)
        self.divertor_xz = divertor_xz

    def run(self):
        """Run method of Designer"""

        # TODO return port koz

        # Task 1 create trajectory
        # step 1, what angle is the divertor taken out at
        # step 2, trace path through reactor
        # step 3, return to horizontal
        #         (at what level or just immediately outside reactor)

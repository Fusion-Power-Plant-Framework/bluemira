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
IVC Boundary Designer
"""
from dataclasses import dataclass
from typing import Dict, Union

from bluemira.base.designer import Designer
from bluemira.base.error import DesignError
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import varied_offset
from bluemira.geometry.wire import BluemiraWire


@dataclass
class IVCBoundaryParams(ParameterFrame):
    """Parameters for running the `WallSolver`."""

    tk_bb_ib: Parameter[float]  # Inboard blanket thickness
    tk_bb_ob: Parameter[float]  # Outboard blanket thickness
    ib_offset_angle: Parameter[float]  # 45 degrees
    ob_offset_angle: Parameter[float]  # 175 degrees


class IVCBoundaryDesigner(Designer[BluemiraWire]):
    """
    Designs the IVC Boundary ie the Vacuum Vessel keep out zone

    Parameters
    ----------
    params:
        IVC Boundary designer parameters
    wall_shape:
        Wall shape as defined by the wall silhouette designer

    """

    param_cls = IVCBoundaryParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        wall_shape: BluemiraWire,
    ):
        super().__init__(params)

        if not wall_shape.is_closed():
            raise DesignError("Wall shape must be closed.")
        self.wall_shape = wall_shape

    def run(self) -> BluemiraWire:
        """
        Run the IVC Boundary designer
        """
        return varied_offset(
            self.wall_shape,
            self.params.tk_bb_ib.value,
            self.params.tk_bb_ob.value,
            self.params.ib_offset_angle.value,
            self.params.ob_offset_angle.value,
        )

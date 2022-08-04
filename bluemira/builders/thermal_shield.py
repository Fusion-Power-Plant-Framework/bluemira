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

from typing import Dict, List, Type, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter import Parameter, ParameterFrame
from bluemira.display.palettes import BLUE_PALETTE

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


class VVTSDesignerParams(ParameterFrame):


class VVTSBuilderParams(ParameterFrame):
    """
    VVTS builder parameters
    """

    n_TF: Parameter[int]


class VVTSDesigner(Designer[BluemiraWire]):

class VVTSBuilder(Builder):
    def build(self) -> VacuumVesselThermalShield:

    def build_xz(self) -> PhysicalComponent:

    def build_xy(self) -> PhysicalComponent:

    def build_xyz(self, degree=360) -> List[PhysicalComponent]:


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

class CryostatTSDesignerParams(ParameterFrame):

class CryostatTSBuilderParams(ParameterFrame):

class CryostatTSDesigner(Designer[BluemiraWire]):

class CryostatTSBuilder(Builder):
    def build(self) -> CryostatThermalShield:

    def build_xz(self) -> PhysicalComponent:

    def build_xy(self) -> PhysicalComponent:

    def build_xyz(self, degree=360) -> List[PhysicalComponent]:

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
Builder for making a parameterised EU-DEMO divertor.
"""
from dataclasses import dataclass
from typing import Dict, List, Type, Union

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import (
    apply_component_display_options,
    circular_pattern_component,
    get_n_sectors,
    pattern_revolved_silhouette,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace


@dataclass
class DivertorBuilderParams(ParameterFrame):
    """
    Divertor builder parameters
    """

    n_TF: Parameter[int]
    n_div_cassettes: Parameter[int]
    c_rm: Parameter[float]


class DivertorBuilder(Builder):
    """
    Divertor builder
    """

    DIV = "DIV"
    BODY = "Body"
    CASETTES = "cassettes"
    SEGMENT_PREFIX = "segment"
    param_cls: Type[DivertorBuilderParams] = DivertorBuilderParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
        divertor_silhouette: BluemiraFace,
    ):
        super().__init__(params, build_config)
        self.div_koz = divertor_silhouette

    def build(self) -> Component:
        """
        Build the divertor component.
        """
        return self.component_tree(
            xz=[self.build_xz()],
            xy=[],
            xyz=self.build_xyz(degree=0),
        )

    def build_xz(self) -> PhysicalComponent:
        """
        Build the x-z components of the divertor.
        """
        body = PhysicalComponent(self.BODY, self.div_koz)
        apply_component_display_options(body, color=BLUE_PALETTE[self.DIV][0])

        return body

    def build_xyz(self, degree: float = 360.0) -> List[PhysicalComponent]:
        """
        Build the x-y-z components of the divertor.
        """
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)
        shapes = pattern_revolved_silhouette(
            self.div_koz,
            self.params.n_div_cassettes.value,
            self.params.n_TF.value,
            self.params.c_rm.value,
        )

        segments = []
        for no, shape in enumerate(shapes):
            segment = PhysicalComponent(f"{self.SEGMENT_PREFIX}_{no}", shape)
            apply_component_display_options(segment, BLUE_PALETTE[self.DIV][no])
            segments.append(segment)

        return circular_pattern_component(
            Component(self.CASETTES, children=segments),
            n_sectors,
            degree=sector_degree * n_sectors,
        )

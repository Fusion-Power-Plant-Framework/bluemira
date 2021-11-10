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
Built-in build steps for making a parameterised plasma
"""

from typing import Dict, List, Tuple, Type, Union

from bluemira.base import PhysicalComponent
from bluemira.base.components import Component
import bluemira.geometry as geo
from bluemira.geometry.parameterisations import GeometryParameterisation

from bluemira.builders.shapes import ParameterisedShapeBuilder


class MakeParameterisedPlasma(ParameterisedShapeBuilder):
    """
    A class that builds a plasma based on a parameterised shape
    """

    _required_config = ParameterisedShapeBuilder._required_params + [
        "targets",
        "segment_angle",
    ]

    _param_class: Type[GeometryParameterisation]
    _variables_map: Dict[str, str]
    _targets: Dict[str, str]

    def _extract_config(self, build_config: Dict[str, Union[float, int, str]]):
        super()._extract_config(build_config)

        self._targets = build_config["targets"]
        self._segment_angle: float = build_config["segment_angle"]

    def build(self, params, **kwargs) -> List[Tuple[str, Component]]:
        """
        Build a plasma using the requested targets and methods.
        """
        super().build(params, **kwargs)

        boundary = self._shape_builder.build(params)[0][1].shape

        result_components = []
        for target, func in self._targets.items():
            result_components.append(getattr(self, func)(boundary, target))

        return result_components

    def build_xz(self, boundary: geo.wire.BluemiraWire, target: str):
        label = target.split("/")[-1]
        return (
            target,
            PhysicalComponent(label, geo.face.BluemiraFace(boundary, label)),
        )

    def build_xy(self, boundary: geo.wire.BluemiraWire, target: str):
        label = target.split("/")[-1]

        inner = geo.tools.make_circle(boundary.bounding_box[0], axis=[0, 1, 0])
        outer = geo.tools.make_circle(boundary.bounding_box[3], axis=[0, 1, 0])

        return (
            target,
            PhysicalComponent(label, geo.face.BluemiraFace([outer, inner], label)),
        )

    def build_xyz(self, boundary: geo.wire.BluemiraWire, target: str):
        label = target.split("/")[-1]

        shell = geo.tools.revolve_shape(
            boundary, direction=(0, 0, 1), degree=self._segment_angle
        )

        return (target, PhysicalComponent(label, shell))

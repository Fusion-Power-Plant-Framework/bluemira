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

from typing import Any, Dict, List, Tuple, Type

from bluemira.base import PhysicalComponent
from bluemira.base.components import Component
import bluemira.geometry as geo
from bluemira.geometry.parameterisations import GeometryParameterisation

from bluemira.builders.shapes import MakeParameterisedShape


class MakeParameterisedPlasma(MakeParameterisedShape):
    """
    A class that builds a plasma based on a parameterised shape
    """

    _required_config = ["param_class", "variables_map", "target", "segment_angle"]

    _param_class: Type[GeometryParameterisation]
    _variables_map: Dict[str, str]
    _target: str

    def __init__(self, params, build_config: Dict[str, Any], **kwargs):
        super().__init__(params, build_config, **kwargs)

        self._label = self._target.split("/")[-1]

    def _extract_config(self, build_config: Dict[str, Any]):
        super()._extract_config(build_config)

        self._segment_angle: float = build_config["segment_angle"]

    def build(self, params, **kwargs) -> List[Tuple[str, Component]]:
        """
        Build a plasma using the provided parameterisation in the xz, xy and xyz
        dimensions.
        """
        shape_component: PhysicalComponent = super().build(params, **kwargs)[0][1]
        boundary = shape_component.shape

        result_components = []
        result_components.append(self._build_xz(boundary))
        result_components.append(self._build_xy(boundary))
        result_components.append(self._build_xyz(boundary))

        return result_components

    def _build_xz(self, boundary: geo.wire.BluemiraWire):
        return (
            f"xz/{self._target}",
            PhysicalComponent(self._label, geo.face.BluemiraFace(boundary, self._label)),
        )

    def _build_xy(self, boundary: geo.wire.BluemiraWire):
        inner = geo.tools.make_circle(boundary.bounding_box[0], axis=[0, 1, 0])
        outer = geo.tools.make_circle(boundary.bounding_box[3], axis=[0, 1, 0])

        return (
            f"xy/{self._target}",
            PhysicalComponent(
                self._label, geo.face.BluemiraFace([outer, inner], self._label)
            ),
        )

    def _build_xyz(self, boundary: geo.wire.BluemiraWire):
        shell = geo.tools.revolve_shape(
            boundary, direction=(0, 0, 1), degree=self._segment_angle
        )

        return (f"xyz/{self._target}", PhysicalComponent(self._label, shell))

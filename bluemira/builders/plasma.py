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

from typing import Dict, List, Optional, Type

from bluemira.base.builder import BuildResult, BuildConfig
from bluemira.base.components import PhysicalComponent
import bluemira.geometry as geo
from bluemira.geometry.parameterisations import GeometryParameterisation

from bluemira.builders.shapes import ParameterisedShapeBuilder


class MakeParameterisedPlasma(ParameterisedShapeBuilder):
    """
    A class that builds a plasma based on a parameterised shape
    """

    _required_config = ParameterisedShapeBuilder._required_params + [
        "targets",
    ]

    _param_class: Type[GeometryParameterisation]
    _variables_map: Dict[str, str]
    _targets: Dict[str, str]
    _segment_angle: float

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        self._targets = build_config["targets"]
        self._segment_angle = build_config.get("segment_angle", 360.0)

    def build(self, **kwargs) -> List[BuildResult]:
        """
        Build a plasma with a boundary shape defined by the parameterisation.

        Constructs a Component at each of the target paths using the defined methods.
        """
        super().build(**kwargs)

        result_components = []
        for target, func in self._targets.items():
            result_components.append(
                getattr(self, func)(self._shape.create_shape(), target, **kwargs)
            )

        return result_components

    def build_xz(self, boundary: geo.wire.BluemiraWire, target: str) -> BuildResult:
        """
        Build a PhysicalComponent with a BluemiraFace using the provided plasma boundary
        in the xz plane.

        Parameters
        ----------
        boundary: BluemiraWire
            The plasma boundary.
        target: str
            The target path in which to insert the component on output.

        Returns
        -------
        result: BuildResult
            The resulting target path and component.
        """
        label = target.split("/")[-1]
        return BuildResult(
            target=target,
            component=PhysicalComponent(label, geo.face.BluemiraFace(boundary, label)),
        )

    def build_xy(self, boundary: geo.wire.BluemiraWire, target: str) -> BuildResult:
        """
        Build a PhysicalComponent with a BluemiraFace using the provided plasma boundary
        in the xy plane.

        The projection onto the xy plane is taken as the ring bound by the maximum and
        minimum values of the boundary in the radial direction.

        Parameters
        ----------
        boundary: BluemiraWire
            The plasma boundary.
        target: str
            The target path in which to insert the component on output.


        Returns
        -------
        result: BuildResult
            The resulting target path and component.
        """
        label = target.split("/")[-1]

        inner = geo.tools.make_circle(boundary.bounding_box.x_min, axis=[0, 1, 0])
        outer = geo.tools.make_circle(boundary.bounding_box.x_max, axis=[0, 1, 0])

        return BuildResult(
            target=target,
            component=PhysicalComponent(
                label, geo.face.BluemiraFace([outer, inner], label)
            ),
        )

    def build_xyz(
        self,
        boundary: geo.wire.BluemiraWire,
        target: str,
        *,
        segment_angle: Optional[float] = None,
    ) -> BuildResult:
        """
        Build a PhysicalComponent with a BluemiraShell using the provided plasma boundary
        in 3D.

        The 3D shell is created by revolving the boundary through the provided segment
        angle.

        Parameters
        ----------
        boundary: BluemiraWire
            The plasma boundary.
        target: str
            The target path in which to insert the component on output.
        segment_angle: float
            The around which to revolve the 3D geometry.

        Returns
        -------
        result: BuildResult
            The resulting target path and component.
        """
        label = target.split("/")[-1]

        if segment_angle is None:
            segment_angle = self._segment_angle

        shell = geo.tools.revolve_shape(
            boundary, direction=(0, 0, 1), degree=segment_angle
        )

        return BuildResult(target=target, component=PhysicalComponent(label, shell))

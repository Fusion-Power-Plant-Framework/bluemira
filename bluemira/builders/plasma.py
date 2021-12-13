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

from typing import Dict, Optional, Type

from bluemira.base.builder import BuildConfig
from bluemira.base.components import Component, PhysicalComponent
import bluemira.geometry as geo
from bluemira.geometry.parameterisations import GeometryParameterisation

from bluemira.builders.shapes import ParameterisedShapeBuilder
from bluemira.geometry.wire import BluemiraWire
from bluemira.display.palettes import BLUE_PALETTE


class MakeParameterisedPlasma(ParameterisedShapeBuilder):
    """
    A class that builds a plasma based on a parameterised shape
    """

    _param_class: Type[GeometryParameterisation]
    _variables_map: Dict[str, str]
    _segment_angle: float
    _boundary: BluemiraWire

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        self._segment_angle = build_config.get("segment_angle", 360.0)

    def reinitialise(self, params, **kwargs):
        """
        Create the GeometryParameterisation from the provided param_class and
        variables_map and extract the resulting shape as the plasma boundary.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        """
        super().reinitialise(params, **kwargs)

        self._boundary = self._shape.create_shape()

    def build(self, label: str = "LCFS", **kwargs):
        """
        Build a plasma with a boundary shape defined by the parameterisation.
        """
        component = super().build(**kwargs)

        component.add_child(self.build_xz(label=label))
        component.add_child(self.build_xy(label=label))
        component.add_child(self.build_xyz(label=label))

        return component

    def build_xz(self, *, label: str = "LCFS", **kwargs) -> Component:
        """
        Build a PhysicalComponent with a BluemiraFace using the provided plasma boundary
        in the xz plane.

        Parameters
        ----------
        label: str
            The label to apply to the resulting component, by default "LCFS".

        Returns
        -------
        result: Component
            The resulting component representing the xz view of the LCFS.
        """
        face = geo.face.BluemiraFace(self._boundary, label)
        component = PhysicalComponent("LCFS", face)
        component.plot_options.face_options["color"] = BLUE_PALETTE["PL"]

        return Component("xz").add_child(component)

    def build_xy(self, *, label: str = "LCFS", **kwargs) -> Component:
        """
        Build a PhysicalComponent with a BluemiraFace using the plasma boundary in the
        xy plane.

        The projection onto the xy plane is taken as the ring bound by the maximum and
        minimum values of the boundary in the radial direction.

        Parameters
        ----------
        label: str
            The label to apply to the resulting component, by default "LCFS".

        Returns
        -------
        result: PhysicalComponent
            The resulting component representing the xy view of the LCFS.
        """
        inner = geo.tools.make_circle(self._boundary.bounding_box.x_min, axis=[0, 1, 0])
        outer = geo.tools.make_circle(self._boundary.bounding_box.x_max, axis=[0, 1, 0])

        face = geo.face.BluemiraFace([outer, inner], label)
        component = PhysicalComponent(label, face)
        component.plot_options.face_options["color"] = BLUE_PALETTE["PL"]

        return Component("xy").add_child(component)

    def build_xyz(
        self,
        *,
        label: str = "LCFS",
        segment_angle: Optional[float] = None,
        **kwargs,
    ) -> PhysicalComponent:
        """
        Build a PhysicalComponent with a BluemiraShell using the plasma boundary in 3D.

        The 3D shell is created by revolving the boundary through the provided segment
        angle.

        Parameters
        ----------
        label: str
            The label to apply to the resulting component, by default "LCFS".
        segment_angle: float
            The angle [°] around which to revolve the 3D geometry.

        Returns
        -------
        result: PhysicalComponent
            The resulting component.
        """
        if segment_angle is None:
            segment_angle = self._segment_angle

        shell = geo.tools.revolve_shape(
            self._boundary, direction=(0, 0, 1), degree=segment_angle
        )
        component = PhysicalComponent(label, shell)
        component.display_cad_options.color = BLUE_PALETTE["PL"]

        return Component("xyz").add_child(component)

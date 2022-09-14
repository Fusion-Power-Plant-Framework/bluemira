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

from typing import Dict, Type

from bluemira.base.builder import BuildConfig
from bluemira.base.components import Component, PhysicalComponent
from bluemira.builders.shapes import ParameterisedShapeBuilder
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import make_circle, make_face, revolve_shape
from bluemira.geometry.wire import BluemiraWire


class MakeParameterisedPlasma(ParameterisedShapeBuilder):
    """
    A class that builds a plasma based on a parameterised shape
    """

    _param_class: Type[GeometryParameterisation]
    _variables_map: Dict[str, str]
    _label: str
    _boundary: BluemiraWire

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        self._label = build_config.get("label", "LCFS")

    def reinitialise(self, params):
        """
        Create the GeometryParameterisation from the provided param_class and
        variables_map and extract the resulting shape as the plasma boundary.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        """
        super().reinitialise(params)

        self._boundary = self._shape.create_shape()

    def build(self):
        """
        Build a plasma with a boundary shape defined by the parameterisation.
        """
        component = super().build()

        component.add_child(self.build_xz())
        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())

        return component

    def build_xz(self) -> Component:
        """
        Build a PhysicalComponent with a BluemiraFace using the provided plasma boundary
        in the xz plane.

        Returns
        -------
        result: Component
            The resulting component representing the xz view of the Plasma.
        """
        face = make_face(self._boundary, self._label)
        component = PhysicalComponent(self._label, face)
        component.plot_options.face_options["color"] = BLUE_PALETTE["PL"]

        return Component("xz").add_child(component)

    def build_xy(self) -> Component:
        """
        Build a PhysicalComponent with a BluemiraFace using the plasma boundary in the
        xy plane.

        The projection onto the xy plane is taken as the ring bound by the maximum and
        minimum values of the boundary in the radial direction.

        Returns
        -------
        result: PhysicalComponent
            The resulting component representing the xy view of the Plasma.
        """
        inner = make_circle(self._boundary.bounding_box.x_min, axis=[0, 1, 0])
        outer = make_circle(self._boundary.bounding_box.x_max, axis=[0, 1, 0])

        face = make_face([outer, inner], self._label)
        component = PhysicalComponent(self._label, face)
        component.plot_options.face_options["color"] = BLUE_PALETTE["PL"]

        return Component("xy").add_child(component)

    def build_xyz(self, degree: float = 360.0) -> PhysicalComponent:
        """
        Build a PhysicalComponent with a BluemiraShell using the plasma boundary in 3D.

        The 3D shell is created by revolving the boundary through the angle provided
        through the degree parameter.

        Parameters
        ----------
        degree: float
            The angle [°] around which to revolve the 3D geometry, by default 360.0.

        Returns
        -------
        result: PhysicalComponent
            The resulting component.
        """
        shell = revolve_shape(self._boundary, direction=(0, 0, 1), degree=degree)
        component = PhysicalComponent(self._label, shell)
        component.display_cad_options.color = BLUE_PALETTE["PL"]

        return Component("xyz").add_child(component)

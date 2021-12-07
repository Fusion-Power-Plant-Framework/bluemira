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
EU-DEMO build classes for TF Coils.
"""
from typing import Optional, List
from copy import deepcopy

from bluemira.base.builder import Builder, BuildConfig
from bluemira.base.look_and_feel import bluemira_warn, bluemira_debug, bluemira_print
from bluemira.base.parameter import ParameterFrame
from bluemira.base.components import Component, PhysicalComponent
from bluemira.builders.tf_coils import RippleConstrainedLengthOpt
from bluemira.geometry.tools import offset_wire, sweep_shape
from bluemira.geometry.face import BluemiraFace


class TFCoilsComponent(Component):
    def __init__(
        self,
        name: str,
        parent: Optional[Component] = None,
        children: Optional[List[Component]] = None,
        magnetics=None,
    ):
        super().__init__(name, parent=parent, children=children)
        self._magnetics = magnetics

    @property
    def magnetics(self):
        return self._magnetics

    @magnetics.setter
    def magnetics(self, magnetics):
        self._magnetics = magnetics


class BuildTFWindingPack:
    """
    A class to build TF coil winding pack geometry
    """

    name = "TFWindingPack"

    def __init__(self, wp_centreline, wp_cross_section):
        self.wp_centreline = wp_centreline
        self.wp_cross_section = wp_cross_section

    def build_xy(self):
        # Should normally be gotten with wire_plane_intersect
        # (it's not OK to assume that the maximum x value occurs on the midplane)
        x_out = self.wp_centreline.bounding_box.x_max

        xs = deepcopy(self.wp_cross_section)
        xs2 = deepcopy(xs)
        xs2.translate((x_out - xs2.center_of_mass[0], 0, 0))

        return [
            PhysicalComponent(self.name, xs),
            PhysicalComponent(self.name, xs2),
        ]

    def build_xz(self):
        x_min = self.wp_cross_section.bounding_box.x_min
        x_centreline_in = self.wp_centreline.bounding_box.x_min
        dx = abs(x_min - x_centreline_in)
        outer = offset_wire(self.wp_centreline, dx)
        inner = offset_wire(self.wp_centreline, -dx)
        return PhysicalComponent(self.name, BluemiraFace([outer, inner], self.name))

    def build_xyz(self):
        solid = sweep_shape(
            self.wp_cross_section.boundary[0], self.wp_centreline, label=self.name
        )
        return PhysicalComponent(self.name, solid)


class TFCoilsBuilder(Builder):
    _required_params: List[str] = []
    _required_config: List[str] = []
    _params: ParameterFrame

    def __init__(self, params, build_config: BuildConfig, **kwargs):
        super().__init__(params, build_config, **kwargs)
        self._sub_builders = [BuildTFWindingPack]

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        """
        bluemira_debug(f"Reinitialising {self.name}")
        self._reset_params(params)

    def build(self, label: str = "TF Coils", **kwargs) -> Component:
        """
        Build the components from parameterised shapes using the provided configuration
        and parameterisation.

        Returns
        -------
        component: Component
            The Component built by this builder.
        """
        super().build(**kwargs)

        component = TFCoilsComponent(self.name)

        component.add_child(self.build_xz(label=label))
        component.add_child(self.build_xy(label=label))
        component.add_child(self.build_xyz(label=label))
        return component

    def build_xz(self, **kwargs):
        component = Component("xz")

        for sub_builder in self._sub_builders:
            sub_comp = sub_builder.build_xz()
            component.add_child(sub_comp)

        return component

    def build_xy(self, **kwargs):
        component = Component("xy")

        return component

    def build_xyz(self, **kwargs):
        component = Component("xyz")

        return component

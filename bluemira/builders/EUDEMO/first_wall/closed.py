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

import copy
from typing import Any, Dict, List

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Component
from bluemira.base.components import PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.builders.shapes import ParameterisedShapeBuilder
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.parameterisations import PolySpline
from bluemira.geometry.wire import BluemiraWire


class FirstWallPolySpline(PolySpline):
    """
    Defines the geometry for reactor first wall, without a divertor,
    based on a PolySpline
    """

    _default_dict = {
        "x1": {"value": 5.8},
        "x2": {"value": 12.1},
        "z2": {"value": 0},
        # height = kappa_95*(R_0/A)*2
        "height": {"value": 1.6 * (9 / 3.1) * 2},
        "top": {"value": 0.5},
        "upper": {"value": 0.7},
        "dz": {"value": 0},
        "flat": {"value": 0},
        "tilt": {"value": 0},
    }

    def __init__(self, var_dict=None):
        if var_dict is None:
            var_dict = {}
        defaults = copy.deepcopy(self._default_dict)
        defaults.update(var_dict)
        super().__init__(defaults)


class ClosedFirstWallBuilder(ParameterisedShapeBuilder):
    """
    Builder class for the first wall of the reactor, closed with
    no divertor.
    """

    _required_params: List[str] = [
        "plasma_type",
        "R_0",  # major radius
        "kappa_95",  # 95th percentile plasma elongation
        "r_fw_ib_in",  # inboard first wall inner radius
        "r_fw_ob_in",  # inboard first wall outer radius
        "A",  # aspect ratio
    ]
    _required_config: List[str] = []
    _params: Configuration
    _default_runmode: str = "mock"

    def __init__(self, params: Dict[str, Any], build_config: BuildConfig, **kwargs):
        super().__init__(params, build_config, **kwargs)

        # boundary should be set by run/mock/read, it is used by the build methods
        self.boundary: BluemiraWire = None

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        return super().reinitialise(params, **kwargs)

    def mock(self):
        """
        Create a basic shape for the wall's boundary.
        """
        self.boundary = self._shape.create_shape()

    def build(self, **kwargs) -> Component:
        """
        Build the components.
        """
        component = super().build(**kwargs)
        component.add_child(self.build_xz())
        return component

    def build_xz(self, **kwargs) -> Component:
        """
        Build the components in the xz plane.
        """
        component = Component("xz")
        component.add_child(
            PhysicalComponent(
                "first_wall", BluemiraWire(self.boundary, label="first_wall")
            )
        )
        component.plot_options.wire_options["color"] = BLUE_PALETTE["DIV"]
        bm_plot_tools.set_component_plane(component, "xz")
        return component

    def _derive_shape_params(self):
        """
        Calculate derived parameters for the GeometryParameterisation.
        """
        params = super()._derive_shape_params()
        r_minor = self._params.R_0 / self._params.A
        height = (self._params.kappa_95 * r_minor) * 2
        params["height"] = {"value": height}
        return params

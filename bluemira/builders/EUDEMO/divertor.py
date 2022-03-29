# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Define builder for divertor
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Builder, Component
from bluemira.base.components import PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.base.error import BuilderError
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import find_flux_surface_through_point, get_legs
from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire


class DivertorBuilder(Builder):
    """
    Build an EUDEMO divertor.
    """

    _required_params: List[str] = [
        "reactor_type",
        "plasma_type",
    ]

    _params: Configuration
    _silhouette: Optional[BluemiraWire] = None
    _default_runmode: str = "run"

    def reinitialise(self, params) -> None:
        """
        Reinitialise the parameters and boundary.

        Parameters
        ----------
        params: dict
            The new parameter values to initialise this builder against.
        """
        super().reinitialise(params)

        self._silhouette = None

    def build(self) -> Component:
        """
        Build the divertor component.

        Returns
        -------
        component: Component
            The Component built by this builder.
        """
        super().build()

        component = Component(name=self.name)
        component.add_child(self.build_xz())
        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())
        return component

    def build_xz(self):
        """
        Build the x-z components of the divertor.
        """
        component = Component("xz", children=[])
        bm_plot_tools.set_component_view(component, "xz")
        return component

    def build_xy(self):
        """
        Build the x-y components of the divertor.
        """
        component = Component("xy", children=[])
        bm_plot_tools.set_component_view(component, "xy")
        return component

    def build_xyz(self, degree=360.0):
        """
        Build the x-y-z components of the divertor.
        """
        component = Component("xyz", children=[])
        return component

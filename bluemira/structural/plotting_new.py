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
Structural module plotting tools
"""
import numpy as np
from matplotlib.colors import DivergingNorm, Normalize

from bluemira.utilities.plot_tools import Plot3D

DEFAULT_PLOT_OPTIONS = {
    "show_all_nodes": True,
    "show_stress": False,
    "show_deflection": False,
    "interpolate": False,
    "show_cross_sections": True,
    "annotate_nodes": True,
    "annotate_elements": True,
    "node_options": {"marker": "o", "ms": 12, "color": "k"},
    "symmetry_node_color": "g",
    "support_node_color": "r",
    "element_options": {"linewidth": 3, "color": "k", "linestyle": "-"},
    "show_as_grey": False,
}


def annotate_node(ax, node, text_size, color):
    name = f"N{node.id_number}"
    ax.text(
        node.x,
        node.y,
        node.z,
        name,
        size=text_size,
        color=color,
    )


def annotate_element(ax, element, text_size, color):
    name = f"E{element.id_number}"
    ax.text(
        *element.mid_point,
        name,
        size=text_size,
        color=color,
    )


class BasePlotter:
    def __init__(self, geometry, ax=None, **kwargs):
        self.geometry = geometry
        if ax is None:
            self.ax = Plot3D()
        else:
            self.ax = ax

        self.options = {**DEFAULT_PLOT_OPTIONS, **kwargs}

    def plot_nodes(self):
        """
        Plots all the nodes in the geometry
        """
        kwargs = self.options["node_options"].copy()
        default_color = kwargs.pop(
            "color", DEFAULT_PLOT_OPTIONS["node_options"]["color"]
        )

        for node in self.geometry.nodes:
            if node.supports.any():
                color = self.options["support_node_color"]
            elif node.symmetry:
                color = self.options["symmetry_node_color"]
            else:
                color = default_color

            self.ax.plot([node.x], [node.y], [node.z], color=color, **kwargs)

            if self.options["annotate_nodes"]:
                annotate_node(self.ax, node, self.text_size, color)

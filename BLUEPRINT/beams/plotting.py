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
FE plotting tools
"""
import numpy as np
from matplotlib.colors import DivergingNorm, Normalize
from BLUEPRINT.utilities.plottools import Plot3D
from BLUEPRINT.beams.constants import (
    LOAD_STR_VECTORS,
    LOAD_INT_VECTORS,
    FLOAT_TYPE,
    STRESS_COLOR,
    DEFLECT_COLOR,
)

# Plotting options
DEFAULT_OPTIONS = {
    "SCALE": 1.1,  # Scaling factor for 3-D axes offset
    "NODE_KWARGS": {"marker": "o", "ms": 12, "color": "k"},
    "ELEM_KWARGS": {"linewidth": 3, "color": "k"},
    "ANNOTATE": True,
    "CROSSSECTIONS": True,
    "STRESSES": False,
    "DEFLECTIONS": False,
    "INTERPOLATE": False,
    "SHOW_ALL_NODES": True,
    "GREY_OUT": False,
}


def arrow_scale(vector, max_length, max_force):
    """
    Scales an arrow such that, regardless of direction, it has a reasonable
    size

    Parameters
    ----------
    vector: np.array(3)
        3-D vector of the arrow
    max_length: float
        The maximum length of the arrow
    max_force: float
        The maximum force value in the model (absolute)

    Returns
    -------
    vector: np.array(3)
        The scaled force arrow
    """
    v_norm = np.linalg.norm(vector)
    if v_norm == 0:
        return vector  # who cares? No numpy warning

    scale = (max_length * np.abs(vector)) / max_force

    return scale * vector / v_norm


def _plot_force(ax, node, vector, color="r"):
    """
    Plots a single force arrow in 3-D to indicate a linear load

    Parameters
    ----------
    ax: matplotlib Axes3D object
        The ax on which to plot
    node: Node object
        The node or location at which the force occurs
    vector: np.array(3)
        The force direction vector
    color: str
        The color to plot the force as
    """
    ax.quiver(
        node.x - vector[0], node.y - vector[1], node.z - vector[2], *vector, color=color
    )


def _plot_moment(ax, node, vector, color="r", support=False):
    """
    Plots a double "moment" arrow in 3-D to indicate a moment load. Offset the
    moment arrows off from the nodes a little, to enable overlaps with forces.

    Parameters
    ----------
    ax: matplotlib Axes3D object
        The ax on which to plot
    node: Node object
        The node or location at which the force occurs
    vector: np.array(3)
        The force direction vector
    color: str
        The color to plot the force as
    """
    if support:
        # Offsets the moment arrows a little so we can see overlaps with forces
        vector *= 2
        f1 = 0.5
        f2 = 0.25
    else:
        f1 = 1
        f2 = 0.5
    ax.quiver(
        node.x - vector[0],
        node.y - vector[1],
        node.z - vector[2],
        *f1 * vector,
        color=color
    )
    ax.quiver(
        node.x - vector[0],
        node.y - vector[1],
        node.z - vector[2],
        *f2 * vector,
        color=color,
        arrow_length_ratio=0.6
    )


class BasePlotter:
    """
    Base utility plotting class for structural models
    """

    def __init__(self, geometry, ax=None, **kwargs):
        if ax is None:
            self.ax = Plot3D()
        else:
            self.ax = ax

        if kwargs:
            for k in kwargs:
                if k in self.options:
                    self.options[k] = kwargs[k]

        self.geometry = geometry
        self._unit_length = None
        self._force_size = None
        self._size = None

        self.color_normer = None

    @property
    def unit_length(self):
        """
        Calculates a characteristic unit length for the model: the minimum
        element size
        """
        if self._unit_length is None:
            lengths = np.zeros(self.geometry.n_elements)
            for i, element in enumerate(self.geometry.elements):
                lengths[i] = element.length
            self._unit_length = np.min(lengths)

        return self._unit_length

    @property
    def force_size(self):
        """
        Calculates a characteristic force vector length for plotting purposes

        Returns
        -------
        f_length: float
            The minimum and maximum forces
        """
        if self._force_size is None:
            loads = []
            for element in self.geometry.elements:
                for load in element.loads:
                    if load["type"] == "Element Load":
                        loads.append(load["Q"])
                    elif load["type"] == "Distributed Load":
                        loads.append(load["w"] / element.length)

            for node in self.geometry.nodes:
                for load in node.loads:
                    loads.append(load["Q"])

            self._force_size = np.max(np.abs(loads))

        return self._force_size

    @property
    def size(self):
        """
        Calculates the size of the model bounding box
        """
        if self._size is None:
            xmax, xmin, ymax, ymin, zmax, zmin = self.geometry.bounds()

            self._size = max([xmax - xmin, ymax - ymin, zmax - zmin])

        return self._size

    @property
    def text_size(self):
        """
        Get a reasonable guess of the font size to use in plotting.

        Returns
        -------
        size: float
            The font size to use in plotting
        """
        return max(10, self.size // 30)

    def plot_nodes(self):
        """
        Plots all the nodes in the geometry
        """
        kwargs = self.options["NODE_KWARGS"].copy()
        color = kwargs["color"]

        for node in self.geometry.nodes:
            name = "N" + str(node.id_number)

            if self.options["GREY_OUT"]:
                continue
            if node.supports.any():
                kwargs["color"] = "g"
            elif node.symmetry:
                kwargs["color"] = "r"

            elif not self.options["SHOW_ALL_NODES"]:
                continue

            self.ax.plot([node.x], [node.y], [node.z], **kwargs)
            if self.options["ANNOTATE"]:
                self.ax.text(
                    node.x,
                    node.y,
                    node.z,
                    name,
                    size=self.text_size,
                    color=kwargs["color"],
                )
            kwargs["color"] = color

    def plot_supports(self):
        """
        Plots all supports in the Geometry model
        """
        for node in self.geometry.nodes:
            if node.supports.any():
                self._plot_support(node)

    def _plot_support(self, node):
        """
        Plots the supports at a single Node in the Geometry model
        """
        # Get a small distance in the model
        lengths = np.array([e.length for e in self.geometry.elements])
        length = lengths.min() / 5
        for i, support in enumerate(node.supports):
            vector = length * LOAD_INT_VECTORS[i]
            if support and i < 3:
                # Linear support (single black arrow)
                _plot_force(self.ax, node, vector, color="k")
            elif support and i >= 3:
                # Moment support (double red arrow, offset to enable overlap)
                _plot_moment(self.ax, node, vector, support=True, color="g")

    def plot_elements(self):
        """
        Plots all of the elements in the geometry
        """
        kwargs = self.options["ELEM_KWARGS"].copy()
        color = kwargs["color"]

        if self.options["GREY_OUT"]:
            kwargs["color"] = "grey"
            kwargs["alpha"] = 0.5

        for element in self.geometry.elements:
            name = "E" + str(element.id_number)
            x = [element.node_1.x, element.node_2.x]
            y = [element.node_1.y, element.node_2.y]
            z = [element.node_1.z, element.node_2.z]

            if not self.options["GREY_OUT"]:
                if self.options["STRESSES"]:
                    c_i = self.color_normer(element.max_stress)
                    kwargs["color"] = STRESS_COLOR(c_i)
                elif self.options["DEFLECTIONS"]:
                    c_i = self.color_normer(element.max_displacement)
                    kwargs["color"] = DEFLECT_COLOR(c_i)
                else:
                    pass

            self.ax.plot(x, y, z, **kwargs)
            if self.options["ANNOTATE"]:
                self.ax.text(
                    sum(x) / 2,
                    sum(y) / 2,
                    sum(z) / 2,
                    name,
                    size=self.text_size,
                    color=kwargs["color"],
                )
            if self.options["INTERPOLATE"]:
                self.ax.plot(*element.shapes, linestyle="--", **kwargs)

        kwargs["color"] = color

    def plot_cross_sections(self):
        """
        Plots the cross-sections for each element in the geometry, rotated to
        the mid-point of the element
        """
        if self.options["GREY_OUT"]:
            facecolor = "grey"
            edgecolor = "darkgrey"
            alpha = 0.2
        else:
            facecolor = "b"
            edgecolor = "k"
            alpha = 0.5

        for element in self.geometry.elements:

            # Get the centre-point of the element cross-section
            dx = (element.node_1.x + element.node_2.x) / 2
            dy = (element.node_1.y + element.node_2.y) / 2
            dz = (element.node_1.z + element.node_2.z) / 2

            cs = element._cross_section.geometry

            if not isinstance(cs, list):
                # Handles multiple Loops and Shells in cross-section
                cs = [cs]

            dcm = element.lambda_matrix[0:3, 0:3]
            for sub_cs in cs:
                loop = sub_cs.rotate_dcm(dcm.T, update=False)
                loop.translate([dx, dy, dz], update=True)
                loop.plot(self.ax, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)

    def plot_loads(self):
        """
        Plots all of the loads applied to the geometry
        """
        for node in self.geometry.nodes:
            if node.loads:
                for load in node.loads:
                    self._plot_node_load(node, load)

        for element in self.geometry.elements:
            for load in element.loads:
                if load["type"] == "Element Load":
                    self._plot_element_load(element, load)
                elif load["type"] == "Distributed Load":
                    self._plot_distributed_load(element, load)

    def _plot_node_load(self, node, load):
        load_value = load["Q"] * LOAD_STR_VECTORS[load["sub_type"]]

        load_value = arrow_scale(load_value, 10 * self.unit_length, self.force_size)

        if "F" in load["sub_type"]:
            _plot_force(self.ax, node, load_value, color="r")

        elif "M" in load["sub_type"]:
            _plot_moment(self.ax, node, load_value, color="r")

    def _plot_element_load(self, element, load):
        load = load["Q"] * LOAD_STR_VECTORS[load["sub_type"]]

        load = arrow_scale(load, 10 * self.unit_length, self.force_size)

        dcm = element.lambda_matrix[0:3, 0:3]
        load = load @ dcm
        point = np.array(
            [element.node_1.x, element.node_1.y, element.node_1.z], dtype=FLOAT_TYPE
        )
        point += (np.array([1.0, 0.0, 0.0]) * np.float(load["x"])) @ dcm
        self.ax.quiver(*point - load, *load, color="r")

    def _plot_distributed_load(self, element, load):
        length = element.length
        n = int(length * 10)
        dcm = element.lambda_matrix[0:3, 0:3]
        load = load["w"] * LOAD_STR_VECTORS[load["sub_type"]] / length

        load = arrow_scale(load, 10 * self.unit_length, self.force_size)

        load = load @ dcm
        load = load * np.ones((3, n)).T
        load = load.T
        point = np.array(
            [element.node_1.x, element.node_1.y, element.node_1.z], dtype=FLOAT_TYPE
        )
        point = point * np.ones((3, n)).T
        point += (
            np.array([x * np.array([1.0, 0.0, 0.0]) for x in np.linspace(0, length, n)])
            @ dcm
        )
        point = point.T
        self.ax.quiver(*point - load, *load, color="r")

    def _set_aspect_equal(self):
        """
        Hack to make matplotlib 3D look good. Draw a white bounding box around
        the nodes
        """
        x_bb, y_bb, z_bb = self.geometry.bounding_box()

        x_bb *= self.options["SCALE"]
        y_bb *= self.options["SCALE"]
        z_bb *= self.options["SCALE"]

        for x, y, z in zip(x_bb, y_bb, z_bb):
            self.ax.plot([x], [y], [z], color="w")


class GeometryPlotter(BasePlotter):
    """
    Utility class for the plotting of Beams geometry models
    """

    def __init__(self, geometry, ax=None, **kwargs):
        self.options = DEFAULT_OPTIONS.copy()
        self.options["STRESSES"] = False
        self.options["DEFLECTIONS"] = False
        super().__init__(geometry, ax, **kwargs)

        self.plot_nodes()
        self.plot_elements()
        self.plot_supports()
        self.plot_loads()
        if self.options["CROSSSECTIONS"]:
            self.plot_cross_sections()
        self._set_aspect_equal()


class DeformedGeometryPlotter(BasePlotter):
    """
    Utility class for the plotting of Beams deformed geometry models and
    overlaying with GeometryPlotters
    """

    def __init__(self, geometry, ax=None, **kwargs):
        self.options = DEFAULT_OPTIONS.copy()
        self.options["NODE_KWARGS"] = {"marker": "o", "ms": 12, "color": "b"}
        self.options["ELEM_KWARGS"] = {"linewidth": 3, "color": "b"}
        self.options["STRESSES"] = False
        self.options["ANNOTATE"] = True
        self.options["INTERPOLATE"] = True
        self.options["SHOW_ALL_NODES"] = False

        super().__init__(geometry, ax, **kwargs)

        self.plot_nodes()
        self.plot_elements()
        self._set_aspect_equal()


class StressDeformedGeometryPlotter(BasePlotter):
    """
    Utility class for the plotting of Beams deformed geometry models and
    overlaying with GeometryPlotters
    """

    def __init__(self, geometry, ax=None, stress=None, deflection=False, **kwargs):
        self.options = DEFAULT_OPTIONS.copy()
        self.options["NODE_KWARGS"] = {"marker": "o", "ms": 12, "color": "b"}
        self.options["ELEM_KWARGS"] = {"linewidth": 3, "color": None}
        self.options["STRESSES"] = True
        self.options["ANNOTATE"] = False
        self.options["INTERPOLATE"] = True
        self.options["SHOW_ALL_NODES"] = False
        self.options["GREY_OUT"] = False

        super().__init__(geometry, ax, **kwargs)

        self.color_normer = self.make_color_normer(stress, deflection)

        self.plot_nodes()
        self.plot_elements()
        self._set_aspect_equal()

    @staticmethod
    def make_color_normer(stress, deflection=False):
        """
        Make a ColorNorm object for the plot based on the stress values.
        """
        smin, smax = min(stress), max(stress)
        if smin == smax:

            class SameColour:
                def __call__(self, value):
                    return 0.5

            return SameColour()

        centre = 0

        if deflection:
            # deflections positive when plotting
            deflections = np.abs(stress)
            return Normalize(vmin=0, vmax=max(deflections))

        if not smin < 0 < smax:
            centre = (smin + smax) / 2

        return DivergingNorm(centre, vmin=min(stress), vmax=max(stress))


if __name__ == "__main__":
    from BLUEPRINT import test

    test()

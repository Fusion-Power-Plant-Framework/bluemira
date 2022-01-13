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
The base component object for CAD modelling
"""
from itertools import cycle

from bluemira.base.look_and_feel import bluemira_warn
from BLUEPRINT.base.error import CADError
from BLUEPRINT.cad.cadtools import (
    get_properties,
    make_compound,
    rotate_shape,
    save_as_STEP,
    save_as_STEP_assembly,
    save_as_STL,
)


class ComponentCAD:
    """
    Component CAD parent (abstract) object.

    Parameters
    ----------
    name: str
        The name of the component
    *args:
        Build specific information

         neutronics: bool
             True: runs self.build_neutronics() on __init__
             False: runs self.build() on __init__

    Attributes
    ----------
    .component: dict
        Dictionary of sub-parts
    """

    def __init__(self, name, *args, pair=False, palette="Paired", n_colors=5, **kwargs):
        self.name = name
        self.args = args  # store build specific argument list
        self.pair = pair  # cycles color saturation within component
        self.palette = cycle(palette)
        self.n_colors = n_colors
        self.component = {
            "shapes": [],
            "names": [],
            "sub_names": [],
            "colors": [],
            "transparencies": [],
        }
        self.n_shape = 0  # number of shapes within component
        self.slice_flag = kwargs.get("slice_flag", False)
        if kwargs.get("neutronics", False):
            self.build_neutronics()
        else:
            self.build(from_compound=kwargs.get("from_compound", False))

    @classmethod
    def from_compound(
        cls, compound, name, pair=False, palette="Paired", n_colors=5, **kwargs
    ):
        """
        Makes a ComponentCAD object from a compound of shapes

        Parameters
        ----------
        compound: OCC Compound object
            The compound of shapes from which to make a component
        name: str
            The name of the component

        Returns
        -------
        component: ComponentCAD
            The compound component
        """
        return cls(name, pair, palette, n_colors, from_compound=compound, **kwargs)

    @staticmethod
    def _check_shape(name, shape):
        if not shape.Closed():
            # The loop is open
            pass

        if shape.IsNull():
            bluemira_warn(f"Object {name} does not exist!")
        if shape.Infinite():
            bluemira_warn(f"Object {name} is infinite!")

    def add_shape(self, shape, **kwargs):
        """
        Adds a shape to the component

        Parameters
        ----------
        shape: OCC TopoDS_* object
            The shape to be added to the component during the customised
            .build() sequence defined in the inherited class.

        Other Parameters
        ----------------
        name: str ==> sub_name
            Name of the shape in the component
        color: TBD
            Colors of the shapes
        transparency:
            Transparencies of the shapes
        """
        self.n_shape += 1
        sub_name = kwargs.get("name", "{}".format(self.n_shape))
        name = "{}_{}".format(self.name, sub_name)
        self._check_shape(name, shape)
        color = kwargs.get("color", next(self.palette))
        transp = kwargs.get("transparency", 1)
        self.component["shapes"].append(shape)
        self.component["sub_names"].append(sub_name)
        self.component["names"].append(name)
        self.component["colors"].append(color)
        self.component["transparencies"].append(transp)

    def remove_shape(self, name):
        """
        Removes a shape from the component build dictionary

        Parameters
        ----------
        name: str
            The name of the shape to remove
        """
        if name not in self.component["names"]:
            raise CADError(f'There is no "{name}" component in this model.')
        index = self.component["names"].index(name)
        for value in self.component.values():
            value.pop(index)
        self.n_shape -= 1

    def build(self, **kwargs):
        """
        Automatically called upon instantiation. Specific to each inherited
        class. May be overriden if: initialising a Component from a compound or
        building a Component for neutronics (alternative build procedure)

        Other Parameters
        ----------------
        from_compound: bool or OCC Compound
            Initialises Component from compound
        """
        compound = kwargs.get("from_compound", False)
        if not compound:
            raise NotImplementedError("CAD build function not set.")
        self.add_shape(compound, name=self.name)

    def build_neutronics(self, **kwargs):
        """
        Automatically called upon instantiation if neutronics=True
        """
        raise NotImplementedError("Neutronics CAD build function not set.")

    def get_compound(self):
        """
        Returns
        -------
        compound: an OCC compound object of all shapes within component
        """
        return make_compound(self.component["shapes"])

    def get_properties(self, lift_points=None):
        """
        Gets the properties of the Component

        Parameters
        ----------
        lift_points: (float, float, float) (default=None)
            If not none, used to calculate the radius of gyration about the
            lift_point with a vertical axis

        Returns
        -------
        props: dict
            Dictionary of properties for each shape:
            {'name1': {'Volume': 0, 'CoG': [0, 0, 0], 'Rg' 0}, {...}}
        """
        props = {}
        if lift_points is None:
            lift_points = {k: None for k in self.component["sub_names"]}
        for name, shape in zip(self.component["sub_names"], self.component["shapes"]):
            props[name] = get_properties(shape, lift_points[name])
        return props

    def component_pattern(self, n):
        """
        Patterns component at the part object level, creating a compound
        """
        shapes = self.component["shapes"]
        self.component["shapes"] = []  # Phoenix
        for shape in shapes:
            pattern = self.part_pattern(shape, n)
            compound = make_compound(pattern)
            self.component["shapes"].append(compound)

    @staticmethod
    def part_pattern(shape, n):
        """
        Patterns a shape axisymmetrically about the machine axis

        Parameters
        ----------
        shape: OCC BRep object
            The shape to pattern
        n: int
            The number of times to pattern the part (equi-spaced)

        Returns
        -------
        shapes: list(BRep, Brep, ..) (length = n)
            The n-list of patterned shapes
        """
        shapes = []
        for i in range(1, int(n) + 1):
            shapes.append(rotate_shape(shape, axis=None, angle=i * 360 / n))
        return shapes

    def split(self, names, subcomponents):
        """
        Splits the ComponentCAD object into multiple ones

        Parameters
        ----------
        names: list(str, str, ..)
            The names of the new components (length = n)
        subcomponents: list(list(str, ..), list(str, ..))
            The n-list of sub-component names to split into n components

        Returns
        -------
        comp_list: list(ComponentCAD, ComponentCAD, ..)
            The n-list of CAD components which were split into n
        """
        subs = []
        for group in subcomponents:
            g_c = []
            for comp in group:
                i = self.component["sub_names"].index(comp)
                g_c.append(self.component["shapes"][i])
            subs.append(make_compound(g_c))
        return [
            ComponentCAD.from_compound(comp, name, color_index=j)
            for j, (comp, name) in enumerate(zip(subs, names))
        ]

    def merge(self):
        """
        Returns a new instance of ComponentCAD, with all parts as a single
        compound

        Returns
        -------
        componentcad: ComponentCAD
            The compounded CAD component
        """
        return ComponentCAD.from_compound(self.get_compound(), self.name)

    def save_as_STL(self, filename, scale=1):  # noqa :N802
        """
        Saves the Component parts to STL files in the same directory

        Parameters
        ----------
        filename: str
            The full root filename, which will be incremented in the save
            filename_1.stl, filename_2.stl
        scale: float (default=1)
            The factor with which to scale the geometry
        """
        if filename.endswith(".stl"):
            filename = filename[:-4]
        if "plasma" in filename:
            save_as_STL(self.component["shapes"][0], f"{filename}.stl")
        else:
            for i, shape in enumerate(self.component["shapes"]):
                filename_mod = f"{filename}_{i}.stl"
                save_as_STL(shape, filename_mod, scale=scale)

    def save_as_STEP(self, filename, scale=1):  # noqa :N802
        """
        Saves all Component parts to STEP files in the same directory

        Parameters
        ----------
        filename: str
            The full root filename, which will be incremented in the save
            filename_shapename_1.STP, filename_shapename_2.STP
        scale: float (default=1)
            The factor with which to scale the geometry
        """
        if filename.endswith(".STP"):
            filename = filename[:-4]
        if "plasma" in filename:
            save_as_STEP(self.component["shapes"][0], f"{filename}.STP")
        else:
            for i, (name, shape) in enumerate(
                zip(self.component["names"], self.component["shapes"])
            ):
                filename_mod = f"{filename}_{name}_{i}.STP"
                save_as_STEP(shape, filename_mod, scale=scale)

    def save_as_STEP_assembly(self, filename):  # noqa :N802
        """
        Saves the Component to a STEP assembly file

        Parameters
        ----------
        filename: str
            The full root filename
        """
        if not filename.endswith(".STP"):
            filename += ".STP"
        save_as_STEP_assembly(self.component["shapes"], filename)

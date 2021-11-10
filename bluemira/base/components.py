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
Module containing the base Component class.
"""

import anytree
from anytree import NodeMixin, RenderTree
import copy
from typing import Any, List, Optional, Type, Union

from bluemira.display.plotter import Plottable
from bluemira.display.displayer import DisplayableCAD

from .error import ComponentError


class Component(NodeMixin, Plottable, DisplayableCAD):
    """
    The Component is the fundamental building block for a bluemira reactor design. It
    encodes the way that the corresponding part of the reactor will be built, along with
    any other derived properties that relate to that component.

    Components define a tree structure, based on the parent and children properties. This
    allows the nodes on that tree to be passed around within bluemira so that
    operations can be performed on the child branches of that structure.

    For example, a reactor design including just a TFCoilSystem may look as below:

    .. digraph:: base_component_tree

      "FusionPowerPlant" -> "TFCoilSystem" -> {"TFWindingPack" "TFCasing"}

    A Component cannot be used directly - only subclasses should be instantiated.
    """

    name: str

    def __init__(
        self,
        name: str,
        parent: Optional["Component"] = None,
        children: Optional[List["Component"]] = None,
    ):
        super().__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children

    def __new__(cls, *args, **kwargs) -> Type["Component"]:
        """
        Constructor for Component class.
        """
        if cls is Component:
            raise ComponentError(
                "Component cannot be initialised directly - use a subclass."
            )
        return super().__new__(cls)

    def __repr__(self) -> str:
        """
        The string representation of the instance
        """
        return self.name + " (" + self.__class__.__name__ + ")"

    def tree(self) -> str:
        """
        Get the tree of descendants of this instance.
        """
        return str(RenderTree(self))

    def get_component(
        self, name: str, first: bool = True, full_tree: bool = False
    ) -> Union["Component", List["Component"]]:
        """
        Find the components with the specified name.

        Parameters
        ----------
        name: str
            The name of the component to search for.
        first: bool
            If True, only the first element is returned, by default True.
        full_tree: bool
            If True, searches the tree from the root, else searches from this node, by
            default False.

        Returns
        -------
        found_components: Union[Component, List[Component]]
            The first component of the search if first is True, else all components
            matching the search.

        Notes
        -----
            This function is just a wrapper of the anytree.search.findall_by_attr
            function.
        """
        if full_tree:
            found_components = anytree.search.findall_by_attr(self.root, name)
        else:
            found_components = anytree.search.findall_by_attr(self, name)

        if len(found_components) == 0:
            return None

        if first:
            return found_components[0]

        return found_components

    def copy(self):
        """
        Provides a deep copy of the Component

        Returns
        -------
        copy: Component
            The copy of the Component
        """
        return copy.deepcopy(self)


class GroupingComponent(Component):
    """
    A Component that groups other Components.
    """

    def add_child(self, child: Component):
        """
        Add a single child to this node

        Parameters
        ----------
        child: Component
            The child to be added
        """
        if child in self.children:
            raise ComponentError(f"Component {child} is already a child of {self}")
        self.children = list(self.children) + [child]

    def add_children(self, children: List[Component]):
        """
        Add multiple children to this node

        Parameters
        ----------
        children: List[Component]
            The children to be added
        """
        duplicates = []
        for child in children:
            if child in self.children:
                duplicates += [child]
        if duplicates != []:
            raise ComponentError(
                f"Components {duplicates} are already a children of {self}"
            )
        self.children = list(self.children) + children


class PhysicalComponent(Component):
    """
    A physical component. It includes shape and materials.
    """

    def __init__(
        self,
        name: str,
        shape: Any,
        material: Any = None,
        parent: Component = None,
        children: Component = None,
    ):
        super().__init__(name, parent, children)
        self.shape = shape
        self.material = material

    @property
    def shape(self):
        """
        The geometric shape of the Component.
        """
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def material(self):
        """
        The material that the Component is built from.
        """
        return self._material

    @material.setter
    def material(self, value):
        self._material = value


class MagneticComponent(PhysicalComponent):
    """
    A magnetic component. It includes a shape, a material, and a source conductor.
    """

    def __init__(
        self,
        name: str,
        shape: Any,
        material: Any = None,
        conductor: Any = None,
        parent: Component = None,
        children: Component = None,
    ):
        super().__init__(name, shape, material, parent, children)
        self.conductor = conductor

    @property
    def conductor(self):
        """
        The conductor used by current-carrying filaments.
        """
        return self._conductor

    @conductor.setter
    def conductor(self, value):
        self._conductor = value

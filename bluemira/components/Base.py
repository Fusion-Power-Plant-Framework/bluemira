#  bluemira is an integrated inter-disciplinary design tool for future fusion
#  reactors. It incorporates several modules, some of which rely on other
#  codes, to carry out a range of typical conceptual fusion reactor design
#  activities.
#  #
#  Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
#                     D. Short
#  #
#  bluemira is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#  #
#  bluemira is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#  Lesser General Public License for more details.
#  #
#  You should have received a copy of the GNU Lesser General Public
#  License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

# import tree lib
import anytree
from anytree import NodeMixin, RenderTree

# import for abstract class
from abc import ABC


class Component(NodeMixin):
    """A component class. It is based on a tree structure."""

    def __init__(self, name: str, parent: Component = None, children: Component = None):
        """Constructor for the Component class."""
        self.name = name
        # parent
        self.parent = parent
        # children
        if children:
            self.children = children

    def get_component(self, name: str, first: bool = True):
        """
        Find the components with the specfied name.

        .. note::
            this function is just a wrapper of the anytree.search.findall_by_attr
            function.

        Parameters
        ----------

        name: str
            component's name to search.
        first: (bool, optional)
            if True, only the first element is returned.

        Returns
        -------

        Component:
            the first component of the search.

        """
        c = anytree.search.findall_by_attr(self, name)
        if not c:
            return c
        if first:
            return c[0]
        return c

    def __repr__(self):
        return self.name + " (" + self.__class__.__name__ + ")"


class PhysicalComponent(Component):
    """A physical component. It includes shape and materials.

        Parameters
        ----------
        name: str
            name of the component
        shape: BluemiraGeo
            main shape of the component
        material: Material
            material dictionary of the component (structure still to be agreed)
        parent: Component or None
        children: list(Component) or None
    """

    def __init__(self, name: str, shape, material=None, parent: Component = None,
                 children: Component = None):
        super().__init__(name, parent, children)
        self._shape = shape
        self._material = material

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def material(self):
        return self._material

    @shape.setter
    def material(self, value):
        self._material = self._material


class MagneticComponent(PhysicalComponent):
    """A magnetic component. It includes a source conductor"""

    def __init__(self, name: str, shape, material=None, conductor=None,
                 parent: Component = None, children: Component = None):
        """
        MagneticComponent constructor

        Parameters
        ----------
        name: str
            name of the component
        shape: BluemiraGeo
            main shape of the component
        material: Material
            material dictionary of the component (structure still to be agreed)
        conductor: SourceConductor
            conductor based on SourceConductor implemented in Magnetostatic (tba).
            It should include materials for the conductor (it's not the same as
            material for the shape)
        parent: Component or None
        children: list(Component) or None

        """
        super().__init__(name, shape, material, parent, children)
        self._conductor = conductor

    @property
    def conductor(self):
        return self._conductor

    @conductor.setter
    def conductor(self, value):
        self._conductor = value

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

import pytest

from bluemira.base.components import (
    Component,
    GroupingComponent,
    PhysicalComponent,
    MagneticComponent,
)
from bluemira.base.error import ComponentError


class TestComponentClass:
    """
    Tests for the base Component functionality.
    """

    def test_no_direct_initialisation(self):
        with pytest.raises(ComponentError):
            Component("Dummy")

    def test_tree(self):
        target_tree = """Parent (GroupingComponent)
└── Child (GroupingComponent)
    └── Grandchild (GroupingComponent)"""

        child = GroupingComponent("Child")
        parent = GroupingComponent("Parent", children=[child])
        grandchild = GroupingComponent("Grandchild", parent=child)
        assert parent.tree() == target_tree

        root: GroupingComponent = grandchild.root
        assert root.tree() == target_tree

    def test_get_component_full_tree(self):
        parent = GroupingComponent("Parent")
        child1 = GroupingComponent("Child1", parent=parent)
        child2 = GroupingComponent("Child2", parent=parent)
        grandchild = GroupingComponent("Grandchild", parent=child1)

        assert grandchild.get_component("Child2", full_tree=True) is child2

    def test_get_component_from_node(self):
        parent = GroupingComponent("Parent")
        child1 = GroupingComponent("Child1", parent=parent)
        child2 = GroupingComponent("Child2", parent=parent)
        grandchild = GroupingComponent("Grandchild", parent=child1)

        assert grandchild.get_component("Child2") is None

    def test_get_component_multiple_full_tree(self):
        parent = GroupingComponent("Parent")
        child1 = GroupingComponent("Child", parent=parent)
        child2 = GroupingComponent("Child", parent=parent)
        grandchild = GroupingComponent("Grandchild", parent=child1)

        components = grandchild.get_component("Child", first=False, full_tree=True)
        assert len(components) == 2
        assert components[0] is not components[1]
        assert components[0].parent == components[1].parent

    def test_get_component_multiple_from_node(self):
        parent = GroupingComponent("Parent")
        child1 = GroupingComponent("Child", parent=parent)
        child2 = GroupingComponent("Child", parent=parent)
        grandchild1 = GroupingComponent("Grandchild", parent=child1)
        grandchild2 = GroupingComponent("Grandchild", parent=child2)

        components = child1.get_component("Grandchild", first=False)
        assert len(components) == 1
        assert components[0] is grandchild1

        components = child2.get_component("Grandchild", first=False)
        assert len(components) == 1
        assert components[0] is grandchild2

    def test_get_component_missing(self):
        parent = GroupingComponent("Parent")
        child = GroupingComponent("Child", parent=parent)
        grandchild = GroupingComponent("Grandchild", parent=child)

        component = grandchild.get_component("Banana")
        assert component is None

    def test_copy(self):
        parent = GroupingComponent("Parent")
        child = GroupingComponent("Child", parent=parent)
        grandchild = GroupingComponent("Grandchild", parent=child)

        component = child.copy()
        assert component is not child
        assert component.parent is not parent
        assert component.name == child.name
        assert component.parent.name == parent.name
        assert all([child_.name == grandchild.name for child_ in child.children])

    def test_add_child(self):
        parent = GroupingComponent("Parent")
        child = GroupingComponent("Child")

        parent.add_child(child)
        assert parent.children == (child,)

    def test_fail_add_duplicate_child(self):
        parent = GroupingComponent("Parent")
        child = GroupingComponent("Child", parent=parent)

        with pytest.raises(ComponentError):
            parent.add_child(child)

    def test_add_children(self):
        parent = GroupingComponent("Parent")
        child1 = GroupingComponent("Child1")
        child2 = GroupingComponent("Child2")

        parent.add_children([child1, child2])
        assert parent.children == (child1, child2)

    def test_fail_add_duplicate_children(self):
        parent = GroupingComponent("Parent")
        child1 = GroupingComponent("Child1", parent=parent)
        child2 = GroupingComponent("Child2")

        with pytest.raises(ComponentError):
            parent.add_children([child1, child2])


class TestPhysicalComponent:
    """
    Tests for the PhysicalComponent class.
    """

    def test_shape(self):
        component = PhysicalComponent("Dummy", shape="A shape")
        assert component.shape == "A shape"

    def test_material_default(self):
        component = PhysicalComponent("Dummy", shape="A shape")
        assert component.material is None

    def test_material(self):
        component = PhysicalComponent("Dummy", shape="A shape", material="A material")
        assert component.material == "A material"


class TestMagneticComponent:
    """
    Tests for the MagneticComponent class.
    """

    def test_shape(self):
        component = MagneticComponent("Dummy", shape="A shape")
        assert component.shape == "A shape"

    def test_conductor_default(self):
        component = MagneticComponent("Dummy", shape="A shape")
        assert component.material is None

    def test_conductor(self):
        component = MagneticComponent("Dummy", shape="A shape", conductor="A conductor")
        assert component.conductor == "A conductor"


if __name__ == "__main__":
    pytest.main([__file__])

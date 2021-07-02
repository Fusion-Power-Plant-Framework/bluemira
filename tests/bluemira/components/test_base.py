# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

from bluemira.base.parameter import Parameter
import pytest
from typing import Type

from bluemira.components.base import (
    Component,
    GroupingComponent,
    PhysicalComponent,
    MagneticComponent,
)
from bluemira.components.error import ComponentError

from tests.bluemira.components.dummy_classes import (
    DummyDivertorProfile,
    DummyBreedingBlanket,
    DummyDivertor,
)


class TestComponentClass:
    """
    Tests for the base Component class functionality.
    """

    class MyComponent(GroupingComponent):
        """
        Test parent Component.
        """

        pass

    class MySubclassComponent(MyComponent):
        """
        Test subclass Component.
        """

        pass

    @pytest.mark.parametrize(
        "class_name,the_class",
        [
            ("MyComponent", MyComponent),
            ("MySubclassComponent", MySubclassComponent),
        ],
    )
    def test_get_class(self, class_name, the_class):
        """
        Test that reactor system classes can be retrieved
        """
        my_class = TestComponentClass.MyComponent.get_class(class_name)
        assert my_class is not None
        assert my_class == the_class

    def test_get_class_fail(self):
        with pytest.raises(ComponentError):
            TestComponentClass.MyComponent.get_class("NotAClass")

    def test_get_class_not_subclass(self):
        with pytest.raises(ComponentError):
            TestComponentClass.MySubclassComponent.get_class("MyComponent")

    def test_avoid_duplicate_classes(self):
        with pytest.raises(ComponentError):

            class MyComponent(GroupingComponent):
                pass

    def test_default_params(self):
        test_params = [
            ["R_0", "Major radius", 9, "m", None, "Input"],
            ["Name", "The reactor name", None, "N/A", None, "Input"],
        ]

        class DefaultParamsComponent(GroupingComponent):
            """
            System for testing.
            """

            default_params = test_params

        component = DefaultParamsComponent("Dummy", {}, {})
        assert len(component.default_params) == len(test_params)

        for param in test_params:
            component.default_params[param[0]] == param[2]
            component.params[param[0]] == component.default_params[param[0]]

    def test_params(self):
        test_params = [
            ["R_0", "Major radius", 9, "m", None, "Input"],
            ["Name", "The reactor name", None, "N/A", None, "Input"],
        ]

        class ParamsComponent(GroupingComponent):
            """
            System for testing.
            """

            default_params = test_params

        config = {
            "R_0": 8.5,
            "Name": "Dummy",
            "Dummy": False,
        }

        component = ParamsComponent("Dummy", config, {})
        assert len(component.params) == len(test_params)

        for param in test_params:
            component.params[param[0]] == config[param[0]]

    def test_add_parameter(self):
        test_params = [
            ["R_0", "Major radius", 9, "m", None, "Input"],
            ["Name", "The reactor name", None, "N/A", None, "Input"],
        ]

        dummy_param = ["Dummy", "A dummy parameter", False, "N/A", None, "Input"]

        class AddParamComponent(GroupingComponent):
            """
            System for testing.
            """

            default_params = test_params

        component = AddParamComponent("Dummy", {}, {})
        component.add_parameter(*dummy_param)
        assert len(component.params) == len(test_params) + 1

        for param in test_params:
            component.params[param[0]] == param[2]
        component.params[dummy_param[0]] == dummy_param[2]

        component = AddParamComponent("Dummy", {}, {})
        dummy_param = Parameter(*dummy_param)
        component.add_parameter(dummy_param)
        assert len(component.params) == len(test_params) + 1

        for param in test_params:
            component.params[param[0]] == param[2]
        component.params[dummy_param.var] == dummy_param

    def test_add_parameters(self):
        test_params = [
            ["R_0", "Major radius", 9, "m", None, "Input"],
            ["Name", "The reactor name", None, "N/A", None, "Input"],
        ]

        dummy_params = [
            ["Dummy", "A dummy parameter", False, "N/A", None, "Input"],
            ["Other", "Another dummy param", [1, 2, 3], "N/A", None, "Input"],
        ]

        class AddParamsComponent(GroupingComponent):
            """
            System for testing.
            """

            default_params = test_params

        component = AddParamsComponent("Dummy", {}, {})
        component.add_parameters(dummy_params)
        assert len(component.params) == len(test_params) + len(dummy_params)

        for param in test_params:
            component.params[param[0]] == param[2]
        for param in dummy_params:
            component.params[param[0]] == param[2]

        component = AddParamsComponent("Dummy", {}, {})
        dummy_param = Parameter(*dummy_params[0])
        other_param = Parameter(*dummy_params[1])
        dummy_params = [dummy_param, other_param]
        component.add_parameters(dummy_params)
        assert len(component.params) == len(test_params) + len(dummy_params)

        for param in test_params:
            component.params[param[0]] == param[2]
        for param in dummy_params:
            component.params[param.var] == param

    def test_no_direct_initialisation(self):
        with pytest.raises(ComponentError):
            component = Component("Dummy", {}, {})

    def test_tree(self):
        target_tree = """Parent (GroupingComponent)
└── Child (GroupingComponent)
    └── Grandchild (GroupingComponent)"""

        child = GroupingComponent("Child", {}, {})
        parent = GroupingComponent("Parent", {}, {}, children=[child])
        grandchild = GroupingComponent("Grandchild", {}, {}, parent=child)
        assert parent.tree() == target_tree

        root: GroupingComponent = grandchild.root
        assert root.tree() == target_tree

    def test_get_component(self):
        parent = GroupingComponent("Parent", {}, {})
        child1 = GroupingComponent("Child1", {}, {}, parent=parent)
        child2 = GroupingComponent("Child2", {}, {}, parent=parent)
        grandchild = GroupingComponent("Grandchild", {}, {}, parent=child1)

        assert grandchild.get_component("Child2") is child2

    def test_get_component_multiple(self):
        parent = GroupingComponent("Parent", {}, {})
        child1 = GroupingComponent("Child", {}, {}, parent=parent)
        child2 = GroupingComponent("Child", {}, {}, parent=parent)
        grandchild = GroupingComponent("Grandchild", {}, {}, parent=child1)

        components = grandchild.get_component("Child", first=False)
        assert len(components) == 2
        assert components[0] is not components[1]
        assert components[0].parent == components[1].parent

    def test_get_component_missing(self):
        parent = GroupingComponent("Parent", {}, {})
        child = GroupingComponent("Child", {}, {}, parent=parent)
        grandchild = GroupingComponent("Grandchild", {}, {}, parent=child)

        component = grandchild.get_component("Banana")
        assert component is None

    def test_copy(self):
        parent = GroupingComponent("Parent", {}, {})
        child = GroupingComponent("Child", {}, {}, parent=parent)
        grandchild = GroupingComponent("Grandchild", {}, {}, parent=child)

        component = child.copy()
        assert component is not child
        assert component.parent is not parent
        assert component.name == child.name
        assert component.parent.name == parent.name
        assert all([child_.name == grandchild.name for child_ in child.children])


class TestPhysicalComponent:
    """
    Tests for the PhysicalComponent class.
    """

    def test_shape(self):
        component = PhysicalComponent("Dummy", {}, {}, shape="A shape")
        assert component.shape == "A shape"

    def test_material_default(self):
        component = PhysicalComponent("Dummy", {}, {}, shape="A shape")
        assert component.material is None

    def test_material(self):
        component = PhysicalComponent(
            "Dummy", {}, {}, shape="A shape", material="A material"
        )
        assert component.material == "A material"


class TestMagneticComponent:
    """
    Tests for the MagneticComponent class.
    """

    def test_shape(self):
        component = MagneticComponent("Dummy", {}, {}, shape="A shape")
        assert component.shape == "A shape"

    def test_conductor_default(self):
        component = MagneticComponent("Dummy", {}, {}, shape="A shape")
        assert component.material is None

    def test_conductor(self):
        component = MagneticComponent(
            "Dummy", {}, {}, shape="A shape", conductor="A conductor"
        )
        assert component.conductor == "A conductor"


class TestDefineSystemClass:
    """
    Tests for dynamic system class setting.
    """

    class DummyComponent(GroupingComponent):
        """
        Test Component with a divertor sub-system.
        """

        div: DummyDivertor

        def __init__(self, inputs):
            self.inputs = inputs

            self._generate_subsystem_classes(self.inputs)

    class DummySubclassComponent(DummyComponent):
        """
        Test Component that's a child of a system with a divertor sub-system.

        Checks that the new annotations don't break the parent class's annotations.
        """

        inputs: dict

    class DummySubclassComponentExtraSub(DummyComponent):
        """
        Test Component that's a child of a system with a divertor sub-system and an
        additional sub system.

        Checks that the new annotations don't break the parent class's annotations.
        """

        bb: DummyBreedingBlanket

    class DummySecondComponent(GroupingComponent):
        """
        Test Component that also defines a div subsystem but with a different type.
        """

        div: DummyDivertorProfile

        def __init__(self, inputs):
            self.inputs = inputs

            self._generate_subsystem_classes(self.inputs)

    class NewDivertor(DummyDivertor):
        """
        A new divertor class.
        """

        pass

    class NewBreedingBlanket(DummyBreedingBlanket):
        """
        A new blanket class.
        """

        pass

    @pytest.mark.parametrize(
        "system,inputs,div_class",
        [
            (DummyComponent, {}, DummyDivertor),
            (DummyComponent, {"div_class_name": "NewDivertor"}, NewDivertor),
            (DummySecondComponent, {}, DummyDivertorProfile),
        ],
    )
    def test_get_subsystem_class(self, system, inputs, div_class):
        """
        Test the sub-system behaviour - get the class using the variable name as key.
        """
        my_system: Component = system(inputs)
        assert my_system.get_subsystem_class("div") is div_class

    @pytest.mark.parametrize(
        "inputs,div_class",
        [
            ({}, DummyDivertor),
            ({"div_class_name": "NewDivertor"}, NewDivertor),
        ],
    )
    def test_get_subsystem_class_from_child(self, inputs, div_class):
        """
        Test the sub-system behaviour - get the class using the variable name as key.
        """
        my_system = TestDefineSystemClass.DummySubclassComponent(inputs)
        assert my_system.get_subsystem_class("div") is div_class

    @pytest.mark.parametrize(
        "inputs,div_class,bb_class",
        [
            ({}, DummyDivertor, DummyBreedingBlanket),
            (
                {"div_class_name": "NewDivertor", "bb_class_name": "NewBreedingBlanket"},
                NewDivertor,
                NewBreedingBlanket,
            ),
        ],
    )
    def test_get_subsystem_class_from_child_extra_sub(self, inputs, div_class, bb_class):
        """
        Test the sub-system behaviour - get the class using the variable name as key.
        """
        my_system = TestDefineSystemClass.DummySubclassComponent(inputs)
        assert my_system.get_subsystem_class("div") is div_class

    def test_error_class_not_defined(self):
        """
        Test a non-existent class in the build config raises an error.
        """
        with pytest.raises(ComponentError):
            TestDefineSystemClass.DummyComponent({"div_class_name": "AnotherDivertor"})

    def test_error_wrong_subclass(self):
        """
        Test providing class that does not match the system raises an error.
        """
        with pytest.raises(ComponentError):
            TestDefineSystemClass.DummyComponent({"div_class_name": "BreedingBlanket"})

    def test_error_system_not_defined(self):
        """
        Test that getting a system that doesn't exist raises an error.
        """
        my_system = TestDefineSystemClass.DummyComponent({})
        with pytest.raises(ComponentError):
            my_system.get_subsystem_class("DIVERTOR")


if __name__ == "__main__":
    pytest.main([__file__])

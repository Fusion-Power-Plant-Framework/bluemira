# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

import pytest
from typing import Type

from BLUEPRINT.base.baseclass import ReactorSystem
from BLUEPRINT.base.error import BLUEPRINTError
from BLUEPRINT.nova.firstwall import DivertorProfile
from BLUEPRINT.systems.blanket import BreedingBlanket
from BLUEPRINT.systems.divertor import Divertor


class TestBaseClass:
    """
    Tests for the base class functionality.
    """

    class MyReactorSystem(ReactorSystem):
        """
        Test parent ReactorSystem.
        """

        pass

    class MyChildReactorSystem(MyReactorSystem):
        """
        Test child ReactorSystem.
        """

        pass

    @pytest.mark.parametrize(
        "class_name,the_class",
        [
            ("MyReactorSystem", MyReactorSystem),
            ("MyChildReactorSystem", MyChildReactorSystem),
        ],
    )
    def test_get_class(self, class_name, the_class):
        """
        Test that reactor system classes can be retrieved
        """
        my_class = TestBaseClass.MyReactorSystem.get_class(class_name)
        assert my_class is not None
        assert my_class == the_class

    @pytest.mark.parametrize(
        "the_class,name",
        [
            (MyReactorSystem, "NotAClass"),
            (MyChildReactorSystem, "MyReactorSystem"),
        ],
    )
    def test_get_class_fail(self, the_class, name):
        with pytest.raises(BLUEPRINTError):
            TestBaseClass.MyReactorSystem.get_class("NotAClass")

    def test_avoid_duplicate_classes(self):
        with pytest.raises(BLUEPRINTError):

            class MyReactorSystem(ReactorSystem):
                pass


class TestDefineSystemClass:
    """
    Tests for dynamic system class setting.
    """

    class DummyReactorSystem(ReactorSystem):
        """
        Test ReactorSystem with a divertor sub-system.
        """

        div: Type[Divertor]

        def __init__(self, inputs):
            self.inputs = inputs

            self._generate_subsystem_classes(self.inputs)

    class DummyChildReactorSystem(DummyReactorSystem):
        """
        Test ReactorSystem that's a child of a system with a divertor sub-system.

        Checks that the new annotations don't break the parent class's annotations.
        """

        inputs: dict

    class DummyChildReactorSystemExtraSub(DummyReactorSystem):
        """
        Test ReactorSystem that's a child of a system with a divertor sub-system and
        an additional sub system.

        Checks that the new annotations don't break the parent class's annotations.
        """

        bb: Type[BreedingBlanket]

    class DummySecondReactorSystem(ReactorSystem):
        """
        Test ReactorSystem that also defines a div subsystem but with a different type.
        """

        div: Type[DivertorProfile]

        def __init__(self, inputs):
            self.inputs = inputs

            self._generate_subsystem_classes(self.inputs)

    class NewDivertor(Divertor):
        """
        A new divertor class.
        """

        pass

    class NewBreedingBlanket(BreedingBlanket):
        """
        A new blanket class.
        """

        pass

    @pytest.mark.parametrize(
        "system,inputs,div_class",
        [
            (DummyReactorSystem, {}, Divertor),
            (DummyReactorSystem, {"div_class_name": "NewDivertor"}, NewDivertor),
            (DummySecondReactorSystem, {}, DivertorProfile),
        ],
    )
    def test_get_subsystem_class(self, system, inputs, div_class):
        """
        Test the sub-system behaviour - get the class using the variable name as key.
        """
        my_system = system(inputs)
        assert my_system.get_subsystem_class("div") is div_class

    @pytest.mark.parametrize(
        "inputs,div_class",
        [
            ({}, Divertor),
            ({"div_class_name": "NewDivertor"}, NewDivertor),
        ],
    )
    def test_get_subsystem_class_from_child(self, inputs, div_class):
        """
        Test the sub-system behaviour - get the class using the variable name as key.
        """
        my_system = TestDefineSystemClass.DummyChildReactorSystem(inputs)
        assert my_system.get_subsystem_class("div") is div_class

    @pytest.mark.parametrize(
        "inputs,div_class,bb_class",
        [
            ({}, Divertor, BreedingBlanket),
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
        my_system = TestDefineSystemClass.DummyChildReactorSystem(inputs)
        assert my_system.get_subsystem_class("div") is div_class

    def test_error_class_not_defined(self):
        """
        Test a non-existent class in the build config raises an error.
        """
        with pytest.raises(BLUEPRINTError):
            TestDefineSystemClass.DummyReactorSystem(
                {"div_class_name": "AnotherDivertor"}
            )

    def test_error_wrong_subclass(self):
        """
        Test providing class that does not match the system raises an error.
        """
        with pytest.raises(BLUEPRINTError):
            TestDefineSystemClass.DummyReactorSystem(
                {"div_class_name": "BreedingBlanket"}
            )

    def test_error_system_not_defined(self):
        """
        Test that getting a system that doesn't exist raises an error.
        """
        my_system = TestDefineSystemClass.DummyReactorSystem({})
        with pytest.raises(BLUEPRINTError):
            my_system.get_subsystem_class("DIVERTOR")


if __name__ == "__main__":
    pytest.main([__file__])
